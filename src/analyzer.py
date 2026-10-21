"""
Tactical Analysis Module

This module performs match analysis on top of detection, tracking and team assignment:
- Ball possession (who controls the ball, and the running Possession % per team)
- Pass detection (possession transferred between two players of the same team)

Possession logic:
1. For every frame where the ball is detected, compute the Euclidean distance between
   the ball center and the feet (bottom-center of the bounding box) of every player.
2. The nearest player controls the ball if the distance is below a threshold. The
   threshold is relative to that player's bounding-box height, which makes it
   perspective-aware (far-away players are smaller in pixels).
3. A new controller is only confirmed after being observed in control of the ball in
   `min_hold_frames` frames with no other player in control in between. This filters
   out the ball merely passing close to a player and isolated false ball detections.
4. Possession % counts frames per team. Between touches (ball in flight, or not
   detected) possession stays with the last team in control, as in broadcast stats.

Who can have the ball:
Only tracks labelled 'team_a' or 'team_b' are candidates. Officials (label 'referee')
are drawn in the video but never considered, so the ball passing near the referee does
not change possession or create passes. Goalkeepers are labelled with the team they
defend (see TeamAssigner.finalize_roles), so their touches count for that team.
The ball position used here is the cleaned trajectory (outliers removed, gaps < 20
frames interpolated, smoothed; see tracker.BallTrajectory).

Pass logic:
- When the confirmed controller changes from player X to player Y:
  same team  -> successful pass X -> Y
  other team -> turnover (interception / lost ball)

Physical metrics (PlayerKinematics, TeamShape) use the camera-compensated pitch
coordinates in meters from tactical_map:
- Speed and distance per track, from smoothed positions (see PlayerKinematics).
- Team shape per frame: convex hull of the visible outfield players, its area (m^2),
  width and depth (see TeamShape).
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class PassEvent:
    """
    A completed pass between two players of the same team.
    """
    frame: int
    from_player_id: int
    to_player_id: int
    team: str
    from_pos: Tuple[float, float]  # video pixels (feet of passer when he last had the ball)
    to_pos: Tuple[float, float]  # video pixels (feet of receiver)
    distance_px: float


@dataclass
class TurnoverEvent:
    """
    Possession lost to the other team.
    """
    frame: int
    from_player_id: int
    to_player_id: int
    from_team: str
    to_team: str


class TacticalAnalyzer:
    """
    Computes possession and passes from tracked players and the ball.
    """

    def __init__(self, fps: float = 30.0, control_ratio: float = 0.6, min_hold_frames: int = 3,
                 player_aliases: Optional[Dict[int, str]] = None,
                 still_px_speed: float = 60.0, still_seconds: float = 1.0,
                 pitch_dims: Tuple[float, float] = (105.0, 68.0), dead_zone_m: float = 2.0):
        """
        Initialize the tactical analyzer.

        Args:
            fps: Video frame rate
            control_ratio: Ball-to-feet distance threshold, as a fraction of the player's
                           bounding-box height (0.6 * ~1.8 m tall player ~= 1.1 m)
            min_hold_frames: Number of in-control observations (uninterrupted by another
                             player) before a new player is confirmed as the ball holder
            player_aliases: track_id -> identity for tracks known to be the same person, e.g.
                            all goalkeeper tracks of a team (the tracker can split one keeper
                            into several IDs, which would otherwise create passes to himself)
            still_px_speed: Ball speed (pixels/second on screen) below which the ball counts
                            as still. Measured in pixel space on purpose: the per-frame
                            homography jitters, so a truly stationary ball can look like it
                            moves ~1 m/s in pitch coords - a static image cannot lie.
            still_seconds: How long the ball must stay still before possession counting pauses
                           (dead-ball situations: a corner-kick ball sitting at the flag does
                           not inflate possession for the player standing next to it)
            pitch_dims: (length, width) of the pitch in meters
            dead_zone_m: Distance from a touchline/goal line within which a still ball counts
                         as out of play. A still ball further inside the pitch is a player
                         legally standing on it (tempo control) - possession keeps counting.
        """
        self.fps = fps
        self.control_ratio = control_ratio
        self.min_hold_frames = min_hold_frames
        self.player_aliases = player_aliases or {}
        self.still_px_speed = still_px_speed
        self.still_seconds = still_seconds
        self.pitch_dims = pitch_dims
        self.dead_zone_m = dead_zone_m

        self.frames_by_team = {'team_a': 0, 'team_b': 0}
        self.current_holder: Optional[int] = None
        self.current_team: Optional[str] = None
        self.holder_last_pos: Optional[Tuple[float, float]] = None
        self._candidate: Optional[int] = None
        self._candidate_frames = 0
        self._still_frames = 0
        self.ball_in_play = True
        self.dead_ball_frames = 0

        self.pass_events: List[PassEvent] = []
        self.turnover_events: List[TurnoverEvent] = []

    def find_nearest_player(self, ball_position: Tuple[float, float], tracks: List,
                            team_assignments: Dict[int, str]) -> Tuple[Optional[object], float]:
        """
        Find the outfield player (team_a/team_b) whose feet are nearest to the ball.

        Returns:
            (track, distance in pixels), or (None, inf) if there are no players
        """
        best, best_dist = None, np.inf
        for track in tracks:
            if team_assignments.get(track.track_id) not in self.frames_by_team:
                continue
            fx, fy = track.foot_position
            dist = float(np.hypot(ball_position[0] - fx, ball_position[1] - fy))
            if dist < best_dist:
                best, best_dist = track, dist
        return best, best_dist

    def _ball_near_boundary(self, pitch_pos: Tuple[float, float]) -> bool:
        """True if the ball's pitch position is in the dead zone or off the pitch."""
        x, y = pitch_pos
        L, W = self.pitch_dims
        m = self.dead_zone_m
        return x < m or x > L - m or y < m or y > W - m  # covers outside bounds too

    def update(self, frame_idx: int, ball_position: Optional[Tuple[float, float]],
               tracks: List, team_assignments: Dict[int, str],
               ball_px_speed: Optional[float] = None,
               ball_pitch_pos: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Update possession and pass detection for the current frame.

        Args:
            frame_idx: Current frame number
            ball_position: Ball center in video pixels (None if not detected)
            tracks: Active Track objects
            team_assignments: track_id -> team label
            ball_px_speed: Ball speed in pixels/second (None = unknown; treated as in play)
            ball_pitch_pos: Ball position in pitch meters (None = unknown; dead-ball
                            gating requires both speed AND position)

        Returns:
            Dict with 'holder_id', 'current_team', 'in_control' (a player is touching
            the ball this frame), 'new_pass' (PassEvent or None), 'ball_in_play' and
            'possession_stats'
        """
        # Dead-ball detection needs BOTH: a still ball for still_seconds AND the ball
        # near/outside a pitch boundary (corner flag, goal line, touchline). A still
        # ball inside the pitch is a player standing on it - play continues.
        if (ball_px_speed is not None and ball_pitch_pos is not None
                and ball_px_speed < self.still_px_speed
                and self._ball_near_boundary(ball_pitch_pos)):
            self._still_frames += 1
        else:
            self._still_frames = 0
        self.ball_in_play = self._still_frames < self.still_seconds * self.fps

        new_pass, in_control = None, False

        if ball_position is not None:
            nearest, dist = self.find_nearest_player(ball_position, tracks, team_assignments)
            if nearest is not None and dist <= self.control_ratio * nearest.height:
                in_control = True
                if self._same_player(nearest.track_id, self.current_holder):
                    self._candidate, self._candidate_frames = None, 0
                    self.current_holder, self.holder_last_pos = nearest.track_id, nearest.foot_position
                else:
                    if self._same_player(nearest.track_id, self._candidate):
                        self._candidate_frames += 1
                    else:
                        self._candidate, self._candidate_frames = nearest.track_id, 1
                    if self._candidate_frames >= self.min_hold_frames:
                        new_pass = self._change_holder(frame_idx, nearest, team_assignments[nearest.track_id])

        if self.ball_in_play:
            if self.current_team is not None:
                self.frames_by_team[self.current_team] += 1
        else:
            self.dead_ball_frames += 1

        return {
            'holder_id': self.current_holder,
            'current_team': self.current_team,
            'in_control': in_control,
            'new_pass': new_pass,
            'ball_in_play': self.ball_in_play,
            'possession_stats': self.get_team_possession_stats(),
        }

    def _same_player(self, a: Optional[int], b: Optional[int]) -> bool:
        if a is None or b is None:
            return False
        return self.player_aliases.get(a, a) == self.player_aliases.get(b, b)

    def _change_holder(self, frame_idx: int, new_holder, new_team: str) -> Optional[PassEvent]:
        """
        Confirm a new ball holder and classify the transfer as a pass or a turnover.
        """
        event = None
        # Holder state still updates during a dead ball, but no pass/turnover is
        # recorded - otherwise swapping the corner taker or moving a placed ball
        # would fabricate events. The kick itself registers once play resumes.
        if self.current_holder is not None and self.ball_in_play:
            if new_team == self.current_team:
                to_pos = new_holder.foot_position
                event = PassEvent(
                    frame=frame_idx,
                    from_player_id=self.current_holder,
                    to_player_id=new_holder.track_id,
                    team=new_team,
                    from_pos=self.holder_last_pos,
                    to_pos=to_pos,
                    distance_px=float(np.hypot(to_pos[0] - self.holder_last_pos[0],
                                               to_pos[1] - self.holder_last_pos[1])),
                )
                self.pass_events.append(event)
            else:
                self.turnover_events.append(TurnoverEvent(
                    frame=frame_idx, from_player_id=self.current_holder, to_player_id=new_holder.track_id,
                    from_team=self.current_team, to_team=new_team))

        self.current_holder, self.current_team = new_holder.track_id, new_team
        self.holder_last_pos = new_holder.foot_position
        self._candidate, self._candidate_frames = None, 0
        return event

    def get_team_possession_stats(self) -> Dict[str, float]:
        """
        Possession percentage per team (frames in possession / frames with a known possessor).
        """
        total = sum(self.frames_by_team.values())
        if total == 0:
            return {'team_a': 0.0, 'team_b': 0.0}
        return {team: 100.0 * n / total for team, n in self.frames_by_team.items()}

    def get_pass_stats(self) -> Dict[str, Dict]:
        """
        Pass and turnover counts per team.
        """
        return {
            team: {
                'passes': sum(p.team == team for p in self.pass_events),
                'turnovers': sum(t.from_team == team for t in self.turnover_events),
            }
            for team in self.frames_by_team
        }

    def export(self) -> Dict:
        """
        All statistics and events as a JSON-serializable dict.
        """
        return {
            'possession_percent': self.get_team_possession_stats(),
            'possession_frames': dict(self.frames_by_team),
            'dead_ball_frames': self.dead_ball_frames,
            'dead_ball_seconds': round(self.dead_ball_frames / self.fps, 1),
            'pass_stats': self.get_pass_stats(),
            'passes': [asdict(p) for p in self.pass_events],
            'turnovers': [asdict(t) for t in self.turnover_events],
        }


class PlayerKinematics:
    """
    Speed (km/h) and distance covered (m) per track from pitch positions over time.

    Raw homography positions jitter by tens of centimeters per frame (detector box jitter,
    ~1 m calibration error), which would add huge fake distances at 30 FPS. Therefore:
    1. Each track is split into continuous runs (gaps <= `max_gap` frames, same camera
       segment); runs shorter than `min_run` frames are ignored.
    2. Positions are de-spiked with a centered rolling median (`median_window`) and then
       smoothed with a centered rolling mean (`mean_window`, ~1 s by default).
    3. Speed = frame-to-frame displacement of the smoothed track x FPS, smoothed again with a
       centered rolling mean (`speed_window`, ~1 s by default).
    4. Steps faster than `max_speed_kmh` (above elite sprint speed, ~36 km/h) are treated as
       tracking/projection errors (e.g. an ID switch or camera-tracking noise during a
       fast pan) and excluded from distance and speed.
    """

    def __init__(self, fps: float, max_gap: int = 3, min_run: int = 15, median_window: int = 5,
                 mean_window: Optional[int] = None, speed_window: Optional[int] = None,
                 max_speed_kmh: float = 40.0):
        self.fps = fps
        self.max_gap = max_gap
        self.min_run = min_run
        self.median_window = median_window
        self.mean_window = mean_window or max(3, int(round(fps)))
        self.speed_window = speed_window or max(3, int(round(fps)))
        self.max_speed_kmh = max_speed_kmh
        self.speed: Dict[int, Dict[int, float]] = {}  # track_id -> frame -> km/h
        self.cum_distance: Dict[int, Dict[int, float]] = {}  # track_id -> frame -> meters covered so far
        self.summary: Dict[int, Dict] = {}

    def compute(self, positions: Dict[int, Dict[int, Optional[Tuple[float, float]]]],
                segment_of_frame: List[int], inside_pitch) -> Dict[int, Dict]:
        """
        Args:
            positions: track_id -> frame -> pitch position in meters (None if unknown)
            segment_of_frame: Camera segment index per frame (runs never cross a cut)
            inside_pitch: Callable(point) -> bool, positions outside the pitch are dropped

        Returns:
            track_id -> {'distance_m', 'max_speed_kmh', 'mean_speed_kmh', 'seconds'}
        """
        for tid, per_frame in positions.items():
            pts = sorted((f, p) for f, p in per_frame.items() if p is not None and inside_pitch(p))
            if not pts:
                continue
            df = pd.DataFrame([(f, p[0], p[1]) for f, p in pts], columns=['frame', 'x', 'y'])
            new_run = (df['frame'].diff() > self.max_gap) | (
                df['frame'].map(lambda f: segment_of_frame[f]).diff() != 0)
            df['run'] = new_run.cumsum()

            speeds, cum_dist, distance, seconds = {}, {}, 0.0, 0.0
            for _, run in df.groupby('run'):
                if len(run) < self.min_run:
                    continue
                run = run.set_index('frame').reindex(range(run['frame'].min(), run['frame'].max() + 1))
                xy = run[['x', 'y']].interpolate(limit_area='inside')
                xy = xy.rolling(self.median_window, center=True, min_periods=1).median()
                xy = xy.rolling(self.mean_window, center=True, min_periods=1).mean()
                step = np.hypot(xy['x'].diff(), xy['y'].diff())
                kmh = (step * self.fps * 3.6).rolling(self.speed_window, center=True, min_periods=1).mean()
                valid = kmh <= self.max_speed_kmh
                run_distance = float(step[valid].sum())
                seconds += len(run) / self.fps
                speeds.update({int(f): float(v) for f, v in kmh[valid].dropna().items()})
                run_cum = step.where(valid, 0).fillna(0).cumsum() + distance
                cum_dist.update({int(f): float(v) for f, v in run_cum.items()})
                distance += run_distance

            self.cum_distance[tid] = cum_dist
            if speeds:
                v = np.array(list(speeds.values()))
                self.speed[tid] = speeds
                self.summary[tid] = {'distance_m': round(distance, 1), 'max_speed_kmh': round(float(v.max()), 1),
                                     'mean_speed_kmh': round(float(v.mean()), 1), 'seconds': round(seconds, 1)}
        return self.summary

    def speed_at(self, track_id: int, frame_idx: int) -> Optional[float]:
        return self.speed.get(track_id, {}).get(frame_idx)


class TeamShape:
    """
    Team compactness per frame: convex hull of a team's visible players (goalkeepers
    included, officials excluded) on the pitch (meters), with area (m^2), width (extent
    along the goal line) and depth (extent along the touchline).

    Only players inside the camera view are known, so the hull describes the visible part
    of the team; frames with fewer than `min_players` visible players are skipped.
    """

    def __init__(self, min_players: int = 5):
        self.min_players = min_players
        self.history: Dict[str, List[Dict]] = {'team_a': [], 'team_b': []}

    def update(self, frame_idx: int, players: List[Tuple[str, Tuple[float, float]]]) -> Dict[str, np.ndarray]:
        """
        Args:
            players: (team, pitch position) of visible team players (officials excluded)

        Returns:
            team -> hull vertices in meters (only teams with enough visible players)
        """
        hulls = {}
        for team in self.history:
            pts = np.array([p for t, p in players if t == team], dtype=np.float32)
            if len(pts) < self.min_players:
                continue
            hull = cv2.convexHull(pts).reshape(-1, 2)
            hulls[team] = hull
            self.history[team].append({'frame': frame_idx, 'players': len(pts),
                                       'area_m2': float(cv2.contourArea(hull)),
                                       'width_m': float(np.ptp(pts[:, 1])), 'depth_m': float(np.ptp(pts[:, 0]))})
        return hulls

    def summary(self) -> Dict[str, Dict]:
        out = {}
        for team, rows in self.history.items():
            if rows:
                df = pd.DataFrame(rows)
                out[team] = {'frames': len(df), 'mean_visible_players': round(df['players'].mean(), 1),
                             'mean_area_m2': round(df['area_m2'].mean(), 1),
                             'median_area_m2': round(df['area_m2'].median(), 1),
                             'mean_width_m': round(df['width_m'].mean(), 1),
                             'mean_depth_m': round(df['depth_m'].mean(), 1)}
        return out
