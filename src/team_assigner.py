"""
Team Assignment Module

This module analyzes player jersey colors and assigns every tracked person to
Team A, Team B, a goalkeeper of one of the teams, or an official (referee / assistant
referee). It works offline, on the whole clip (collect during pass 1, fit afterwards).

Jersey colors and team kits
---------------------------
1. Jersey color of a detection = median color of the non-grass pixels in the torso region
   of its bounding box, in CIE Lab space (perceptually uniform).
2. Lightness is down-weighted (LIGHTNESS_WEIGHT) in all color distances: sun and shadow
   mostly change a kit's lightness, whereas the kits that must be told apart (outfield
   kits vs hi-vis officials vs goalkeepers) differ in hue/chroma. On test_match.mp4 the
   dark kit measures L~85 in sunlight but L~72 in shadow.
3. The two outfield kits are found with a trimmed KMeans (k=2) on samples from the WHOLE
   clip: fit, drop samples far from both centers (officials, goalkeepers), refit. Fitting
   on the first frames only failed when the camera moved into shadow.
4. A detection far from both kits (> outlier_factor x median in-kit distance) is a color
   outlier ('referee' label until roles are resolved).
5. Per track, labels are smoothed with a centered sliding-window majority vote, which
   removes flicker (occlusion, motion blur) but still follows a tracker ID switch.

Referees vs goalkeepers (resolve_roles)
---------------------------------------
Officials and goalkeepers both wear kits that differ from the outfield kits, so both are
color outliers. They are separated spatially, in pitch coordinates:
- Goalkeeper: an outlier track (or a track whose kit could not be measured at all, e.g. a
  lime kit inside the grass color range) that, in most of the frames it is seen, is the
  deepest person on its half (closest to that goal line) and does not live on a touchline. It is
  assigned to the team defending that goal (the team whose players are, on median, closer
  to that goal). A penalty-area rule would be too strict: modern "sweeper keepers" often
  stand 20 m out when their team has the ball.
- Assistant referees run along the touchlines, hence the touchline exclusion.
- Officials: every other outlier track. Their shared kit color is learned from these
  tracks, and a goalkeeper candidate wearing that kit is kept as an official.
Officials are still tracked and drawn, but the analyzer ignores them.

Color override of the model's 'referee' class (Phase 6)
-----------------------------------------------------
The fine-tuned detector has a 'referee' class (mAP50 ~0.96), but it still labels some
players - typically standing still near the touchline - as referees. The jersey color
is the more reliable evidence, so _apply_color_overrides() runs at the start of
resolve_roles(): a track whose MEDIAN jersey color sits well inside a team kit
(< strong_factor x the median in-kit distance) cannot be an official - its labels are
rewritten to that team whatever the model predicted. The model's role votes are still
used as evidence where color is ambiguous: a track with unmeasurable kit color (e.g.
inside the grass HSV range) and a clear YOLO 'referee' majority stays an official,
while a YOLO 'goalkeeper' majority is enough for a goalkeeper candidate on its own.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans
from collections import Counter, defaultdict

LIGHTNESS_WEIGHT = 0.5


class TeamAssigner:
    """
    Assigns tracked persons to teams and roles based on jersey color and position.
    """

    TEAMS = ('team_a', 'team_b')

    def __init__(self, outlier_factor: float = 2.5, min_outlier_distance: float = 12.0,
                 vote_window: int = 45, max_fit_samples: int = 20000):
        """
        Args:
            outlier_factor: A color farther than outlier_factor x (median in-kit distance)
                            from both kits is a color outlier
            min_outlier_distance: Lower bound for the outlier threshold (weighted Lab units)
            vote_window: Length (frames) of the centered majority-vote window per track
            max_fit_samples: Random subsample size used to fit the kits
        """
        self.outlier_factor = outlier_factor
        self.min_outlier_distance = min_outlier_distance
        self.vote_window = vote_window
        self.max_fit_samples = max_fit_samples

        self.samples: Dict[int, List[Tuple[int, Optional[np.ndarray], str]]] = defaultdict(list)  # tid -> (frame, Lab, yolo role)
        self.kmeans: Optional[KMeans] = None
        self.team_order: List[int] = []  # cluster index for team_a, team_b
        self.outlier_threshold = np.inf
        self.inlier_scale = 1.0  # Median weighted distance of in-kit samples (reference color scale)
        self.track_labels: Dict[int, Dict[int, str]] = {}  # tid -> frame -> smoothed label
        self.goalkeepers: Dict[int, str] = {}
        self.officials: set = set()
        self.official_color: Optional[np.ndarray] = None
        self.role_report: Dict[int, Dict] = {}  # Per outlier track: evidence used for the role decision
        self.yolo_role_votes: Dict[int, Counter] = {}
        self.overrides: Dict[int, str] = {}  # tid -> team: YOLO 'referee' overridden by kit color

        self.green_lower = np.array([30, 40, 40])
        self.green_upper = np.array([90, 255, 255])

    @staticmethod
    def _weighted(lab: np.ndarray) -> np.ndarray:
        return lab * np.array([LIGHTNESS_WEIGHT, 1.0, 1.0], dtype=np.float32)

    # ------------------------------------------------------------------ colors
    def extract_jersey_color(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract the jersey color of a player as a Lab vector.

        Args:
            frame: BGR video frame
            bbox: Player bounding box [x1, y1, x2, y2]

        Returns:
            Median Lab color of the torso's non-grass pixels, or None if not enough pixels
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        # Torso: skip the head (top 15%) and legs (bottom 50%), and the side margins
        tx1, tx2 = x1 + int(0.25 * w), x2 - int(0.25 * w)
        ty1, ty2 = y1 + int(0.15 * h), y1 + int(0.5 * h)
        H, W = frame.shape[:2]
        tx1, ty1, tx2, ty2 = max(0, tx1), max(0, ty1), min(W, tx2), min(H, ty2)
        if tx2 - tx1 < 2 or ty2 - ty1 < 2:
            return None

        torso = frame[ty1:ty2, tx1:tx2]
        not_grass = cv2.inRange(cv2.cvtColor(torso, cv2.COLOR_BGR2HSV), self.green_lower, self.green_upper) == 0
        pixels = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)[not_grass]
        if len(pixels) < 10:
            # Includes green/lime kits that fall inside the grass range (e.g. some goalkeepers).
            # Using the whole torso instead was tested and failed: grass-dominated crops of small
            # players then formed their own "kit" cluster. These tracks stay 'unknown' and can
            # still be found as goalkeepers by position (resolve_roles).
            return None
        return np.median(pixels, axis=0).astype(np.float32)

    def collect(self, frame_idx: int, frame: np.ndarray, tracks: List) -> None:
        """
        Pass 1: store the jersey color and the detector's role class of every tracked person.
        """
        for t in tracks:
            self.samples[t.track_id].append(
                (frame_idx, self.extract_jersey_color(frame, t.bbox), getattr(t, 'role', 'person')))

    # ------------------------------------------------------------------ kits
    def fit(self) -> None:
        """
        Fit the two outfield kits on the whole clip with a trimmed KMeans and label every
        detection (smoothed per track).
        """
        outfield = [c for s in self.samples.values() for _, c, r in s
                    if c is not None and r not in ('referee', 'goalkeeper')]
        all_samples = [c for s in self.samples.values() for _, c, _ in s if c is not None]
        X = np.array(outfield if len(outfield) >= 10 else all_samples, dtype=np.float32)
        if len(X) < 10:
            raise RuntimeError("Not enough jersey samples to fit the team kits")
        rng = np.random.default_rng(42)
        if len(X) > self.max_fit_samples:
            X = X[rng.choice(len(X), self.max_fit_samples, replace=False)]
        Xw = self._weighted(X)

        inliers = np.ones(len(Xw), bool)
        for _ in range(3):  # Trimmed KMeans: refit without the outliers (officials, goalkeepers)
            self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(Xw[inliers])
            dists = np.min(self.kmeans.transform(Xw), axis=1)
            self.outlier_threshold = max(self.outlier_factor * float(np.median(dists[inliers])),
                                         self.min_outlier_distance)
            inliers = dists <= self.outlier_threshold
        # Deterministic naming: team_a = darker kit, team_b = lighter kit
        self.team_order = list(np.argsort(self.kmeans.cluster_centers_[:, 0]))
        self.inlier_scale = float(np.median(dists[inliers])) if inliers.any() else 1.0

        print(f"  - Team kits fitted on {len(X)} samples ({(~inliers).mean():.1%} color outliers). "
              f"Team A (BGR) {self.team_color_bgr('team_a')}, Team B (BGR) {self.team_color_bgr('team_b')}")

        self.yolo_role_votes = {tid: Counter(r for _, _, r in seq) for tid, seq in self.samples.items()}
        half = self.vote_window // 2
        for tid, seq in self.samples.items():
            raw = [None if c is None else self.classify_color(c) for _, c, _ in seq]
            labels = {}
            for i, (f, _, _) in enumerate(seq):
                window = [l for l in raw[max(0, i - half):i + half + 1] if l is not None]
                labels[f] = Counter(window).most_common(1)[0][0] if window else 'unknown'
            self.track_labels[tid] = labels

    def team_color_bgr(self, team: str) -> tuple:
        """
        Representative BGR color of a team's kit (for logging/visualization).
        """
        lab = self.kmeans.cluster_centers_[self.team_order[self.TEAMS.index(team)]] / np.array([LIGHTNESS_WEIGHT, 1, 1])
        return self._lab_to_bgr(lab)

    @staticmethod
    def _lab_to_bgr(lab: np.ndarray) -> tuple:
        bgr = cv2.cvtColor(np.uint8([[np.clip(lab, 0, 255)]]), cv2.COLOR_LAB2BGR)[0, 0]
        return tuple(int(c) for c in bgr)

    def classify_color(self, color: np.ndarray) -> str:
        """
        Classify a single Lab jersey color into 'team_a', 'team_b' or 'referee' (color outlier).
        """
        dists = self.kmeans.transform(self._weighted(color).reshape(1, -1))[0]
        cluster = int(np.argmin(dists))
        if dists[cluster] > self.outlier_threshold:
            return 'referee'
        return self.TEAMS[self.team_order.index(cluster)]

    def labels_for_frame(self, frame_idx: int, track_ids: List[int]) -> Dict[int, str]:
        """
        Final label per track for one frame (after fit() and resolve_roles()):
        'team_a' | 'team_b' | 'referee' | 'unknown'. Goalkeepers carry their team label.
        """
        out = {}
        for tid in track_ids:
            if tid in self.goalkeepers:
                out[tid] = self.goalkeepers[tid]
            elif tid in self.officials:
                out[tid] = 'referee'
            else:
                out[tid] = self.track_labels.get(tid, {}).get(frame_idx, 'unknown')
        return out

    # ------------------------------------------------------------------ roles
    def _yolo_majority(self, tid: int) -> Optional[str]:
        """The detector's most frequent role prediction for a track ('person' = COCO, ignored)."""
        votes = {r: n for r, n in self.yolo_role_votes.get(tid, {}).items() if r != 'person'}
        return Counter(votes).most_common(1)[0][0] if votes else None

    def _apply_color_overrides(self, strong_factor: float = 1.5, min_fraction: float = 0.5):
        """
        Overrule 'referee' decisions with jersey color evidence. The fine-tuned model
        sometimes calls still or touchline players 'referee'; a track whose MEDIAN kit
        color is clearly inside a team kit (<= strong_factor x the median in-kit distance,
        and >= min_fraction of its samples matching that team) is rewritten to that team,
        whether the 'referee' came from the model or from a color-outlier label.
        """
        self.overrides = {}
        for tid, seq in self.samples.items():
            labels = self.track_labels.get(tid, {})
            majority_label = Counter(labels.values()).most_common(1)[0][0] if labels else 'unknown'
            if majority_label in self.TEAMS and self._yolo_majority(tid) != 'referee':
                continue  # Clearly a player already - nothing to fix
            colors = [c for _, c, _ in seq if c is not None]
            if not colors:
                continue
            if len(colors) < 0.7 * len(seq) and self.yolo_role_votes.get(tid, {}).get('referee', 0) >= 0.8 * len(seq):
                continue  # Shirt mostly inside grass/lime HSV range; remaining pixels are dark shorts/collar
            median = np.median(colors, axis=0)
            dists = self.kmeans.transform(self._weighted(median).reshape(1, -1))[0]
            cluster = int(np.argmin(dists))
            team = self.TEAMS[self.team_order.index(cluster)]
            if dists[cluster] > strong_factor * self.inlier_scale:
                continue  # Not clearly a team kit - keep the referee/outlier verdict
            fraction = sum(self.classify_color(c) == team for c in colors) / len(seq)
            if fraction < min_fraction:
                continue
            self.overrides[tid] = team
            self.track_labels[tid] = {f: team for f in labels}
            self.role_report.setdefault(tid, {})['color_override'] = {
                'role': 'player', 'team': team, 'yolo_roles': dict(self.yolo_role_votes.get(tid, {})),
                'kit_distance': round(float(dists[cluster]), 1)}

    def resolve_roles(self, positions: Dict[int, Dict[int, Optional[Tuple[float, float]]]],
                      pitch_length: float = 105.0, pitch_width: float = 68.0, min_obs: int = 15,
                      official_ratio: float = 0.5, min_deepest_ratio: float = 0.5,
                      touchline_margin: float = 6.0, official_match: float = 20.0,
                      min_role_votes: int = 10) -> Tuple[Dict[int, str], set]:
        """
        Identify goalkeepers and officials among the color-outlier tracks.

        Args:
            positions: track_id -> frame -> pitch position in meters (None if no homography)
            pitch_length, pitch_width: Pitch size in meters
            min_obs: Minimum observations (with a pitch position) for a goalkeeper candidate
            official_ratio: Fraction of a track's frames labelled as outlier to be an outlier track
            min_deepest_ratio: Fraction of its frames in which a goalkeeper must be the deepest
                               person on its half
            touchline_margin: Tracks closer than this to a touchline (median) are assistant referees
            official_match: Weighted Lab distance below which a candidate wears the officials' kit
            min_role_votes: Minimum detector role predictions for a YOLO role to count as evidence

        Returns:
            (goalkeepers: track_id -> defending team, officials: set of track_ids)
        """
        self.role_report = {}
        self._apply_color_overrides()  # YOLO 'referee' + strong team kit color -> player

        half = pitch_length / 2
        team_x = {team: [p[0] for tid, lab in self.track_labels.items() for f, l in lab.items()
                         if l == team and positions.get(tid, {}).get(f) is not None
                         for p in [positions[tid][f]]] for team in self.TEAMS}
        left_team = min(self.TEAMS, key=lambda t: np.median(team_x[t]) if team_x[t] else np.inf)
        right_team = self.TEAMS[1 - self.TEAMS.index(left_team)]

        # Deepest on-pitch person per frame and half (any label)
        best, deepest = {}, {}
        for tid, per_frame in positions.items():
            for f, p in per_frame.items():
                if p is None or not (-3 <= p[0] <= pitch_length + 3 and -3 <= p[1] <= pitch_width + 3):
                    continue
                key = (f, 'left' if p[0] < half else 'right')
                depth = p[0] if key[1] == 'left' else pitch_length - p[0]
                if key not in best or depth < best[key]:
                    best[key], deepest[key] = depth, tid

        outlier_tracks, candidates = [], {}
        yolo_officials = set()
        for tid, labels in self.track_labels.items():
            n_outlier = sum(l == 'referee' for l in labels.values())
            n_unknown = max(sum(l == 'unknown' for l in labels.values()),
                            sum(c is None for _, c, _ in self.samples[tid]))
            is_outlier = n_outlier >= official_ratio * len(labels)
            yolo_role = self._yolo_majority(tid)
            if is_outlier:
                outlier_tracks.append(tid)
            elif yolo_role == 'referee' and self.yolo_role_votes[tid]['referee'] >= min(min_role_votes, len(labels)) \
                    and n_unknown >= 0.3 * len(labels):
                # Kit could not be measured (grass-colored) but the model insists: referee
                yolo_officials.add(tid)
            elif n_unknown < official_ratio * len(labels):
                continue  # Clearly an outfield player of one team
            colors = [c for _, c, _ in self.samples[tid] if c is not None and self.classify_color(c) == 'referee']
            seen = [(f, p) for f, p in positions.get(tid, {}).items() if p is not None]
            if len(seen) < min_obs:
                continue
            pts = np.array([p for _, p in seen])
            side = 'left' if np.median(pts[:, 0]) < half else 'right'
            deepest_ratio = float(np.mean([deepest.get((f, side)) == tid for f, _ in seen]))
            touchline_dist = float(np.median(np.minimum(pts[:, 1], pitch_width - pts[:, 1])))
            report = {'frames': len(seen), 'kit': 'outlier' if is_outlier else 'unknown (no non-grass pixels)',
                      'yolo_roles': dict(self.yolo_role_votes.get(tid, {})),
                      'side': side, 'deepest_ratio': round(deepest_ratio, 2),
                      'median_touchline_distance_m': round(touchline_dist, 1),
                      'kit_bgr': self._lab_to_bgr(np.median(colors, axis=0)) if colors else None}
            if (deepest_ratio >= min_deepest_ratio and touchline_dist >= touchline_margin) \
                    or (yolo_role == 'goalkeeper' and self.yolo_role_votes[tid]['goalkeeper'] >= min_role_votes):
                candidates[tid] = (left_team if side == 'left' else right_team,
                                   np.median(colors, axis=0) if colors else None)
                report['goalkeeper_candidate'] = True
                if yolo_role == 'goalkeeper':
                    report['goalkeeper_by'] = 'yolo' if deepest_ratio < min_deepest_ratio else 'yolo+position'
            self.role_report[tid] = {**self.role_report.get(tid, {}), **report}

        official_colors = [(len(self.samples[t]), c) for t in outlier_tracks if t not in candidates
                           for c in [np.median([c for _, c, _ in self.samples[t] if c is not None], axis=0)]]
        if official_colors:
            self.official_color = np.average([c for _, c in official_colors], axis=0,
                                             weights=[n for n, _ in official_colors])

        self.goalkeepers = {}
        for tid, (team, color) in candidates.items():
            kit_distance = None if self.official_color is None or color is None else float(
                np.linalg.norm(self._weighted(color) - self._weighted(self.official_color)))
            self.role_report[tid]['official_kit_distance'] = None if kit_distance is None else round(kit_distance, 1)
            if kit_distance is None or kit_distance >= official_match:
                self.goalkeepers[tid] = team
                self.role_report[tid]['goalkeeper_of'] = team
        self.officials = (set(outlier_tracks) | yolo_officials) - set(self.goalkeepers)
        for tid in self.overrides:
            self.role_report.setdefault(tid, {}).setdefault('yolo_roles', dict(self.yolo_role_votes.get(tid, {})))
        return self.goalkeepers, self.officials

    def official_color_bgr(self) -> Optional[tuple]:
        return None if self.official_color is None else self._lab_to_bgr(self.official_color)

    def get_team_stats(self) -> Dict[str, int]:
        """
        Number of distinct track IDs per final role.
        """
        stats = Counter(Counter(l.values()).most_common(1)[0][0] for tid, l in self.track_labels.items()
                        if l and tid not in self.goalkeepers and tid not in self.officials)
        return {'team_a': stats.get('team_a', 0), 'team_b': stats.get('team_b', 0),
                'goalkeeper': len(self.goalkeepers), 'official': len(self.officials)}
