"""
Tactical Map (Homography) Module

This module handles the perspective transformation from video frame coordinates to
a 2D top-down tactical pitch using a homography, and renders the tactical map.

Coordinate systems:
- Video space: pixels of the broadcast frame.
- Pitch space: meters, origin at the top-left corner flag, x along the touchline
  (0..105), y along the goal line (0..68). y=0 is the far touchline in the video.
- Map space: pixels of the rendered tactical map image (pitch meters * scale + margin).

Dynamic camera (optical flow)
-----------------------------
The broadcast camera pans, tilts and zooms, so a single static homography is only valid
for one frame. CameraMotionEstimator measures the frame-to-frame camera motion with
pyramidal Lucas-Kanade optical flow on static scene features (players masked out). For a
camera that rotates/zooms about a fixed point, all static points move according to the
same 3x3 homography, so the motion is fitted as a homography with RANSAC. Chaining these
motions links every frame t to a reference frame r of its segment:

    G_t           : video_t -> video_r        (product of the frame-to-frame motions)
    H_t           = H_r @ G_t                  (video_t -> pitch)

The (dx, dy) shift of the image center is also reported per frame (e.g. for plots).

Segments and calibration
------------------------
A camera cut (or a replay wipe) breaks the chain, so the video is split into segments at
detected cuts. Each segment needs at least one calibration (hand-measured pitch-line
intersections, see CALIBRATIONS). Calibration points from several frames of the same
segment are mapped into its reference frame with G and fitted jointly
(cv2.findHomography), so the homography is constrained over the whole area the camera
covers, not just the part visible in one frame. Frames in a segment without calibration
get no homography (the tactical map is then disabled rather than wrong).
"""

import json
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PitchDimensions:
    """
    Standard football pitch dimensions in meters.
    """
    length: float = 105.0  # Touchline length (x axis)
    width: float = 68.0  # Goal line length (y axis)
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.32
    goal_area_length: float = 5.5
    goal_area_width: float = 18.32
    center_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0
    goal_width: float = 7.32


# Hardcoded calibration for data/input/test_match.mp4: pitch-line intersections measured by
# hand on single frames. Format: frame -> list of ((video_x, video_y), (pitch_x, pitch_y)).
CALIBRATIONS: Dict[int, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {
    0: [  # Frame 0 calibration so short clips / previews (<300 frames) have homography immediately
        ((536, 400), (16.5, 13.84)),    # Left penalty area, far front corner
        ((428, 472), (16.5, 26.69)),    # Left penalty area front line meets penalty arc (far)
        ((264, 581), (16.5, 41.31)),    # Left penalty area front line meets penalty arc (near)
        ((38, 731), (16.5, 54.16)),     # Left penalty area, near front corner
        ((1387, 349), (52.5, 0.0)),     # Halfway line meets far touchline
        ((1399, 495), (52.5, 24.85)),   # Halfway line meets center circle (far side)
        ((1416, 650), (52.5, 43.15)),   # Halfway line meets center circle (near side)
    ],
    300: [
        ((680, 328), (52.5, 0.0)),     # Halfway line meets the far touchline
        ((697, 462), (52.5, 24.85)),   # Halfway line meets the center circle (far side)
        ((722, 626), (52.5, 43.15)),   # Halfway line meets the center circle (near side)
        ((1566, 354), (88.5, 13.84)),  # Right penalty area, far corner
        ((1699, 428), (88.5, 26.69)),  # Right penalty area front line meets the penalty arc
        ((1802, 279), (105.0, 0.0)),   # Right corner flag (far touchline meets goal line)
    ],
    1175: [
        ((150, 760), (0.0, 68.0)),     # Left near corner flag
        ((580, 550), (0.0, 43.16)),    # Goal line meets goal area (near side)
        ((765, 568), (5.5, 43.16)),    # Goal area, near front corner
        ((978, 472), (5.5, 24.84)),    # Goal area, far front corner
        ((868, 692), (16.5, 54.16)),   # Penalty area, near front corner
        ((1055, 583), (16.5, 41.31)),  # Penalty area front line meets the penalty arc (near)
        ((1190, 502), (16.5, 26.69)),  # Penalty area front line meets the penalty arc (far)
    ],
    1500: [  # After the replay wipe at frames ~1288-1295 (separate segment)
        ((720, 550), (0.0, 43.16)),    # Goal line meets goal area (near side)
        ((865, 568), (5.5, 43.16)),    # Goal area, near front corner
        ((1118, 458), (5.5, 24.84)),   # Goal area, far front corner
        ((515, 638), (0.0, 54.16)),    # Goal line meets penalty area (near side)
        ((983, 722), (16.5, 54.16)),   # Penalty area, near front corner
    ],
}


@dataclass
class CameraMotion:
    """
    Camera motion estimated for one frame.
    """
    frame: int
    src_frame: Optional[int] = None  # Frame the motion is measured from (None = no estimate)
    H: Optional[np.ndarray] = None  # 3x3 homography: src_frame pixels -> this frame pixels
    new_segment: bool = False  # Cut detected / tracking lost: not linked to earlier frames
    inliers: int = 0
    tracked: int = 0  # Features that passed the forward-backward check
    hist_diff: float = 0.0  # Color-histogram (Bhattacharyya) distance to the previous frame
    dx: float = 0.0  # Shift of the image center (pixels) relative to src_frame
    dy: float = 0.0


class CameraMotionEstimator:
    """
    Estimates frame-to-frame camera motion with pyramidal Lucas-Kanade optical flow
    (cv2.calcOpticalFlowPyrLK).

    - Features (cv2.goodFeaturesToTrack) are taken from static scene content: everything
      except player/ball boxes and the broadcast graphics corner. With `use_background=False`
      only the pitch (grass + lines) is used; on test_match.mp4 that drifted by ~100-200 px
      over 900 frames because grass has little trackable texture, while including the
      static advertising boards and stands removed the drift.
    - A forward-backward consistency check removes badly tracked points.
    - The motion is fitted with RANSAC (cv2.findHomography), rejecting remaining outliers.
    - Cut detection: a large color-histogram change or losing most tracked features
      starts a new segment.
    """

    def __init__(self, scale: float = 0.5, max_corners: int = 600, min_inliers: int = 40,
                 max_skip: int = 5, use_background: bool = True,
                 cut_hist_threshold: float = 0.08, cut_track_ratio: float = 0.5):
        self.scale = scale
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.max_skip = max_skip
        self.use_background = use_background
        self.cut_hist_threshold = cut_hist_threshold
        self.cut_track_ratio = cut_track_ratio
        self._last = None  # (frame_idx, gray, points)
        self._prev_hist = None
        S = np.diag([scale, scale, 1.0])
        self._S, self._S_inv = S, np.linalg.inv(S)

    def _features(self, gray: np.ndarray, pitch_mask: np.ndarray, boxes: List[List[int]]) -> Optional[np.ndarray]:
        h, w = gray.shape
        if self.use_background:
            mask = np.full((h, w), 255, np.uint8)
            mask[:int(0.12 * h), :int(0.25 * w)] = 0  # Broadcast graphics (score bug / logos) are static on screen
        else:
            mask = cv2.dilate(cv2.resize(pitch_mask, (w, h), interpolation=cv2.INTER_NEAREST), np.ones((15, 15), np.uint8))
        pad = 6
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(mask, (int(x1 * self.scale) - pad, int(y1 * self.scale) - pad),
                          (int(x2 * self.scale) + pad, int(y2 * self.scale) + pad), 0, -1)
        return cv2.goodFeaturesToTrack(gray, self.max_corners, qualityLevel=0.01, minDistance=8, mask=mask)

    def update(self, frame_idx: int, frame: np.ndarray, pitch_mask: np.ndarray,
               exclude_boxes: List[List[int]]) -> CameraMotion:
        """
        Estimate the camera motion between the last good frame and this frame.

        Args:
            frame_idx: Index of this frame
            frame: BGR frame
            pitch_mask: Pitch mask of this frame (255 = grass)
            exclude_boxes: Boxes of moving objects (players, ball) to ignore

        Returns:
            CameraMotion record for this frame
        """
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hist_diff = 0.0 if self._prev_hist is None else cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        self._prev_hist = hist
        record = CameraMotion(frame=frame_idx, hist_diff=float(hist_diff))

        cut = hist_diff > self.cut_hist_threshold
        if not cut and self._last is not None and self._last[2] is not None and len(self._last[2]) >= self.min_inliers:
            last_idx, last_gray, last_pts = self._last
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(last_gray, gray, last_pts, None, winSize=(21, 21), maxLevel=3)
            back, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, last_gray, nxt, None, winSize=(21, 21), maxLevel=3)
            good = (st.ravel() == 1) & (st_b.ravel() == 1) & (np.linalg.norm(back - last_pts, axis=2).ravel() < 1.0)
            record.tracked = int(good.sum())
            cut = record.tracked < self.cut_track_ratio * len(last_pts)
            if not cut and good.sum() >= self.min_inliers:
                H_small, inl = cv2.findHomography(last_pts[good], nxt[good], cv2.RANSAC, 1.5)
                if H_small is not None and inl.sum() >= self.min_inliers:
                    H = self._S_inv @ H_small @ self._S
                    c = np.array([[[frame.shape[1] / 2, frame.shape[0] / 2]]], dtype=np.float32)
                    moved = cv2.perspectiveTransform(c, H)[0, 0]
                    record.src_frame, record.H, record.inliers = last_idx, H, int(inl.sum())
                    record.dx, record.dy = float(moved[0] - c[0, 0, 0]), float(moved[1] - c[0, 0, 1])

        if record.H is None and (cut or self._last is None or frame_idx - self._last[0] > self.max_skip):
            record.new_segment = True
        if record.H is not None or record.new_segment:
            self._last = (frame_idx, gray, self._features(gray, pitch_mask, exclude_boxes))
        return record


class TacticalMap:
    """
    Transforms video coordinates to pitch coordinates and renders the 2D tactical map.
    """

    def __init__(self, pitch_dimensions: Optional[PitchDimensions] = None,
                 scale: int = 10, margin: int = 20, calibrations: Optional[Dict] = None):
        """
        Initialize the tactical map.

        Args:
            pitch_dimensions: Pitch dimensions (standard FIFA dimensions if None)
            scale: Map pixels per meter
            margin: Map border around the pitch, in pixels
            calibrations: frame -> list of ((video_x, video_y), (pitch_x, pitch_y)); defaults to CALIBRATIONS
        """
        self.dims = pitch_dimensions or PitchDimensions()
        self.scale = scale
        self.margin = margin
        self.map_width = int(self.dims.length * scale) + 2 * margin
        self.map_height = int(self.dims.width * scale) + 2 * margin
        self.pitch_template = self.draw_pitch()

        self.calibrations = {int(k): v for k, v in (calibrations or CALIBRATIONS).items()}
        self.frame_homographies: Optional[List[Optional[np.ndarray]]] = None  # Per-frame video -> pitch
        self.segments: List[Dict] = []
        first = min(self.calibrations)
        self.homography_matrix = self.fit_homography(self.calibrations[first])  # Static fallback

    # ------------------------------------------------------------------ calibration
    @staticmethod
    def fit_homography(points: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> np.ndarray:
        """
        Video -> pitch homography from >= 4 correspondences: exact for 4 points
        (cv2.getPerspectiveTransform), least squares for more (cv2.findHomography).
        """
        src = np.array([v for v, _ in points], dtype=np.float32)
        dst = np.array([p for _, p in points], dtype=np.float32)
        if len(src) < 4:
            raise ValueError("A homography needs at least 4 point correspondences")
        return cv2.getPerspectiveTransform(src, dst) if len(src) == 4 else cv2.findHomography(src, dst, 0)[0]

    def save_calibration(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump({str(k): v for k, v in self.calibrations.items()}, f, indent=2)

    def load_calibration(self, filepath: str) -> None:
        with open(filepath) as f:
            self.calibrations = {int(k): [tuple(map(tuple, p)) for p in v] for k, v in json.load(f).items()}
        self.homography_matrix = self.fit_homography(self.calibrations[min(self.calibrations)])

    def set_camera_motion(self, motions: List[CameraMotion]) -> Dict:
        """
        Build per-frame homographies from the camera motions and the calibrations.

        Args:
            motions: One CameraMotion per frame, in frame order

        Returns:
            Summary: segments (start, end, reference frame, calibration error) and valid frame count
        """
        n = len(motions)
        starts = [i for i, m in enumerate(motions) if m.new_segment or i == 0] + [n]
        self.frame_homographies = [None] * n
        self.segments = []

        for s, e in zip(starts[:-1], starts[1:]):
            calib_frames = sorted(f for f in self.calibrations if s <= f < e)
            if len(calib_frames) > 1 and 0 in calib_frames and 300 in calib_frames:
                calib_frames = [f for f in calib_frames if f > 0]
            seg = {'start': s, 'end': e - 1, 'reference_frame': None, 'calibration_rmse_m': None}
            self.segments.append(seg)
            if not calib_frames:
                continue
            ref = calib_frames[0]
            G = self._chain_to_reference(motions, s, e, ref)

            points = []
            for f in calib_frames:
                if G[f] is None:
                    continue
                v = np.array([p[0] for p in self.calibrations[f]], dtype=np.float32).reshape(-1, 1, 2)
                v_ref = cv2.perspectiveTransform(v, G[f]).reshape(-1, 2)
                points += [(tuple(vr), p[1]) for vr, p in zip(v_ref, self.calibrations[f])]
            H_ref = self.fit_homography(points)

            pred = cv2.perspectiveTransform(np.array([p[0] for p in points], np.float32).reshape(-1, 1, 2), H_ref)
            err = np.linalg.norm(pred.reshape(-1, 2) - np.array([p[1] for p in points]), axis=1)
            seg.update(reference_frame=ref, calibration_rmse_m=float(np.sqrt(np.mean(err ** 2))),
                       calibration_points=len(points))
            for t in range(s, e):
                if G[t] is not None:
                    self.frame_homographies[t] = H_ref @ G[t]

        return {'segments': self.segments, 'valid_frames': sum(h is not None for h in self.frame_homographies)}

    @staticmethod
    def _chain_to_reference(motions: List[CameraMotion], s: int, e: int, ref: int) -> List[Optional[np.ndarray]]:
        """
        G[t] maps frame t pixels to reference frame pixels, for t in segment [s, e).
        Frames whose own motion was not measured hold the neighbouring estimate.
        """
        G: List[Optional[np.ndarray]] = [None] * len(motions)
        G[ref] = np.eye(3)
        for t in range(ref + 1, e):  # Forward: G_t = G_src @ inv(H_src->t)
            m = motions[t]
            G[t] = G[m.src_frame] @ np.linalg.inv(m.H) if m.H is not None and G[m.src_frame] is not None else G[t - 1]
        for u in range(ref, s, -1):  # Backward: G_src = G_u @ H_src->u
            m = motions[u]
            if G[u] is not None and m.H is not None and m.src_frame >= s and G[m.src_frame] is None:
                G[m.src_frame] = G[u] @ m.H
            if G[u - 1] is None:
                G[u - 1] = G[u]
        return G

    def homography_for(self, frame_idx: Optional[int]) -> Optional[np.ndarray]:
        """
        Video -> pitch homography for a frame (static calibration if no camera motion is set).
        """
        if frame_idx is None or self.frame_homographies is None:
            return self.homography_matrix
        return self.frame_homographies[frame_idx] if frame_idx < len(self.frame_homographies) else None

    # ------------------------------------------------------------------ transforms
    def transform_points(self, video_points: np.ndarray, frame_idx: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Transform N video points (pixels) to pitch coordinates (meters).

        Args:
            video_points: Array of shape (N, 2)
            frame_idx: Frame the points belong to (uses its camera-compensated homography)

        Returns:
            Array of shape (N, 2) in meters, or None if the frame has no valid homography
        """
        H = self.homography_for(frame_idx)
        if H is None:
            return None
        pts = np.asarray(video_points, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    def transform_point(self, x: float, y: float, frame_idx: Optional[int] = None) -> Optional[Tuple[float, float]]:
        """
        Transform a single video point (pixels) to pitch coordinates (meters).
        """
        pts = self.transform_points(np.array([[x, y]]), frame_idx)
        return None if pts is None else (float(pts[0, 0]), float(pts[0, 1]))

    def pitch_to_map(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """
        Convert pitch meters to tactical map pixels.
        """
        return (int(round(point[0] * self.scale + self.margin)),
                int(round(point[1] * self.scale + self.margin)))

    def is_inside_pitch(self, point: Tuple[float, float], tolerance: float = 3.0) -> bool:
        """
        True if a pitch point lies on the pitch (plus a tolerance in meters).
        """
        return (-tolerance <= point[0] <= self.dims.length + tolerance and
                -tolerance <= point[1] <= self.dims.width + tolerance)

    # ------------------------------------------------------------------ rendering
    def draw_pitch(self) -> np.ndarray:
        """
        Draw a to-scale 2D football pitch: green background with white markings.

        Returns:
            BGR image of size (map_height, map_width)
        """
        d, white, t = self.dims, (255, 255, 255), 2
        pitch = np.full((self.map_height, self.map_width, 3), (40, 140, 40), dtype=np.uint8)
        # Alternating mowing stripes for readability
        stripe = self.dims.length / 10
        for i in range(0, 10, 2):
            p1, p2 = self.pitch_to_map((i * stripe, 0)), self.pitch_to_map(((i + 1) * stripe, d.width))
            cv2.rectangle(pitch, p1, p2, (50, 155, 50), -1)

        P = self.pitch_to_map
        r = lambda m: int(round(m * self.scale))
        cv2.rectangle(pitch, P((0, 0)), P((d.length, d.width)), white, t)
        cv2.line(pitch, P((d.length / 2, 0)), P((d.length / 2, d.width)), white, t)
        center = P((d.length / 2, d.width / 2))
        cv2.circle(pitch, center, r(d.center_circle_radius), white, t)
        cv2.circle(pitch, center, 4, white, -1)

        cy = d.width / 2
        for side in (0, 1):
            sign = 1 if side == 0 else -1
            gx = 0 if side == 0 else d.length
            # Penalty and goal areas
            cv2.rectangle(pitch, P((gx, cy - d.penalty_area_width / 2)),
                          P((gx + sign * d.penalty_area_length, cy + d.penalty_area_width / 2)), white, t)
            cv2.rectangle(pitch, P((gx, cy - d.goal_area_width / 2)),
                          P((gx + sign * d.goal_area_length, cy + d.goal_area_width / 2)), white, t)
            # Penalty spot and arc (the part of the 9.15 m circle outside the box)
            spot = P((gx + sign * d.penalty_spot_distance, cy))
            cv2.circle(pitch, spot, 3, white, -1)
            half_angle = np.degrees(np.arccos((d.penalty_area_length - d.penalty_spot_distance)
                                              / d.center_circle_radius))
            base = 0 if side == 0 else 180
            cv2.ellipse(pitch, spot, (r(d.center_circle_radius),) * 2, base, -half_angle, half_angle, white, t)
            # Goal
            cv2.rectangle(pitch, P((gx - sign * 1.5, cy - d.goal_width / 2)),
                          P((gx, cy + d.goal_width / 2)), white, t)
        return pitch

    def render(self, players: List[Dict], ball: Optional[Tuple[float, float]],
               passes: List[Dict], team_colors: Dict[str, Tuple[int, int, int]],
               background: Optional[np.ndarray] = None, message: Optional[str] = None,
               hulls: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Render the tactical map with players, the ball and recent passes.

        Args:
            players: Dicts with 'pitch_position' (meters), 'team', 'track_id', 'has_ball', 'is_goalkeeper'
            ball: Ball position in meters, or None
            passes: Dicts with 'from_pos', 'to_pos' (meters) and 'team' for recently completed passes
            team_colors: BGR color per team label
            background: Optional pre-rendered pitch (e.g. with a heatmap) to draw on
            message: Optional text shown on the map (e.g. when no homography is available)
            hulls: team -> convex hull vertices in pitch meters (controlled area, drawn
                   as a semi-transparent polygon in the team color under the players)

        Returns:
            BGR tactical map image
        """
        img = (background if background is not None else self.pitch_template).copy()

        for team, hull in (hulls or {}).items():
            if len(hull) < 3:
                continue
            poly = np.array([self.pitch_to_map(tuple(p)) for p in hull], dtype=np.int32)
            color = team_colors.get(team, (160, 160, 160))
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)

        for p in passes:
            color = team_colors.get(p['team'], (255, 255, 255))
            cv2.arrowedLine(img, self.pitch_to_map(p['from_pos']), self.pitch_to_map(p['to_pos']),
                            color, 3, tipLength=0.08)

        for p in players:
            pos = self.pitch_to_map(p['pitch_position'])
            color = team_colors.get(p['team'], (160, 160, 160))
            if p.get('is_goalkeeper'):
                cv2.rectangle(img, (pos[0] - 11, pos[1] - 11), (pos[0] + 11, pos[1] + 11), color, -1)
                cv2.rectangle(img, (pos[0] - 11, pos[1] - 11), (pos[0] + 11, pos[1] + 11),
                              (0, 255, 255) if p.get('has_ball') else (255, 255, 255), 2)
            else:
                cv2.circle(img, pos, 11, color, -1)
                cv2.circle(img, pos, 11, (0, 255, 255) if p.get('has_ball') else (255, 255, 255), 2)
            cv2.putText(img, str(p['track_id']), (pos[0] - 9, pos[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if ball is not None:
            pos = self.pitch_to_map(ball)
            cv2.circle(img, pos, 7, (255, 255, 255), -1)
            cv2.circle(img, pos, 7, (0, 0, 0), 2)

        if message:
            (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            org = ((self.map_width - tw) // 2, (self.map_height + th) // 2)
            cv2.putText(img, message, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(img, message, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
        return img
