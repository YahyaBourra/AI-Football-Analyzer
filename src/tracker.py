"""
Object Tracker Module

This module handles multi-object tracking to assign consistent IDs to players
across video frames using ByteTrack (from the supervision library).

Key responsibilities:
- Associate person detections across frames (Kalman filter + IoU matching)
- Maintain consistent IDs through short occlusions (lost-track buffer)

Note: ByteTrack's association step is lightweight CPU math (Kalman filter + Hungarian
matching); the expensive part of the pipeline (YOLO inference) runs on the GPU.

The ball is not tracked with ByteTrack: there is only one ball, so its trajectory is
built from the detector's per-frame ball detections by BallTrajectory, which removes
outliers, interpolates short gaps and smooths the result.
"""

import warnings
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Track:
    """
    Represents a single tracked object.
    """
    track_id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    role: str = 'person'  # Detector class name ('person' for COCO; player/goalkeeper/referee when fine-tuned)

    @property
    def foot_position(self) -> tuple:
        """Bottom-center of the bounding box (where the player touches the ground)."""
        return ((self.bbox[0] + self.bbox[2]) // 2, self.bbox[3])

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class ObjectTracker:
    """
    Multi-object tracker for players using ByteTrack.
    """

    def __init__(self, fps: float = 30.0, lost_track_buffer: int = 60,
                 class_names: Optional[Dict[int, str]] = None):
        """
        Initialize the object tracker.

        Args:
            fps: Video frame rate (ByteTrack scales its buffer with it)
            lost_track_buffer: Frames to keep a lost track alive before its ID is dropped
            class_names: Detector class_id -> name map (model.names); used to tag each
                         Track with the detector's role prediction
        """
        self.fps = fps
        self.lost_track_buffer = lost_track_buffer
        self.class_names = class_names or {}
        self.tracker = None

    def initialize(self) -> None:
        """
        Initialize the ByteTrack backend.
        """
        with warnings.catch_warnings():
            # sv.ByteTrack is deprecated in supervision>=0.28 (removed in 0.31, pinned in requirements.txt)
            warnings.simplefilter('ignore', FutureWarning)
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.4,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=0.8,
                frame_rate=self.fps,
            )

    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracks with new person detections.

        Args:
            detections: List of detection dictionaries from the detector

        Returns:
            List of active Track objects for the current frame
        """
        if self.tracker is None:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        persons = [d for d in detections if d['class_name'] == 'person']
        if persons:
            sv_detections = sv.Detections(
                xyxy=np.array([d['bbox'] for d in persons], dtype=np.float32),
                confidence=np.array([d['confidence'] for d in persons], dtype=np.float32),
                class_id=np.array([d['class_id'] for d in persons], dtype=int),
            )
        else:
            sv_detections = sv.Detections.empty()

        tracked = self.tracker.update_with_detections(sv_detections)

        return [
            Track(
                track_id=int(tracker_id),
                bbox=tracked.xyxy[i].astype(int).tolist(),
                confidence=float(tracked.confidence[i]) if tracked.confidence is not None else 1.0,
                class_id=int(tracked.class_id[i]),
                class_name='person',
                role=self.class_names.get(int(tracked.class_id[i]), 'person'),
            )
            for i, tracker_id in enumerate(tracked.tracker_id)
        ]

    def reset(self) -> None:
        """
        Reset the tracker, clearing all active tracks.
        """
        if self.tracker is not None:
            self.tracker.reset()


def select_ball_candidates(candidates: List[List[Tuple[Tuple[float, float], float]]],
                           pitch_positions: List[Optional[List[Optional[Tuple[float, float]]]]],
                           pitch_length: float = 105.0, pitch_width: float = 68.0, margin: float = 0.5,
                           radius: float = 1.5, min_frames: int = 30, min_span: int = 90
                           ) -> Tuple[List[Optional[Tuple[float, float]]], List[Dict]]:
    """
    Choose one ball per frame from all candidates, removing static off-pitch objects.

    Some objects look like a ball and never move, e.g. a goalkeeper's towel inside the goal.
    Such an object is recognised in pitch coordinates (which are camera-motion compensated):
    candidates outside the field of play (beyond `margin` m of the lines) that stay within
    `radius` m of the same spot in at least `min_frames` frames spanning >= `min_span` frames.
    Only off-pitch static objects are removed, because a real ball can be static ON the pitch
    for many seconds (corner kicks, free kicks).

    Args:
        candidates: Per frame, list of (center in video pixels, confidence)
        pitch_positions: Per frame, list of candidate pitch positions in meters (None if unknown),
                         or None if the frame has no homography

    Returns:
        (per-frame chosen ball center or None, list of detected static objects)
    """
    def off_pitch(p):
        return not (-margin <= p[0] <= pitch_length + margin and -margin <= p[1] <= pitch_width + margin)

    clusters: List[Dict] = []  # {'center', 'frames'}
    for f, pts in enumerate(pitch_positions):
        for p in pts or []:
            if p is None or not off_pitch(p):
                continue
            for c in clusters:
                if np.hypot(p[0] - c['center'][0], p[1] - c['center'][1]) <= radius:
                    c['points'].append(p)
                    c['frames'].add(f)
                    c['center'] = tuple(np.mean(c['points'], axis=0))
                    break
            else:
                clusters.append({'center': p, 'points': [p], 'frames': {f}})

    static = [{'pitch_position': [round(float(v), 2) for v in c['center']], 'frames': len(c['frames']),
               'first_frame': min(c['frames']), 'last_frame': max(c['frames'])}
              for c in clusters if len(c['frames']) >= min_frames and max(c['frames']) - min(c['frames']) >= min_span]

    chosen = []
    for f, cands in enumerate(candidates):
        pts = pitch_positions[f] or [None] * len(cands)
        keep = [(center, conf) for (center, conf), p in zip(cands, pts)
                if p is None or not off_pitch(p)
                or all(np.hypot(p[0] - s['pitch_position'][0], p[1] - s['pitch_position'][1]) > radius for s in static)]
        chosen.append(max(keep, key=lambda c: c[1])[0] if keep else None)
    return chosen, static


class BallTrajectory:
    """
    Offline post-processing of the per-frame ball detections (video pixels).

    1. Outlier rejection: a detection is kept only if another detection within
       +/- `consistency_window` frames is reachable at a physically plausible speed
       (`max_speed_px` pixels per frame). Isolated false positives are removed.
    2. Gap filling: pandas `interpolate()` (linear, inside gaps only) fills gaps shorter
       than `max_gap` frames. Longer gaps stay empty: the ball may have left the frame,
       and inventing a long trajectory would corrupt possession and pass statistics.
    3. Smoothing: a centered moving average (`smooth_window` frames) removes detector
       jitter so trajectories and passes look natural. Each continuous segment is
       smoothed independently.
    """

    def __init__(self, max_gap: int = 20, smooth_window: int = 5, max_speed_px: float = 60.0,
                 consistency_window: int = 10):
        self.max_gap = max_gap
        self.smooth_window = smooth_window
        self.max_speed_px = max_speed_px
        self.consistency_window = consistency_window

    def _reject_outliers(self, xy: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(xy[:, 0])
        idx = np.flatnonzero(valid)
        keep = valid.copy()
        for k, i in enumerate(idx):
            neighbors = idx[max(0, k - self.consistency_window):k + self.consistency_window + 1]
            neighbors = neighbors[(neighbors != i) & (np.abs(neighbors - i) <= self.consistency_window)]
            dist = np.linalg.norm(xy[neighbors] - xy[i], axis=1)
            keep[i] = bool(np.any(dist <= self.max_speed_px * np.abs(neighbors - i)))
        return keep

    def process(self, detections: List[Optional[Tuple[float, float]]]) -> Tuple[pd.DataFrame, Dict]:
        """
        Build the ball trajectory.

        Args:
            detections: Per-frame ball center in video pixels, or None if not detected

        Returns:
            (DataFrame indexed by frame with columns x, y, raw_x, raw_y and source in
             {'detected', 'interpolated', 'missing'}, summary statistics dict)
        """
        raw = np.array([d if d is not None else (np.nan, np.nan) for d in detections], dtype=float).reshape(-1, 2)
        keep = self._reject_outliers(raw)
        clean = np.where(keep[:, None], raw, np.nan)

        df = pd.DataFrame(clean, columns=['x', 'y'])
        missing = df['x'].isna()
        run_id = (missing != missing.shift()).cumsum()
        run_len = missing.groupby(run_id).transform('size')
        filled = df.interpolate(method='linear', limit_area='inside')
        filled.loc[missing & (run_len >= self.max_gap)] = np.nan

        has_ball = filled['x'].notna()
        segment = (has_ball != has_ball.shift()).cumsum()
        smoothed = filled.groupby(segment).transform(
            lambda s: s.rolling(self.smooth_window, center=True, min_periods=1).mean())
        smoothed.loc[~has_ball] = np.nan

        out = smoothed.assign(raw_x=raw[:, 0], raw_y=raw[:, 1])
        out['source'] = np.where(keep, 'detected', np.where(has_ball, 'interpolated', 'missing'))
        out.index.name = 'frame'

        stats = {
            'frames': len(out),
            'raw_detections': int((~np.isnan(raw[:, 0])).sum()),
            'rejected_outliers': int((~np.isnan(raw[:, 0])).sum() - keep.sum()),
            'detected': int(keep.sum()),
            'interpolated': int((out['source'] == 'interpolated').sum()),
            'missing': int((out['source'] == 'missing').sum()),
        }
        return out, stats


class KalmanBallTracker(BallTrajectory):
    """
    Physics-based ball trajectory (Phase 7): constant-velocity Kalman filter
    (cv2.KalmanFilter, state [x, y, vx, vy]) + Rauch-Tung-Striebel backward smoother.

    - A valid detection is a Kalman *correction*. A large innovation (the ball
      reappearing far away after a replay wipe or a spell off-camera) re-initializes
      the filter at the measurement instead of letting it lag behind the jump; the
      re-initialized velocity comes from the displacement across the gap.
    - Without a detection the filter only *predicts*; predictions are used for at most
      `max_predict` consecutive frames (beyond that the position is marked missing -
      the ball may have left the frame and inventing a path would corrupt stats).
    - The RTS backward pass re-estimates every covered frame using both past and
      future measurements. It replaces both the linear gap-filling and the moving
      average of BallTrajectory: the output path follows the ball's momentum through
      occlusions instead of teleporting straight between detections or bouncing with
      the ~5 px detector jitter.
    - The outlier rejection of BallTrajectory runs first, so the filter never sees
      isolated false positives.
    """

    def __init__(self, max_predict: int = 15, meas_noise: float = 25.0, accel_noise: float = 10.0,
                 reinit_distance: float = 150.0, **kwargs):
        """
        Args:
            max_predict: Max consecutive frames filled by prediction alone
            meas_noise: R - detector position variance (px^2; ~5 px jitter -> 25)
            accel_noise: Process noise spectral density (px/frame^2)^2 - large enough
                         to let the velocity follow kicks
            reinit_distance: Innovation (px) above which the filter is re-initialized
            **kwargs: Passed to BallTrajectory (outlier rejection parameters)
        """
        super().__init__(**kwargs)
        self.max_predict = max_predict
        self.meas_noise = meas_noise
        self.accel_noise = accel_noise
        self.reinit_distance = reinit_distance

    def _new_filter(self) -> cv2.KalmanFilter:
        """Constant-velocity Kalman filter: state [x, y, vx, vy], measurement [x, y]."""
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        # Discrete white-noise acceleration model (kicks are velocity impulses)
        kf.processNoiseCov = self.accel_noise * np.array(
            [[0.25, 0, 0.5, 0], [0, 0.25, 0, 0.5],
             [0.5, 0, 1, 0], [0, 0.5, 0, 1]], np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * np.float32(self.meas_noise)
        return kf

    def process(self, detections: List[Optional[Tuple[float, float]]]) -> Tuple[pd.DataFrame, Dict]:
        """
        Build the ball trajectory with the Kalman filter + RTS smoother.

        Args:
            detections: Per-frame ball center in video pixels, or None if not detected

        Returns:
            (DataFrame indexed by frame with columns x, y, raw_x, raw_y and source in
             {'detected', 'kalman', 'missing'}, summary statistics dict)
        """
        raw = np.array([d if d is not None else (np.nan, np.nan) for d in detections],
                       dtype=float).reshape(-1, 2)
        keep = self._reject_outliers(raw)
        meas = np.where(keep[:, None], raw, np.nan)
        n = len(meas)

        kf = self._new_filter()
        F = np.asarray(kf.transitionMatrix, dtype=np.float64)
        P0 = np.diag([25.0, 25.0, 400.0, 400.0])  # tight position, loose velocity
        xf = np.full((n, 4), np.nan)      # Filtered (posterior) state per frame
        Pf = np.full((n, 4, 4), np.nan)
        xp = np.full((n, 4), np.nan)      # Predicted (prior) state per frame
        Pp = np.full((n, 4, 4), np.nan)
        used_pred = np.zeros(n, bool)     # Output comes from prediction this frame
        reinit = np.zeros(n, bool)

        initialized, miss_run, last_meas = False, 0, None
        for t in range(n):
            m, has = meas[t], not np.isnan(meas[t, 0])
            if not initialized:
                if not has:
                    continue
                kf.statePost = np.array([[m[0]], [m[1]], [0], [0]], np.float32)
                kf.errorCovPost = P0.astype(np.float32)
                xp[t], Pp[t] = kf.statePost.ravel().astype(float), P0
                xf[t], Pf[t] = xp[t].copy(), P0.copy()
                initialized, last_meas = True, t
                continue

            kf.predict()
            xp[t], Pp[t] = kf.statePre.ravel().astype(float), np.asarray(kf.errorCovPre, float)
            if has:
                innovation = float(np.hypot(m[0] - xp[t, 0], m[1] - xp[t, 1]))
                if innovation > self.reinit_distance:
                    # Re-initialize at the measurement; velocity from the gap displacement
                    gap = t - last_meas if last_meas is not None else 1
                    v = np.clip((m - meas[last_meas]) / gap, -self.max_speed_px, self.max_speed_px) \
                        if last_meas is not None else np.zeros(2)
                    kf.statePost = np.array([[m[0]], [m[1]], [v[0]], [v[1]]], np.float32)
                    kf.errorCovPost = P0.astype(np.float32)
                    reinit[t] = True
                else:
                    kf.correct(m.astype(np.float32).reshape(2, 1))
                miss_run, last_meas = 0, t
            else:
                miss_run += 1
                used_pred[t] = miss_run <= self.max_predict
            xf[t], Pf[t] = kf.statePost.ravel().astype(float), np.asarray(kf.errorCovPost, float)

        # Rauch-Tung-Striebel backward smoothing; re-init frames break the segment so a
        # jump (e.g. over a replay wipe) never drags earlier estimates along.
        xs = xf.copy()
        for t in range(n - 2, -1, -1):
            if np.isnan(xf[t, 0]) or np.isnan(xp[t + 1, 0]) or reinit[t + 1]:
                continue
            gain = Pf[t] @ F.T @ np.linalg.pinv(Pp[t + 1])
            xs[t] = xf[t] + gain @ (xs[t + 1] - xp[t + 1])

        measured = ~np.isnan(meas[:, 0])
        covered = measured | used_pred
        out_xy = np.where(covered[:, None], xs[:, :2], np.nan)

        out = pd.DataFrame({'x': out_xy[:, 0], 'y': out_xy[:, 1],
                            'raw_x': raw[:, 0], 'raw_y': raw[:, 1]})
        out['source'] = np.where(measured, 'detected', np.where(covered, 'kalman', 'missing'))
        out.index.name = 'frame'

        stats = {
            'frames': n,
            'raw_detections': int((~np.isnan(raw[:, 0])).sum()),
            'rejected_outliers': int((~np.isnan(raw[:, 0])).sum() - keep.sum()),
            'detected': int(keep.sum()),
            'predicted': int(used_pred.sum()),
            'missing': int((out['source'] == 'missing').sum()),
            'reinitializations': int(reinit.sum()),
            'smoother': 'kalman+RTS',
        }
        return out, stats
