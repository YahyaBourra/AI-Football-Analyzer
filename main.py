"""
Football Tactical Analysis AI - Main Entry Point

The pipeline runs in two passes, because some steps need information from the whole clip
(interpolating a ball gap needs the detections after the gap; a goalkeeper is identified
by where he stays over time):

Pass 1 (GPU, per frame)
  1. YOLO11 detection of persons and the ball; pitch masking removes off-pitch people
  2. ByteTrack tracking -> persistent player IDs
  3. Jersey color sample of every tracked person
  4. Optical-flow camera motion estimation (players masked out)

Between passes (whole clip)
  5. Team kits: trimmed KMeans on all jersey samples (Team A / Team B / color outlier);
     tracks the model labelled 'referee' are overridden to 'player' when their median
     kit color clearly matches a team (Phase 6)
  6. Per-frame homographies from the camera motion + calibrations (dynamic tactical map)
  7. Roles: goalkeepers (deepest outlier on a half, or the model's 'goalkeeper' class),
     officials; speed & distance per track from the smoothed pitch positions (PlayerKinematics)
  8. Ball: remove static off-pitch look-alikes (e.g. a towel in the goal), pick the best
     candidate per frame, then outlier rejection, interpolation of gaps < 20 frames, smoothing.
     With --ball-mode sliced/sliced+full the ball is also searched on overlapping tiles
     (SAHI-style), because at ~5-9 px it falls below the detection limit of a full-frame pass.

Pass 2 (per frame)
  9. Possession and pass analysis (officials ignored, goalkeepers count for their team)
 10. Team compactness: convex hull of each team's visible players on the map (TeamShape)
 11. Annotated video with PiP tactical map + hulls + heatmaps + speed tags + scoreboard
     + top-distance leaderboard

Model selection: the newest fine-tuned checkpoint runs/detect/*/weights/best.pt (see train.py)
is used if one exists, otherwise the COCO-pretrained models/yolo11m.pt. Override with --model.

Usage:
    python main.py --input data/input/test_match.mp4 --output data/output/test_output_phase6.mp4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / 'src'))
sys.path.append(str(ROOT / 'utils'))

from detector import YOLODetector
from tracker import ObjectTracker, BallTrajectory, KalmanBallTracker, select_ball_candidates
from team_assigner import TeamAssigner
from analyzer import TacticalAnalyzer, PlayerKinematics, TeamShape
from tactical_map import TacticalMap, CameraMotionEstimator
from visualization import annotate_frame, ColorScheme, TeamHeatmap


BASE_MODEL = ROOT / 'models' / 'yolo11m.pt'


def default_model() -> Path:
    """Newest fine-tuned checkpoint (runs/detect/*/weights/best.pt), else the COCO base model."""
    checkpoints = sorted((ROOT / 'runs' / 'detect').glob('*/weights/best.pt'), key=lambda p: p.stat().st_mtime)
    if checkpoints:
        return checkpoints[-1]
    return BASE_MODEL if BASE_MODEL.exists() else Path('yolo11m.pt')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Football Tactical Analysis AI')
    parser.add_argument('--input', type=str, default='data/input/test_match.mp4', help='Input video file')
    parser.add_argument('--output', type=str, default='data/output/test_output_phase7_kalman.mp4', help='Output video file')
    parser.add_argument('--model', type=str, default=None,
                        help='YOLO11 weights (default: newest runs/detect/*/weights/best.pt, else models/yolo11m.pt)')
    parser.add_argument('--ball-mode', type=str, default='sliced+full',
                        choices=['full', 'sliced', 'sliced+full'],
                        help='Ball inference: full frame, SAHI-style tiles over the pitch, or both (default: sliced+full)')
    parser.add_argument('--tile-size', type=int, default=640, help='Ball tile side in frame pixels')
    parser.add_argument('--tile-imgsz', type=int, default=960,
                        help='Inference size per tile (default 960: measured best recall/runtime on test_match.mp4)')
    parser.add_argument('--ball-smoother', type=str, default='kalman', choices=['kalman', 'linear'],
                        help='Ball trajectory: Kalman filter + RTS smoother with prediction (default), '
                             'or linear interpolation + moving average')
    parser.add_argument('--device', type=str, default='cuda', help="'cuda' (default) or 'cpu'")
    parser.add_argument('--calibration', type=str, default=None,
                        help='JSON calibration file {frame: [[[vx, vy], [px, py]], ...]} (overrides the built-in one)')
    parser.add_argument('--max-frames', type=int, default=None, help='Only process the first N frames')
    return parser.parse_args()


def _progress_iter(iterable, desc, progress_cb, phase):
    """Iterate with tqdm (CLI) or report progress to progress_cb(phase, i, n) (e.g. Streamlit)."""
    if progress_cb is None:
        yield from tqdm(iterable, desc=desc, unit='frame')
    else:
        n = len(iterable)
        for i, x in enumerate(iterable):
            progress_cb(phase, i, n)
            yield x
        progress_cb(phase, n, n)


def run_pass1(cap, total_frames, detector, tracker, team_assigner, motion_estimator, progress_cb=None):
    """Detection, tracking, jersey color sampling and camera motion for every frame."""
    records = []
    for frame_idx in _progress_iter(range(total_frames), "Pass 1/2 (detect+track)", progress_cb, 'pass1'):
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        balls = [(((d['bbox'][0] + d['bbox'][2]) / 2, (d['bbox'][1] + d['bbox'][3]) / 2), d['confidence'])
                 for d in detections if d['class_name'] == 'ball']
        tracks = tracker.update(detections)
        team_assigner.collect(frame_idx, frame, tracks)
        motion = motion_estimator.update(frame_idx, frame, detector.last_pitch_mask,
                                         [d['bbox'] for d in detections])
        records.append({'tracks': tracks, 'balls': balls, 'motion': motion})
    return records


def run_pipeline(input_path, output_path, model_path=None, device='cuda',
                 ball_mode='sliced+full', tile_size=640, tile_imgsz=960, ball_smoother='kalman',
                 calibration=None, max_frames=None, progress_cb=None, log=print) -> Dict:
    """
    Run the complete two-pass analysis pipeline on a video.

    Args:
        input_path, output_path: Video in / annotated video out
        model_path: YOLO weights (None = newest fine-tuned best.pt, else yolo11m.pt)
        device: 'cuda' or 'cpu'; ball_mode/tile_*: see YOLODetector
        ball_smoother: 'kalman' (KalmanBallTracker) or 'linear' (BallTrajectory)
        calibration: Optional JSON file overriding the built-in pitch calibration
        max_frames: Process only the first N frames
        progress_cb: Optional callback(phase, current, total) for pass1/pass2 progress
        log: Logging callable (pass e.g. a no-op from a web app)

    Returns:
        Dict with 'stats', 'video', 'stats_path', 'heatmaps', 'ball_csv', 'motion_csv',
        'fps', 'frames', 'model'
    """
    log("=" * 60)
    log("Football Tactical Analysis AI - Phase 7")
    log("=" * 60)
    log(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
        + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    log(f"\nVideo: {input_path} | {width}x{height} @ {fps:.2f} FPS | {total_frames} frames")

    log("\nInitializing modules...")
    model_path = Path(model_path) if model_path else default_model()
    if not model_path.exists():
        model_path = Path('yolo11m.pt')
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    detector = YOLODetector(model_path=str(model_path), device=device,
                            ball_mode=ball_mode, tile_size=tile_size, tile_imgsz=tile_imgsz)
    detector.load_model()
    tracker = ObjectTracker(fps=fps, class_names=detector.model.names)
    tracker.initialize()
    team_assigner = TeamAssigner()
    tactical_map = TacticalMap()
    if calibration:
        tactical_map.load_calibration(calibration)
    colors = ColorScheme()

    # ---------------------------------------------------------------- pass 1
    records = run_pass1(cap, total_frames, detector, tracker, team_assigner, CameraMotionEstimator(),
                        progress_cb)
    cap.release()
    n = len(records)

    # ---------------------------------------------------------------- whole-clip processing
    log("")
    team_assigner.fit()
    camera = tactical_map.set_camera_motion([r['motion'] for r in records])
    log(f"\nCamera: {len(camera['segments'])} segment(s), homography valid in {camera['valid_frames']}/{n} frames")
    for seg in camera['segments']:
        if seg['reference_frame'] is not None:
            log(f"  - frames {seg['start']}-{seg['end']}: calibrated at frame {seg['reference_frame']}, "
                f"calibration RMSE {seg['calibration_rmse_m']:.2f} m ({seg['calibration_points']} points)")

    positions = {}
    for i, r in enumerate(records):
        r['pitch'] = tactical_map.transform_points([t.foot_position for t in r['tracks']], i)
        for k, t in enumerate(r['tracks']):
            positions.setdefault(t.track_id, {})[i] = None if r['pitch'] is None else tuple(r['pitch'][k])

    goalkeepers, officials = team_assigner.resolve_roles(positions)
    for i, r in enumerate(records):
        r['labels'] = team_assigner.labels_for_frame(i, [t.track_id for t in r['tracks']])
    log(f"Roles: {len(goalkeepers)} goalkeeper track(s) {goalkeepers}, {len(officials)} official track(s)"
        + (f", officials' kit (BGR) {team_assigner.official_color_bgr()}" if team_assigner.official_color is not None else ""))
    if team_assigner.overrides:
        log(f"  - {len(team_assigner.overrides)} track(s) reclassified referee -> player by kit color: "
            f"{team_assigner.overrides}")

    # Real-world speed and distance per track (uses the video's actual FPS)
    seg_of_frame = np.zeros(n, dtype=int)
    for k, seg in enumerate(camera['segments']):
        seg_of_frame[seg['start']:seg['end'] + 1] = k
    kinematics = PlayerKinematics(fps)
    kinematics.compute(positions, seg_of_frame, tactical_map.is_inside_pitch)
    log(f"Kinematics: {len(kinematics.summary)} track(s) with usable position runs")

    ball_pitch = []
    for i, r in enumerate(records):
        pts = tactical_map.transform_points([c for c, _ in r['balls']], i)
        ball_pitch.append(None if pts is None else [tuple(p) for p in pts])
    chosen_balls, static_objects = select_ball_candidates([r['balls'] for r in records], ball_pitch)
    for s in static_objects:
        log(f"Static off-pitch ball look-alike removed at pitch {s['pitch_position']} m "
            f"(frames {s['first_frame']}-{s['last_frame']}, seen in {s['frames']})")
    smoother = (KalmanBallTracker(max_predict=15) if ball_smoother == 'kalman'
                else BallTrajectory())
    ball_df, ball_stats = smoother.process(chosen_balls)
    ball_stats['frames_with_candidates'] = sum(bool(r['balls']) for r in records)
    ball_stats['static_objects_removed'] = static_objects
    log(f"Ball: { {k: v for k, v in ball_stats.items() if k != 'static_objects_removed'} }")

    # Ball stillness for dead-ball possession gating: speed in pixel space (robust
    # to homography jitter) and pitch position for the boundary test.
    ball_px_speeds = np.full(n, np.nan)
    ball_pitch_sm = np.full((n, 2), np.nan)
    for i in range(n):
        bx, by = ball_df.loc[i, 'x'], ball_df.loc[i, 'y']
        if not np.isnan(bx):
            p = tactical_map.transform_point(float(bx), float(by), i)
            if p is not None:
                ball_pitch_sm[i] = p
    px_step = np.linalg.norm(np.diff(ball_df[['x', 'y']].to_numpy(), axis=0), axis=1)
    ball_px_speeds[1:] = pd.Series(px_step * fps).rolling(9, center=True, min_periods=1).mean()

    # ---------------------------------------------------------------- pass 2
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    cap = cv2.VideoCapture(str(input_path))

    aliases = {tid: f'GK_{team}' for tid, team in goalkeepers.items()}
    analyzer = TacticalAnalyzer(fps=fps, player_aliases=aliases)
    heatmap = TeamHeatmap(tactical_map)
    team_shape = TeamShape()
    pass_display_frames = int(2 * fps)  # Show a pass arrow on the map for 2 seconds
    recent_passes = []
    ball_trail = []
    frames_in_control = 0
    live_cum, live_team = {}, {}  # track_id -> distance so far / last team seen (leaderboard)

    for frame_idx in _progress_iter(range(n), "Pass 2/2 (analyze+render)", progress_cb, 'pass2'):
        ret, frame = cap.read()
        if not ret:
            break
        r = records[frame_idx]
        tracks, labels, pitch = r['tracks'], r['labels'], r['pitch']
        row = ball_df.loc[frame_idx]
        ball_pos = None if np.isnan(row['x']) else (float(row['x']), float(row['y']))

        bs, bp = ball_px_speeds[frame_idx], ball_pitch_sm[frame_idx]
        analysis = analyzer.update(frame_idx, ball_pos, tracks, labels,
                                   ball_px_speed=float(bs) if np.isfinite(bs) else None,
                                   ball_pitch_pos=tuple(bp) if np.isfinite(bp).all() else None)
        frames_in_control += analysis['in_control']

        speeds, hull_players = {}, []
        map_players, map_ball, message = [], None, None
        if pitch is None:
            message = "No camera calibration"
        else:
            for t, p in zip(tracks, pitch):
                if t.track_id in live_cum or t.track_id in kinematics.cum_distance:
                    live_cum[t.track_id] = kinematics.cum_distance.get(t.track_id, {}).get(frame_idx,
                                                                                         live_cum.get(t.track_id, 0.0))
                if labels[t.track_id] in ('team_a', 'team_b'):
                    live_team[t.track_id] = labels[t.track_id]
                if not tactical_map.is_inside_pitch(p):
                    continue
                map_players.append({'pitch_position': tuple(p), 'team': labels[t.track_id], 'track_id': t.track_id,
                                    'has_ball': t.track_id == analysis['holder_id'],
                                    'is_goalkeeper': t.track_id in goalkeepers})
                heatmap.add(labels[t.track_id], tuple(p))
                if labels[t.track_id] in ('team_a', 'team_b'):
                    hull_players.append((labels[t.track_id], tuple(p)))
                speed = kinematics.speed_at(t.track_id, frame_idx)
                if speed is not None and speed <= kinematics.max_speed_kmh:
                    speeds[t.track_id] = speed
            if ball_pos is not None:
                map_ball = tactical_map.transform_point(*ball_pos, frame_idx)
                map_ball = map_ball if (map_ball is not None and tactical_map.is_inside_pitch(map_ball)) else None
        if map_ball is not None:
            ball_trail.append(map_ball)
            if len(ball_trail) > 15:
                ball_trail = ball_trail[-15:]

        hulls = team_shape.update(frame_idx, hull_players)
        totals, teams_of = {}, {}
        for tid, dist in live_cum.items():
            name = aliases.get(tid, str(tid))
            totals[name] = totals.get(name, 0.0) + dist
            teams_of[name] = name[3:] if name.startswith('GK_') else live_team.get(tid, 'unknown')
        leaderboard = sorted(((n, teams_of[n], d) for n, d in totals.items()), key=lambda e: -e[2])[:5]

        if analysis['new_pass'] is not None and pitch is not None:
            p = analysis['new_pass']
            recent_passes.append({'frame': frame_idx, 'team': p.team,
                                  'from_pos': tactical_map.transform_point(*p.from_pos, frame_idx),
                                  'to_pos': tactical_map.transform_point(*p.to_pos, frame_idx)})
        recent_passes = [p for p in recent_passes if frame_idx - p['frame'] < pass_display_frames]

        out.write(annotate_frame(frame, tracks, labels, ball_pos, row['source'] != 'detected', analysis,
                                 analyzer.get_pass_stats(), None, colors, set(goalkeepers),
                                 speeds=speeds, leaderboard=leaderboard,
                                 minimap_players=map_players, minimap_ball=map_ball,
                                 minimap_passes=recent_passes, minimap_hulls=hulls,
                                 ball_trail=ball_trail, minimap_message=message))

    cap.release()
    out.release()

    # ---------------------------------------------------------------- outputs
    prefix = str(output_path.with_suffix(''))
    heatmap_paths = heatmap.save(prefix)
    ball_df.to_csv(f"{prefix}_ball_trajectory.csv")
    pd.DataFrame([{'frame': m.frame, 'dx': m.dx, 'dy': m.dy, 'inliers': m.inliers, 'tracked': m.tracked,
                   'hist_diff': m.hist_diff, 'new_segment': m.new_segment,
                   'homography_valid': tactical_map.frame_homographies[m.frame] is not None}
                  for m in (r['motion'] for r in records)]).to_csv(f"{prefix}_camera_motion.csv", index=False)

    stats = analyzer.export()
    stats['roles'] = {'track_ids_per_label': team_assigner.get_team_stats(),
                      'goalkeepers': {str(k): v for k, v in goalkeepers.items()},
                      'official_track_ids': sorted(officials),
                      'referee_overrides': {str(k): v for k, v in team_assigner.overrides.items()},
                      'outlier_track_report': {str(k): v for k, v in team_assigner.role_report.items()}}
    stats['ball'] = {**ball_stats, 'frames_with_player_in_control': frames_in_control,
                     'dead_ball_frames': stats.pop('dead_ball_frames'),
                     'dead_ball_seconds': stats.pop('dead_ball_seconds')}
    stats['camera'] = camera
    stats['model'] = {'weights': str(model_path), 'fine_tuned': detector.is_finetuned,
                      'person_imgsz': detector.person_imgsz, 'ball_imgsz': detector.ball_imgsz,
                      'ball_conf': detector.ball_conf, 'ball_mode': detector.ball_mode,
                      'tile_size': detector.tile_size, 'tile_imgsz': detector.tile_imgsz}
    per_track = {str(tid): s for tid, s in kinematics.summary.items()}
    per_person = {}
    for tid, s in kinematics.summary.items():
        name = aliases.get(tid, str(tid))
        a = per_person.setdefault(name, {'distance_m': 0.0, 'max_speed_kmh': 0.0, 'seconds': 0.0})
        a['distance_m'] = round(a['distance_m'] + s['distance_m'], 1)
        a['max_speed_kmh'] = round(max(a['max_speed_kmh'], s['max_speed_kmh']), 1)
        a['seconds'] = round(a['seconds'] + s['seconds'], 1)
    stats['physical'] = {
        'fps': fps,
        'per_track': per_track,
        'per_person': per_person,
        'top_distance': sorted(per_person.items(), key=lambda kv: -kv[1]['distance_m'])[:10],
        'top_speed': sorted(per_person.items(), key=lambda kv: -kv[1]['max_speed_kmh'])[:10],
    }
    stats['team_shape'] = team_shape.summary()
    stats_path = Path(f"{prefix}_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))

    log("\n" + "=" * 60)
    log("Match statistics")
    log("=" * 60)
    poss = stats['possession_percent']
    log(f"Possession:  Team A {poss['team_a']:.1f}%  |  Team B {poss['team_b']:.1f}%")
    for team, s in stats['pass_stats'].items():
        log(f"{team}: {s['passes']} successful passes, {s['turnovers']} turnovers")
    log(f"Frames with a player in control of the ball: {frames_in_control}/{n}")
    log("\nTop distance covered:")
    for name, s in stats['physical']['top_distance'][:5]:
        log(f"  {name}: {s['distance_m']:.0f} m (max {s['max_speed_kmh']} km/h)")
    for team, s in stats['team_shape'].items():
        log(f"{team} compactness: median hull area {s['median_area_m2']} m^2, "
            f"width {s['mean_width_m']} m, depth {s['mean_depth_m']} m")
    log(f"\nVideo:     {output_path}")
    log(f"Stats:     {stats_path}")
    log(f"Heatmaps:  {', '.join(heatmap_paths)}")
    log(f"CSV:       {prefix}_ball_trajectory.csv, {prefix}_camera_motion.csv")

    return {'stats': stats, 'video': str(output_path), 'stats_path': str(stats_path),
            'heatmaps': heatmap_paths, 'ball_csv': f"{prefix}_ball_trajectory.csv",
            'motion_csv': f"{prefix}_camera_motion.csv",
            'fps': fps, 'frames': n, 'model': str(model_path)}


def main():
    args = parse_arguments()
    try:
        run_pipeline(args.input, args.output, model_path=args.model, device=args.device,
                     ball_mode=args.ball_mode, tile_size=args.tile_size, tile_imgsz=args.tile_imgsz,
                     ball_smoother=args.ball_smoother, calibration=args.calibration,
                     max_frames=args.max_frames)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
