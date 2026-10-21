# Football Tactical Analysis AI

A comprehensive Computer Vision system for advanced Football (Soccer) Tactical Analysis. This project goes beyond simple tracking to include possession calculation, pass detection, and tactical map generation.

## Project Structure

```
Football_AI_Thesis/
├── src/                    # Source code modules
│   ├── detector.py         # YOLO11 detection (players, referee, ball)
│   ├── tracker.py          # Multi-object tracking (ByteTrack/DeepSORT)
│   ├── team_assigner.py    # Team assignment based on jersey colors
│   ├── analyzer.py         # Tactical analysis (possession, passes, player metrics)
│   └── tactical_map.py     # Homography for 2D tactical pitch mapping
├── utils/                  # Helper functions
│   └── visualization.py    # Drawing annotations and visualizations
├── data/                   # Data directories
│   ├── input/             # Input videos
│   ├── output/            # Annotated videos and statistics
│   └── weights/           # Custom trained weights
├── models/                 # Pre-trained model weights
│   └── yolo11m.pt         # YOLO11 medium model (COCO-pretrained)
├── venv/                   # Python virtual environment
├── requirements.txt        # Python dependencies
└── main.py                # Main entry point
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

2. Install dependencies (requirements.txt pulls the CUDA 12.1 build of PyTorch):
```bash
pip install -r requirements.txt
```

3. Verify the GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

4. YOLO11 weights are in `models/yolo11m.pt` (official file name: `yolo11m.pt`, no "v").

## Usage

```bash
python main.py --input data/input/test_match.mp4 --output data/output/test_output_phase7_kalman.mp4
```

Optional arguments:
- `--model`: Path to YOLO weights (default: newest `runs/detect/*/weights/best.pt`, else `models/yolo11m.pt`)
- `--ball-mode`: `sliced+full` (default), `sliced` or `full` — SAHI-style tiled ball inference (640 px tiles upscaled to 960) on top of the full-frame pass
- `--ball-smoother`: `kalman` (default: Kalman filter + RTS smoother, predicts up to 15 frames) or `linear` (interpolation + moving average, for ablation)
- `--tile-size`, `--tile-imgsz`: tile geometry for sliced ball inference (default 640 / 960, measured on test_match.mp4)
- `--calibration`: JSON file `{frame: [[[video_x, video_y], [pitch_x, pitch_y]], ...]}` (overrides the built-in calibration)
- `--device`: `cuda` (default; fails loudly if CUDA is unavailable) or `cpu`
- `--max-frames`: Only process the first N frames (quick tests)

Outputs (next to the video):
- `<name>_stats.json`: possession, passes, turnovers, roles (goalkeepers/officials with evidence,
  referee→player color overrides), ball coverage, camera segments, per-track/per-person speed &
  distance (m, km/h), top-distance/top-speed leaderboards, team compactness (hull area/width/depth)
- `<name>_heatmap_team_a.png`, `<name>_heatmap_team_b.png`: team heatmaps
- `<name>_ball_trajectory.csv`: raw vs cleaned ball position per frame (detected / interpolated / missing)
- `<name>_camera_motion.csv`: per-frame camera shift (dx, dy), optical-flow inliers, cut detection

## Fine-tuning (train.py)

```bash
python train.py            # yolo11m.pt on datasets/data.yaml, imgsz 640, batch 8, AMP, 50 epochs
```

Dataset: Roboflow "football-players-and-ball" (CC BY 4.0), classes ball / goalkeeper / player / referee,
708 train / 202 valid / 101 test images. Training plots for the report (results.png, confusion matrix,
PR/F1 curves) are written to `runs/detect/train/`.

Validation mAP50 of `best.pt` by inference size:

| class | @640 | @1280 | @1920 |
|---|---|---|---|
| ball | 0.564 | 0.671 | 0.555 |
| goalkeeper | 0.965 | 0.887 | 0.672 |
| player | 0.989 | 0.980 | 0.932 |
| referee | 0.964 | 0.936 | 0.655 |

The detector therefore runs people at 640 and the ball in a second pass. Phase 6 added
SAHI-style sliced inference: the pitch region is cut into overlapping 640 px tiles that are
inferred at 960. On 475 sampled frames of test_match.mp4 the ball was found in 388 frames
(full @2560), 418 (slices only) and 429 of 475 for the union (`--ball-mode sliced+full`,
the default) — at the cost of roughly 3x the runtime of the ball pass.

## Pipeline

Two passes, because ball interpolation and goalkeeper identification need the whole clip:

1. **Pass 1 (GPU):** YOLO11 detection (persons @640, ball sliced+full) with pitch masking → ByteTrack → jersey-colour + YOLO role samples → optical-flow camera motion
2. **Whole clip:** per-frame homographies; team kits (trimmed KMeans); referee→player color override; goalkeeper/official roles (position + YOLO role votes); ball trajectory cleaning; speed/distance per track from smoothed pitch coordinates
3. **Pass 2:** possession + passes, rendering (PiP tactical map with team convex hulls + heatmaps, speed tags, top-distance leaderboard, scoreboard)

## Module Overview

### detector.py
- YOLO11 (COCO or fine-tuned) detection of persons and the ball, on CUDA with FP16
- SAHI-style sliced ball inference: overlapping tiles over the pitch region, one batched GPU call, boxes remapped to frame coordinates and merged with NMS (`ball_mode`: full / sliced / sliced+full)
- Pitch masking: green HSV mask (largest grass region, holes filled); people whose feet are not on the grass are dropped
- Ball false-positive filter: inside the pitch mask, surrounded by grass, not on a person's torso, not strongly saturated (hi-vis vests)

### tracker.py
- `ObjectTracker`: ByteTrack (supervision) for persistent player IDs
- `BallTrajectory`: speed-consistency outlier rejection, pandas interpolation of gaps < 20 frames, centered moving-average smoothing
- `KalmanBallTracker` (Phase 7, default): constant-velocity Kalman filter (`cv2.KalmanFilter`, state [x, y, vx, vy]) + Rauch-Tung-Striebel backward smoother. Detections are corrections; misses are predicted for up to 15 consecutive frames; innovations > 150 px re-initialize the filter (replay wipes, off-camera spells); the RTS pass produces the final jitter-free trajectory used downstream

### team_assigner.py
- Torso colour (median Lab of non-grass pixels, lightness half-weighted so sun/shadow don't matter) → trimmed KMeans on the whole clip into the two team kits; colour outliers = official/goalkeeper candidates
- Centered sliding-window majority vote per track
- **Colour override** (Phase 6): a track whose median kit colour sits well inside a team kit is a player even if the model (or the per-frame colour votes) called it 'referee' — fixes players misclassified near the touchline
- `resolve_roles`: goalkeeper = outlier (or unmeasurable, e.g. lime kit) track that is the deepest person on its half most of the time, or that the model mostly calls 'goalkeeper' → assigned to the team defending that goal; colour-unmeasurable tracks the model mostly calls 'referee' stay officials; all other outliers are non-players (referee, assistant referees, touchline staff) and are ignored by the analyzer

### analyzer.py
- **Possession**: nearest player's feet to the ball within 0.6 × player height, confirmed after 3 observations; possession % by frames
- **Passes**: holder change within a team = successful pass, across teams = turnover
- Officials are never considered; goalkeepers count for their team
- **`PlayerKinematics`** (Phase 6): per-track speed (km/h) and distance (m) from smoothed pitch coordinates — continuous runs split at gaps > 3 frames and camera cuts, median+mean desmoothing, steps > 40 km/h rejected as tracking/projection errors
- **`TeamShape`** (Phase 6): per-frame convex hull of a team's visible players → controlled area (m²), width and depth (m)

### tactical_map.py
- To-scale 2D pitch, video → pitch homography from hand-measured pitch-line intersections
- `CameraMotionEstimator`: Lucas-Kanade optical flow (players masked), RANSAC homography per frame, cut detection
- Per-frame homographies by chaining camera motion to a calibration frame; one calibration per camera segment
- `render(..., hulls=...)`: semi-transparent team-coloured convex-hull polygons (controlled area)

### visualization.py
- Ellipses, IDs, GK/REF tags, speed (km/h) next to the ID, ball marker (hollow = interpolated), holder marker
- PiP tactical map with team heatmaps and convex hulls, possession scoreboard, top-distance leaderboard

## Hardware Requirements

- GPU: NVIDIA RTX 4060 (or similar CUDA-capable GPU)
- CPU: AMD Ryzen 7 8845H (or equivalent)
- RAM: 16GB+ recommended
- OS: Windows 10/11

## Future Enhancements (Phase 2+)

- Fine-tune YOLO model on football-specific dataset
- Implement advanced pass classification with ML
- Add formation detection and analysis
- Real-time analysis with streaming video
- Web-based visualization dashboard

## License

This project is licensed under the MIT License.
