# Football_AI_Thesis - agent notes

## Environment (Windows, PowerShell)
- Virtual env: `venv\` - script activation is blocked by execution policy, call `.\venv\Scripts\python.exe` directly.
- GPU: NVIDIA RTX 4060 Laptop, PyTorch `2.5.1+cu121`. Check: `.\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`
- Never install plain `torch` from PyPI (CPU-only on Windows); requirements.txt uses the cu121 extra index.
- YOLO11 weights file name is `yolo11m.pt` (not `yolov11m.pt`).
- `supervision` is pinned `<0.31` because `sv.ByteTrack` is removed there.

## Run / verify
- Full run (two passes, see main.py docstring): `.\venv\Scripts\python.exe main.py` -> `data/output/test_output_phase4.mp4` + stats/heatmaps/CSVs
- Quick smoke test: `.\venv\Scripts\python.exe main.py --max-frames 400 --output data/output/_smoke.mp4`
  (keep >= 301 frames so the frame-300 calibration is included)
- No automated test suite yet; verify by inspecting output frames and `data/output/*_stats.json`.
- Homography check: project pitch lines back into frames with `np.linalg.inv(tactical_map.homography_for(i))` and compare visually.

## Known behaviour
- COCO "sports ball" fires on hi-vis yellow vests/kits; `YOLODetector._is_plausible_ball` filters these
  (ball box median HSV saturation ~85-115 vs ~175-185 for hi-vis).
- Calibration (`tactical_map.CALIBRATIONS`) is hand-measured for test_match.mp4 only (frames 300, 1175, 1500).
  A different video needs its own calibration points.
- test_match.mp4 has a replay wipe at frames ~1288-1295 -> two camera segments.
- Optical flow on grass alone drifted badly; `CameraMotionEstimator(use_background=True)` (default) uses stands/boards too.
- Team kits must be fitted on the whole clip (TeamAssigner.fit after pass 1): kits fitted on early frames
  misclassified shadowed players as referees. Do not fall back to whole-torso colour for grass-coloured crops
  (tested: creates a bogus "grass kit" cluster).
- Full Phase 4 run (COCO weights): ~210 s for the 1897-frame clip; Phase 5 (fine-tuned, ball @2560): ~270 s.

## Fine-tuned model (Phase 5)
- Train: `.\venv\Scripts\python.exe train.py` (needs `if __name__ == '__main__'` guard on Windows; ~50 epochs, 4.3 GB VRAM at batch 8).
- `datasets/data.yaml` uses an absolute `path:`; Roboflow's original `../train/images` paths do not resolve.
- main.py picks the newest `runs/detect/*/weights/best.pt` automatically. A new training run creates `train2`, ...
- Detector reads class names from the weights: COCO ('person', 'sports ball') or fine-tuned
  ('ball', 'goalkeeper', 'player', 'referee'). Fine-tuned: persons @640, ball @2560 (measured, see detector.py).
- test_match.mp4 has ball look-alikes the fine-tuned model detects: a towel inside the left goal (pitch ~(-0.9, 38.9))
  and objects beyond the touchline; `tracker.select_ball_candidates` removes static OFF-pitch objects only
  (a corner-kick ball sits still at the corner flag for ~14 s, frames ~1180-1600).

## Phase 6 (SAHI ball, role override, kinematics)
- `--ball-mode sliced+full` (default): 640 px tiles over the pitch upscaled to 960, batched on GPU,
  boxes remapped + NMS. Probe on 475 frames: 429 detections vs 388 full@2560 alone, ~270 vs ~90 ms/frame.
- Samples now store (frame, Lab, yolo_role); `TeamAssigner._apply_color_overrides` rewrites tracks whose
  median kit sits <= 1.5x median in-kit distance from a team kit, even if labelled 'referee'.
- `PlayerKinematics`: per-track runs split at gaps>3 frames and camera cuts; median+mean smoothing;
  steps > 40 km/h dropped. `TeamShape`: convex hull per team per frame (>=5 visible players).
- Smoke test: `--max-frames 400` covers the whole pipeline (hulls, leaderboard, speed tags) in ~1.5 min.

## Phase 7 (Kalman ball) and Phase 8 (hard-negative mining)
- `extract_hard_negatives.py` mines relabel candidates: saves frames where the RAW model
  (no pitch mask / team override) predicts 'referee' or no 'ball'; `--ball-only` isolates
  condition B, `--cooldown N` skips frames after each save. Writes images + manifest.csv.
- Officials are visible in ~every frame, so referee saves saturate at the cooldown cap.
- Raw ball pass @2560/conf 0.1 almost always emits SOME box (towel/look-alikes count);
  'zero ball detections' is rare. Raise `--conf-ball` to mine frames where only weak FPs fire.
- `auto_augment_train.py`: offline alternative to relabeling - targeted augs (MotionBlur/
  GaussianBlur/Downscale/brightness/noise), 2x copies for ball images, originals hard-linked,
  val split untouched; then fine-tunes from best.pt -> runs/detect/phase8_aug.
  Installed with --no-deps (pip would swap opencv-python for headless); needs albucore==0.0.23.
- phase8_aug results: val mAP50 0.880 (ball 0.564->0.592). On test_match: ball detected 1873/1897
  (was 1459), 0 missing, BUT reinitializations 255->508 and ~36% of co-detected frames moved
  >100px - higher recall, more candidate flicker. Ultralytics double-nested the run dir
  (runs/detect/runs/detect/phase8_aug) when project= was passed; moved back manually.
- Dashboard: `python -m streamlit run app.py`. st.video needs browser codecs - cv2 'VP80'
  -> .webm works (openh264 DLL missing, external ffmpeg blocked from file I/O here).

## Phase 7 (Kalman ball tracking)
- `KalmanBallTracker` (tracker.py): cv2.KalmanFilter [x,y,vx,vy] + RTS backward smoother; predict-only
  for <=15 consecutive misses; innovation >150 px -> re-init (velocity from gap displacement), re-inits
  break the RTS segment. Default via `--ball-smoother kalman`; `linear` keeps Phase 6 ablation.
- Ball sources in the CSV: 'detected' | 'kalman' | 'missing' (was 'interpolated').

## Dead-ball gating (Phase 8 final)

- Dead ball = ball pixel-speed < 60 px/s for >1 s AND pitch position within 2 m of
  a boundary line or outside the pitch. Implemented in TacticalAnalyzer.update
  (ball_px_speed, ball_pitch_pos params); main.py computes px speed from the
  smoothed trajectory and pitch pos via tactical_map.transform_point.
- IMPORTANT: stillness is measured in PIXEL space, not meters. Per-frame homography
  jitter makes a stationary ball look like it moves ~0.5-1 m/s in pitch coords, so
  a m/s stillness threshold never fires. A static image cannot lie.
- Verified on test_match.mp4: 32 dead frames (~1.1 s) at the corner-flag stretch
  (frames ~1140-1260); the low-speed stretches near x~1570 px stay in play because
  they map inside the pitch. Possession 16.2/83.8.
