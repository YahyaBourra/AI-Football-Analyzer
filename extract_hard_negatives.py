"""Phase 8 active learning: mine hard-negative frames for Roboflow relabeling.

Runs the fine-tuned YOLO model directly on every frame (no pitch mask, no
team-assignment correction) and saves frames where the raw model fails:

  A) it detects a 'referee'  -> usually a player at/near the boundary the
     pipeline would otherwise filter or color-override away
  B) it detects no 'ball'    -> tiny/blurred ball frames the SAHI pass
     would have to rescue downstream

A cooldown skips COOLDOWN frames after each save so we don't extract
near-identical consecutive frames.

    .\\venv\\Scripts\\python.exe extract_hard_negatives.py
    .\\venv\\Scripts\\python.exe extract_hard_negatives.py --cooldown 30 --conf-ball 0.15
"""

import argparse
import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

DEFAULT_VIDEO = 'data/input/test_match.mp4'
DEFAULT_MODEL = 'runs/detect/train/weights/best.pt'
DEFAULT_OUT = 'data/hard_negatives'

# Same operating points as the production pipeline (see detector.py)
PERSON_IMGSZ, PERSON_CONF = 640, 0.4
BALL_IMGSZ, BALL_CONF = 2560, 0.1


def main():
    ap = argparse.ArgumentParser(description='Extract hard-negative frames for relabeling')
    ap.add_argument('--video', default=DEFAULT_VIDEO)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--cooldown', type=int, default=15,
                    help='frames to skip after each saved frame')
    ap.add_argument('--ball-only', action='store_true',
                    help='only mine frames where the ball is missed (condition A off); '
                         'officials are on camera almost every frame, so referee saves '
                         'otherwise starve the ball condition via the cooldown')
    ap.add_argument('--conf-ball', type=float, default=BALL_CONF)
    ap.add_argument('--conf-person', type=float, default=PERSON_CONF)
    ap.add_argument('--max-frames', type=int, default=None, help='debug limit')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(args.model)
    names = model.names                      # {id: name} from the weights
    ball_id = next(k for k, v in names.items() if v == 'ball')
    ref_id = next(k for k, v in names.items() if v == 'referee')
    print(f'Model {args.model} on {device} | classes: {names}')

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    print(f'Processing {total} frames of {args.video} -> {out_dir}/')

    manifest_path = out_dir / 'manifest.csv'
    saved = {'referee': 0, 'ball_missing': 0}
    n_saved = 0
    cooldown = 0
    frame_idx = 0

    with manifest_path.open('w', newline='') as mf:
        writer = csv.writer(mf)
        writer.writerow(['frame', 'reasons', 'n_referee', 'n_ball', 'file'])

        while frame_idx < total:
            ok, frame = cap.read()
            if not ok:
                break
            idx = frame_idx
            frame_idx += 1

            if cooldown > 0:
                cooldown -= 1
                continue

            # One fused pass would be faster, but the production pipeline runs
            # persons and ball at different imgsz/conf - mirror that here.
            if args.ball_only:
                n_ref = 0
            else:
                res_p = model.predict(frame, imgsz=PERSON_IMGSZ, conf=args.conf_person,
                                      device=device, verbose=False)[0]
                n_ref = int((res_p.boxes.cls == ref_id).sum()) if res_p.boxes is not None else 0

            res_b = model.predict(frame, imgsz=BALL_IMGSZ, conf=args.conf_ball,
                                  device=device, verbose=False)[0]
            n_ball = int((res_b.boxes.cls == ball_id).sum()) if res_b.boxes is not None else 0

            reasons = []
            if n_ref > 0 and not args.ball_only:
                reasons.append('referee')
            if n_ball == 0:
                reasons.append('ball_missing')
            if not reasons:
                continue

            fname = f'f{idx:05d}_{"+".join(reasons)}.jpg'
            cv2.imwrite(str(out_dir / fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            writer.writerow([idx, '+'.join(reasons), n_ref, n_ball, fname])
            for r in reasons:
                saved[r] += 1
            n_saved += 1
            cooldown = args.cooldown

            if n_saved % 25 == 0:
                print(f'  frame {idx}/{total} | saved so far: {n_saved}')

    cap.release()
    print(f'\nDone. {n_saved} hard-negative frames -> {out_dir}/')
    print(f'  referee detections : {saved["referee"]} frames')
    print(f'  ball missing       : {saved["ball_missing"]} frames')
    print(f'  manifest           : {manifest_path}')


if __name__ == '__main__':
    main()
