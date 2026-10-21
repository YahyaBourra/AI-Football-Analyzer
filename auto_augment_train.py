"""Phase 8: programmatic hard-negative training via targeted augmentation.

Instead of manually relabeling extracted hard negatives in Roboflow, this
script synthesizes the model's known failure modes directly into the existing
training set:

  - MotionBlur / GaussianBlur  -> fast-moving tiny ball (motion blur misses)
  - RandomBrightnessContrast   -> shadow/lighting cases (referee confusion)
  - GaussNoise + Downscale     -> low-signal frames where the ball is ~5 px

Ball-containing images get --aug-ball augmented copies, others get --aug-other.
Originals are hard-linked into the new dataset (no extra disk space), the
validation split is NOT augmented (honest mAP), and YOLO fine-tuning is then
launched from the current best.pt.

    .\\venv\\Scripts\\python.exe auto_augment_train.py
    .\\venv\\Scripts\\python.exe auto_augment_train.py --epochs 15 --skip-train
"""

import argparse
import os
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent
SRC_YAML = ROOT / 'datasets' / 'data.yaml'
DST_DIR = ROOT / 'data' / 'augmented_dataset'
START_WEIGHTS = ROOT / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
BALL_CLASS = 0


def build_transform() -> A.Compose:
    """Targeted augmentation for our failure modes. Pixel-only: boxes unchanged."""
    return A.Compose([
        # Fast ball / camera pan blur - the dominant miss mode
        A.OneOf([
            A.MotionBlur(blur_limit=(13, 35), p=1.0),
            A.GaussianBlur(blur_limit=(7, 21), sigma_limit=(0.5, 3.0), p=1.0),
        ], p=0.55),
        # Tiny ball: shrink effective resolution then back up
        A.Downscale(scale_range=(0.35, 0.6), interpolation_pair={'downscale': cv2.INTER_LINEAR, 'upscale': cv2.INTER_LINEAR}, p=0.25),
        # Shadow / lighting (Chelsea-in-shadow -> 'referee' bug)
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.6),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), p=0.15),
        # Sensor/compression noise on dark broadcast frames
        A.GaussNoise(std_range=(0.02, 0.10), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],
                                min_visibility=0.0))


def read_yolo_label(path: Path):
    boxes, cls = [], []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cls.append(int(parts[0]))
                boxes.append([float(x) for x in parts[1:5]])
    return boxes, cls


def write_yolo_label(path: Path, boxes, cls):
    path.write_text(''.join(f'{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n'
                            for c, b in zip(cls, boxes)))


def link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def augment_dataset(src_yaml: Path, dst_dir: Path, aug_ball: int, aug_other: int) -> Path:
    cfg = yaml.safe_load(src_yaml.read_text())
    src_root = Path(cfg['path'])
    src_images = src_root / cfg['train']
    src_labels = src_root / 'train' / 'labels'
    val_dir = src_root / cfg['val']

    dst_images = dst_dir / 'train' / 'images'
    dst_labels = dst_dir / 'train' / 'labels'
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    tf = build_transform()
    image_paths = sorted(p for p in src_images.iterdir()
                         if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    n_aug = 0
    for i, img_path in enumerate(image_paths):
        label_path = src_labels / (img_path.stem + '.txt')
        boxes, cls = read_yolo_label(label_path)
        has_ball = BALL_CLASS in cls
        copies = aug_ball if has_ball else aug_other

        # Original via hard link (zero extra space)
        link_or_copy(img_path, dst_images / img_path.name)
        if label_path.exists():
            link_or_copy(label_path, dst_labels / label_path.name)
        else:
            (dst_labels / label_path.name).write_text('')

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for k in range(copies):
            r = tf(image=img, bboxes=boxes, class_labels=cls)
            name = f'{img_path.stem}_aug{k}.jpg'
            cv2.imwrite(str(dst_images / name), r['image'],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            write_yolo_label(dst_labels / f'{img_path.stem}_aug{k}.txt',
                             r['bboxes'], r['class_labels'])
            n_aug += 1
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(image_paths)} images processed, {n_aug} augmented')

    out_yaml = dst_dir / 'data.yaml'
    out_yaml.write_text(
        f"path: {dst_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val: {val_dir.as_posix()}\n\n"          # unaugmented val = honest mAP
        f"nc: {cfg['nc']}\n"
        f"names: {cfg['names']}\n")
    print(f'Augmented dataset: {len(image_paths)} originals + {n_aug} augmented '
          f'= {len(image_paths) + n_aug} train images -> {dst_dir}')
    return out_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--aug-ball', type=int, default=2,
                    help='augmented copies per image containing a ball label')
    ap.add_argument('--aug-other', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-train', action='store_true')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not START_WEIGHTS.exists():
        sys.exit(f'Starting weights not found: {START_WEIGHTS}')
    yaml_path = augment_dataset(SRC_YAML, DST_DIR, args.aug_ball, args.aug_other)

    if args.skip_train:
        print('Skipping training (--skip-train).')
        return

    from ultralytics import YOLO
    model = YOLO(str(START_WEIGHTS))
    print(f'\nFine-tuning from {START_WEIGHTS} on {yaml_path} ...')
    model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, amp=True, device='cuda',
                project='runs/detect', name='phase8_aug', exist_ok=True)


if __name__ == '__main__':
    main()
