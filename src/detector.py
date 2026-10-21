"""
YOLO11 Detector Module

This module handles object detection for football match analysis using YOLO11.
It detects persons (players, goalkeepers, referees) and the ball, and supports two
kinds of weights, detected automatically from the model's class names:

- COCO-pretrained (models/yolo11m.pt): classes 'person' (0) and 'sports ball' (32).
- Fine-tuned (runs/detect/<run>/weights/best.pt, see train.py): classes 'ball',
  'goalkeeper', 'player', 'referee'. The model's role class is kept in each person
  detection ('role'), but downstream team/role assignment does not rely on it yet.

Key responsibilities:
- Load YOLO11 on the GPU (CUDA) and run inference on video frames
- Apply class-specific confidence thresholds and inference resolutions (the ball is small)
- Pitch masking: discard people whose feet are not on the grass (spectators, bench,
  coaches and stewards behind the advertising boards)
- Discard implausible ball candidates (outside the pitch mask, not surrounded by grass,
  on a person's torso, or strongly saturated like hi-vis vests)
- Return detections in a standardized format for downstream processing
"""

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class YOLODetector:
    """
    YOLO11-based detector for football entities (players, referees, ball).
    """

    PERSON_NAMES = {'person', 'player', 'goalkeeper', 'referee'}
    BALL_NAMES = {'sports ball', 'ball'}

    # Defaults per weight type: (person_imgsz, ball_imgsz, ball_conf)
    # COCO: the ball is only a few pixels wide; native resolution (1920) finds it far more
    #       reliably than 1280 (Phase 3 measurements).
    # Fine-tuned (trained at 640): validation mAP50 for the ball is 0.564 @640, 0.671 @1280,
    #       0.555 @1920, while people are best at the training size (player 0.989, goalkeeper
    #       0.965, referee 0.964 @640), so persons and ball are detected in two passes.
    #       On test_match.mp4, however, the ball is smaller (~5-9 px at native resolution vs ~15 px
    #       in the dataset), and ball recall on 475 sampled frames (after the plausibility filter)
    #       was 235 @1280, 321 @1920 and 388 @2560 (COCO @1920: 178), so the fine-tuned ball
    #       pass runs upscaled at 2560.
    # Phase 6 measured sliced (SAHI-style) inference on the same 475 frames: 640 px tiles at
    #       native size add nothing (384), tiles upscaled to 960 reach 418, and the union of
    #       slices @960 + full @2560 reaches 429 while keeping every full-frame detection.
    COCO_DEFAULTS = (1920, 1920, 0.05)
    FINETUNED_DEFAULTS = (640, 2560, 0.1)
    MAX_BALL_CANDIDATES = 5

    def __init__(self, model_path: str, person_conf: float = 0.4, ball_conf: Optional[float] = None,
                 device: str = 'cuda', person_imgsz: Optional[int] = None, ball_imgsz: Optional[int] = None,
                 ball_mode: str = 'full', tile_size: int = 640, tile_overlap: float = 0.2,
                 tile_imgsz: Optional[int] = None):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to the YOLO11 weights file (.pt)
            person_conf: Minimum confidence for person detections
            ball_conf: Minimum confidence for ball detections (None = default for the weight type)
            device: 'cuda' (default) or 'cpu'
            person_imgsz, ball_imgsz: Inference resolutions (None = default for the weight type)
            ball_mode: 'full' (whole frame at ball_imgsz), 'sliced' (SAHI-style tiles over the
                       pitch area) or 'sliced+full' (union of both, merged with NMS)
            tile_size: Tile side in original-frame pixels
            tile_overlap: Fractional overlap between neighbouring tiles
            tile_imgsz: Inference size per tile (None = tile_size, i.e. native resolution)
        """
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
        self.model_path = model_path
        self.person_conf = person_conf
        self.ball_conf = ball_conf
        self.device = device
        self.person_imgsz = person_imgsz
        self.ball_imgsz = ball_imgsz
        if ball_mode not in ('full', 'sliced', 'sliced+full'):
            raise ValueError(f"Unknown ball_mode {ball_mode!r}")
        self.ball_mode = ball_mode
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_imgsz = tile_imgsz or tile_size
        self.person_classes: List[int] = []
        self.ball_classes: List[int] = []
        self.is_finetuned = False
        self.half = device.startswith('cuda')
        self.model: Optional[YOLO] = None
        self.last_pitch_mask: Optional[np.ndarray] = None  # Reused by the camera-motion estimator

        # HSV range of the grass, used to reject detections that are off the pitch
        self.green_lower = np.array([30, 40, 40])
        self.green_upper = np.array([90, 255, 255])

    def load_model(self) -> None:
        """
        Load the YOLO model and move it to the target device.
        """
        weights = self.model_path if Path(self.model_path).exists() else 'yolo11m.pt'
        self.model = YOLO(weights)
        self.model.to(self.device)
        names = self.model.names
        self.person_classes = [i for i, n in names.items() if n in self.PERSON_NAMES]
        self.ball_classes = [i for i, n in names.items() if n in self.BALL_NAMES]
        if not self.person_classes or not self.ball_classes:
            raise RuntimeError(f"Model classes {names} contain no person and/or ball class")
        self.is_finetuned = 'sports ball' not in names.values()
        p_sz, b_sz, b_conf = self.FINETUNED_DEFAULTS if self.is_finetuned else self.COCO_DEFAULTS
        self.person_imgsz = self.person_imgsz or p_sz
        self.ball_imgsz = self.ball_imgsz or b_sz
        self.ball_conf = self.ball_conf if self.ball_conf is not None else b_conf

        device_name = torch.cuda.get_device_name(0) if self.device.startswith('cuda') else 'CPU'
        print(f"  - Model {self.model_path} ({'fine-tuned' if self.is_finetuned else 'COCO'}: "
              f"{', '.join(names[i] for i in self.person_classes + self.ball_classes)}) "
              f"loaded on {self.device} ({device_name}), FP16={self.half}")
        ball = {'full': f"full frame @{self.ball_imgsz}",
                'sliced': f"sliced {self.tile_size}px tiles ({self.tile_overlap:.0%} overlap) @{self.tile_imgsz}",
                'sliced+full': f"sliced {self.tile_size}px tiles @{self.tile_imgsz} + full frame @{self.ball_imgsz}"}
        print(f"  - Inference: persons @{self.person_imgsz} (conf {self.person_conf}), "
              f"ball: {ball[self.ball_mode]} (conf {self.ball_conf})")

    def _predict(self, frame: np.ndarray, classes: List[int], imgsz: int, conf: float):
        r = self.model.predict(frame, conf=conf, classes=classes, device=self.device, imgsz=imgsz,
                               quantize=16 if self.half else None, verbose=False)[0].boxes
        return r.xyxy.cpu().numpy().astype(int), r.conf.cpu().numpy(), r.cls.cpu().numpy().astype(int)

    @staticmethod
    def tile_grid(region: Tuple[int, int, int, int], tile: int, overlap: float) -> List[Tuple[int, int, int, int]]:
        """
        Overlapping tiles (x1, y1, x2, y2) covering `region`; the last row/column is aligned
        to the region edge so nothing is cut off.
        """
        x0, y0, x1, y1 = region
        stride = max(1, int(tile * (1 - overlap)))

        def starts(a, b):
            if b - a <= tile:
                return [a]
            s = list(range(a, b - tile, stride))
            return s + [b - tile]

        return [(x, y, min(x + tile, x1), min(y + tile, y1)) for y in starts(y0, y1) for x in starts(x0, x1)]

    def _predict_sliced(self, frame: np.ndarray, classes: List[int], conf: float,
                        region: Tuple[int, int, int, int]):
        """
        SAHI-style sliced inference: run the model on overlapping tiles of `region` (one GPU
        batch), shift the boxes back to frame coordinates and merge duplicates from the
        overlaps with NMS.
        """
        tiles = self.tile_grid(region, self.tile_size, self.tile_overlap)
        results = self.model.predict([frame[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles], conf=conf, classes=classes,
                                     device=self.device, imgsz=self.tile_imgsz,
                                     quantize=16 if self.half else None, verbose=False)
        boxes, scores, cls = [], [], []
        for (tx, ty, _, _), r in zip(tiles, results):
            b = r.boxes
            if len(b):
                boxes.append(b.xyxy + torch.tensor([tx, ty, tx, ty], device=b.xyxy.device))
                scores.append(b.conf)
                cls.append(b.cls)
        if not boxes:
            return np.empty((0, 4), int), np.empty(0, np.float32), np.empty(0, int)
        boxes, scores, cls = torch.cat(boxes), torch.cat(scores), torch.cat(cls)
        keep = torchvision.ops.batched_nms(boxes.float(), scores.float(), cls, iou_threshold=0.3)
        return (boxes[keep].cpu().numpy().astype(int), scores[keep].cpu().numpy(),
                cls[keep].cpu().numpy().astype(int))

    def _pitch_region(self, mask: np.ndarray, pad: int = 32) -> Tuple[int, int, int, int]:
        ys, xs = np.nonzero(mask[::4, ::4])
        h, w = mask.shape
        if len(xs) == 0:
            return 0, 0, w, h
        return (max(0, xs.min() * 4 - pad), max(0, ys.min() * 4 - pad),
                min(w, xs.max() * 4 + pad), min(h, ys.max() * 4 + pad))

    def _predict_ball(self, frame: np.ndarray, pitch_mask: np.ndarray):
        parts = []
        if self.ball_mode in ('sliced', 'sliced+full'):
            parts.append(self._predict_sliced(frame, self.ball_classes, self.ball_conf, self._pitch_region(pitch_mask)))
        if self.ball_mode in ('full', 'sliced+full'):
            parts.append(self._predict(frame, self.ball_classes, self.ball_imgsz, self.ball_conf))
        if len(parts) == 1:
            return parts[0]
        xyxy, confs, classes = (np.concatenate(a) for a in zip(*parts))
        if len(xyxy) == 0:
            return xyxy, confs, classes
        keep = torchvision.ops.nms(torch.tensor(xyxy, dtype=torch.float32), torch.tensor(confs), 0.3).numpy()
        return xyxy[keep], confs[keep], classes[keep]

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Perform detection on a single frame.

        Args:
            frame: Input BGR frame

        Returns:
            List of detection dictionaries, each containing:
            - bbox: [x1, y1, x2, y2] (pixel coordinates)
            - confidence: float (0-1)
            - class_id: int (model class index)
            - class_name: 'person' or 'ball'
            - role (persons): model class name ('person' for COCO; 'player', 'goalkeeper'
              or 'referee' for fine-tuned weights)
            All plausible ball candidates are returned (most confident first, at most
            MAX_BALL_CANDIDATES), so that whole-clip filters can choose between them.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        pitch_mask = self.last_pitch_mask = self.pitch_mask(frame)
        if self.ball_mode == 'full' and self.person_imgsz == self.ball_imgsz:
            xyxy, confs, classes = self._predict(frame, self.person_classes + self.ball_classes,
                                                 self.person_imgsz, min(self.person_conf, self.ball_conf))
        else:
            parts = [self._predict(frame, self.person_classes, self.person_imgsz, self.person_conf),
                     self._predict_ball(frame, pitch_mask)]
            xyxy, confs, classes = (np.concatenate(a) for a in zip(*parts))
        all_person_boxes = [b.tolist() for b, c, k in zip(xyxy, confs, classes)
                            if k in self.person_classes and c >= self.person_conf]
        persons, balls = [], []
        for bbox, conf, cls in zip(xyxy, confs, classes):
            det = {'bbox': bbox.tolist(), 'confidence': float(conf), 'class_id': int(cls)}
            if cls in self.person_classes and conf >= self.person_conf:
                if self._is_on_pitch(pitch_mask, det['bbox']):
                    persons.append({**det, 'class_name': 'person', 'role': self.model.names[int(cls)]})
            elif (cls in self.ball_classes and conf >= self.ball_conf
                  and self._is_on_pitch(pitch_mask, det['bbox'])
                  and self._is_plausible_ball(frame, det['bbox'], all_person_boxes)):
                balls.append({**det, 'class_name': 'ball'})

        balls.sort(key=lambda b: -b['confidence'])
        return persons + balls[:self.MAX_BALL_CANDIDATES]

    def _is_plausible_ball(self, frame: np.ndarray, bbox: List[int], person_boxes: List[List[int]],
                           min_grass_ratio: float = 0.5, max_saturation: float = 140) -> bool:
        """
        Reject false-positive balls. The COCO 'sports ball' class frequently fires on
        round bright blobs such as stewards' hi-vis vests or the referee's yellow kit,
        so a candidate is kept only if:
        1. it is not on a person's head/torso (a ball at a player's feet is in the
           bottom 20% of that player's box),
        2. its surroundings (3x the box size) are mostly grass, and
        3. its box is not strongly saturated. The ball is only ~5-9 px wide and blends with
           the grass (median HSV saturation ~85-115 on test_match.mp4), whereas hi-vis
           yellow kits measure ~175-185.
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        for px1, py1, px2, py2 in person_boxes:
            if px1 <= cx <= px2 and py1 <= cy <= py1 + 0.8 * (py2 - py1):
                return False

        H, W = frame.shape[:2]
        ball = frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        pad = max(x2 - x1, y2 - y1, 8)
        patch = frame[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
        if ball.size == 0 or patch.size == 0:
            return False

        if np.median(cv2.cvtColor(ball, cv2.COLOR_BGR2HSV)[..., 1]) > max_saturation:
            return False

        grass = cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV), self.green_lower, self.green_upper)
        return np.count_nonzero(grass) / grass.size >= min_grass_ratio

    def pitch_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Binary mask (255 = pitch) of the playing surface.

        1. Threshold green in HSV (at 1/4 resolution for speed).
        2. Morphological closing removes the holes left by players, lines and the ball.
        3. Keep only the largest connected grass region and fill its interior holes.
        Unlike a convex hull, this follows the real outline of the grass, so the
        advertising boards, track, bench and stands stay outside the mask.
        """
        small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), self.green_lower, self.green_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        if contours:
            cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        return cv2.resize(filled, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _is_on_pitch(mask: np.ndarray, bbox: List[int], min_ratio: float = 0.5) -> bool:
        """
        A detection is on the pitch if its feet region (bottom 10% of the box, central
        half of its width) lies mostly inside the pitch mask.
        """
        h, w = mask.shape
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        fx1, fx2 = int(np.clip(x1 + 0.25 * bw, 0, w - 1)), int(np.clip(x2 - 0.25 * bw, 0, w - 1)) + 1
        fy1, fy2 = int(np.clip(y2 - 0.1 * bh, 0, h - 1)), int(np.clip(y2, 0, h - 1)) + 1
        feet = mask[fy1:fy2, fx1:fx2]
        return feet.size > 0 and np.count_nonzero(feet) / feet.size >= min_ratio
