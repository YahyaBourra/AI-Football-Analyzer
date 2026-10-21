"""
Fine-tune YOLO11 on the football dataset (datasets/data.yaml).

Classes: ball, goalkeeper, player, referee (Roboflow "football-players-and-ball", CC BY 4.0).

Training configuration for an NVIDIA RTX 4060 Laptop GPU (8 GB VRAM):
- device='cuda', amp=True (mixed precision, roughly halves activation memory)
- imgsz=640, batch=8 (fits in 8 GB for yolo11m without CUDA out-of-memory errors)
- epochs=50

Ultralytics saves everything needed for the thesis report into the run directory
(runs/detect/<name>/):
- results.png / results.csv     training & validation losses, precision, recall, mAP50, mAP50-95 per epoch
- confusion_matrix.png (+ _normalized)
- BoxPR_curve.png, BoxF1_curve.png, BoxP_curve.png, BoxR_curve.png
- labels.jpg (label distribution), train_batch*.jpg, val_batch*_labels/pred.jpg
- weights/best.pt (best mAP50-95 on the validation set) and weights/last.pt

Usage:
    python train.py                      # defaults below
    python train.py --epochs 100 --batch 4 --imgsz 1280
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine-tune YOLO11 on the football dataset')
    parser.add_argument('--model', type=str, default=str(ROOT / 'models' / 'yolo11m.pt'), help='Base weights')
    parser.add_argument('--data', type=str, default=str(ROOT / 'datasets' / 'data.yaml'), help='Dataset YAML')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4, help='Dataloader worker processes')
    parser.add_argument('--name', type=str, default='train', help='Run name (runs/detect/<name>)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available - training on the CPU would take days. "
                         "Check the CUDA build of PyTorch (see README).")
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name} ({gpu.total_memory / 1024 ** 3:.1f} GB) | PyTorch {torch.__version__}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        amp=True,
        workers=args.workers,
        project=str(ROOT / 'runs' / 'detect'),
        name=args.name,
        plots=True,  # results.png, confusion matrix, PR/F1 curves, sample batches
        seed=42,
        deterministic=True,
    )

    # Final evaluation of the best checkpoint on the validation split (also writes plots)
    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=0,
                                  project=str(ROOT / 'runs' / 'detect'), name=f'{args.name}_val', plots=True)
    print(f"\nBest weights: {best}")
    print(f"Validation mAP50 {metrics.box.map50:.3f} | mAP50-95 {metrics.box.map:.3f}")
    for i, name in metrics.names.items():
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"  {name:<11} P {p:.3f}  R {r:.3f}  mAP50 {ap50:.3f}  mAP50-95 {ap:.3f}")


if __name__ == '__main__':  # Required on Windows: dataloader workers are spawned processes
    main()
