"""
EggVision: Test Set Evaluation & Error Analysis Script (Phase 7)
Evaluates EggVision.pt on the untouched v2 test split (10%), extracts honest metrics,
identifies failure cases, and categorizes top error types.
"""

import os
import shutil
import cv2
from pathlib import Path
from ultralytics import YOLO

def evaluate_test_set(data_yaml="data/datasets/v2/data.yaml"):
    base_dir = Path(__file__).resolve().parent.parent.parent
    model_path = base_dir / "models" / "weights" / "EggVision.pt"
    data_path = base_dir / data_yaml
    eval_dir = base_dir / "runs" / "detect" / "test_eval_v2"
    failures_dir = eval_dir / "failures"

    print("=" * 65)
    print("🥚 EggVision: Phase 7 Honest Test Split Evaluation (v2)")
    print("=" * 65)
    print(f"📦 Model Weight : {model_path}")
    print(f"📊 Dataset Config: {data_path}")
    print("=" * 65)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # 1. Run Official Test Split Validation
    model = YOLO(str(model_path))
    test_metrics = model.val(
        data=str(data_path),
        split="test",
        project=str(base_dir / "runs" / "detect"),
        name="test_eval_v2",
        device=0,
        exist_ok=True,
        verbose=True,
        plots=True
    )

    map50 = test_metrics.box.map50
    map50_95 = test_metrics.box.map
    precision = test_metrics.box.mp
    recall = test_metrics.box.mr

    print("\n" + "=" * 65)
    print("🎯 Honest Test Split Metrics (Phase 7):")
    print(f"   - Precision (P)    : {precision:.4f} ({precision*100:.1f}%)")
    print(f"   - Recall (R)       : {recall:.4f} ({recall*100:.1f}%)")
    print(f"   - mAP @ 0.50       : {map50:.4f} ({map50*100:.1f}%)")
    print(f"   - mAP @ 0.50:0.95  : {map50_95:.4f} ({map50_95*100:.1f}%)")
    print(f"   - Output Artifacts : {eval_dir}")
    print("=" * 65)

if __name__ == "__main__":
    evaluate_test_set()
