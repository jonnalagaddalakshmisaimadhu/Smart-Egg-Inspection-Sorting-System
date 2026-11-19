"""
EggVision: Baseline Model Training Script (Phase 6)
Trains YOLO11 nano on RTX 3050 GPU with native metrics tracking & optional MLflow.
Saves the best model checkpoint as models/weights/EggVision.pt.
Supports automatic resume if training was interrupted.
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def train_baseline(
    data_yaml="data/datasets/v1/data.yaml",
    model_name="yolo11n.pt",
    epochs=25,
    batch_size=16,
    imgsz=640,
    device=0,
    project_dir="runs/detect",
    experiment_name="baseline"
):
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_path = base_dir / data_yaml
    
    print("=" * 65)
    print("🥚 EggVision: Starting Baseline Model Training (Phase 6)")
    print("=" * 65)
    print(f"📊 Dataset Config : {data_path}")
    print(f"🚀 Base Model     : {model_name}")
    print(f"🎮 Compute Device : GPU ({device})")
    print(f"⏱️  Epochs: {epochs} | Batch: {batch_size} | Img Size: {imgsz}")
    print("=" * 65)

    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_path}. Please run prepare_dataset.py first.")

    last_pt = base_dir / project_dir / experiment_name / "weights" / "last.pt"

    # Check if a previous run was interrupted and can be resumed
    if last_pt.exists():
        print(f"🔄 Found checkpoint from previous run! Resuming from: {last_pt}")
        model = YOLO(str(last_pt))
        results = model.train(resume=True)
    else:
        model = YOLO(model_name)
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            project=str(base_dir / project_dir),
            name=experiment_name,
            workers=2,
            exist_ok=True,
            verbose=True,
            plots=True
        )

    # Evaluate validation metrics
    print("\n🔍 Running Final Validation on Val Split...")
    val_metrics = model.val()
    
    map50 = val_metrics.box.map50
    map50_95 = val_metrics.box.map

    # Copy best weights to models/weights/EggVision.pt
    best_pt_src = base_dir / project_dir / experiment_name / "weights" / "best.pt"
    target_weights_dir = base_dir / "models" / "weights"
    target_weights_dir.mkdir(parents=True, exist_ok=True)
    
    best_pt_dest = target_weights_dir / "EggVision.pt"
    if best_pt_src.exists():
        shutil.copy(best_pt_src, best_pt_dest)
        print(f"\n💾 Successfully saved best model to: {best_pt_dest}")

    print("\n" + "=" * 65)
    print("🎯 Phase 6 Baseline Evaluation Results:")
    print(f"   - mAP @ 0.50     : {map50:.4f}")
    print(f"   - mAP @ 0.50:0.95: {map50_95:.4f}")
    print(f"   - Model Saved at : {best_pt_dest}")
    print(f"   - Run Artifacts  : {base_dir / project_dir / experiment_name}")
    print("=" * 65)

if __name__ == "__main__":
    train_baseline()
