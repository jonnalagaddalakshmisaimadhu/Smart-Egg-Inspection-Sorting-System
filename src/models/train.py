"""
EggVision: Baseline Model Training Script (Phase 6)
Trains YOLO11 nano on RTX 3050 GPU with native metrics tracking & optional MLflow.
Saves the best model checkpoint as models/weights/EggVision.pt.
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

    # Optional MLflow integration (graceful fallback if blocked by Windows Application Control)
    mlflow_active = False
    try:
        import mlflow
        mlflow.set_experiment("EggVision-Baseline")
        mlflow.start_run(run_name="yolo11n-baseline")
        mlflow.log_params({
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "device": str(device)
        })
        mlflow_active = True
        print("📊 MLflow tracking: ACTIVE")
    except Exception as e:
        print(f"ℹ️  MLflow bypassed ({type(e).__name__}). Using Ultralytics native metrics logger.")

    # Load YOLO model
    model = YOLO(model_name)

    # Train model on GPU
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
    print("\n🔍 Running Validation on Val Split...")
    val_metrics = model.val()
    
    map50 = val_metrics.box.map50
    map50_95 = val_metrics.box.map
    
    if mlflow_active:
        try:
            import mlflow
            mlflow.log_metric("val_mAP50", float(map50))
            mlflow.log_metric("val_mAP50_95", float(map50_95))
        except Exception:
            pass

    # Copy best weights to models/weights/EggVision.pt
    best_pt_src = base_dir / project_dir / experiment_name / "weights" / "best.pt"
    target_weights_dir = base_dir / "models" / "weights"
    target_weights_dir.mkdir(parents=True, exist_ok=True)
    
    best_pt_dest = target_weights_dir / "EggVision.pt"
    if best_pt_src.exists():
        shutil.copy(best_pt_src, best_pt_dest)
        print(f"\n💾 Successfully saved best model to: {best_pt_dest}")
        if mlflow_active:
            try:
                import mlflow
                mlflow.log_artifact(str(best_pt_dest))
                mlflow.end_run()
            except Exception:
                pass

    print("\n" + "=" * 65)
    print("🎯 Phase 6 Baseline Evaluation Results:")
    print(f"   - mAP @ 0.50     : {map50:.4f}")
    print(f"   - mAP @ 0.50:0.95: {map50_95:.4f}")
    print(f"   - Model Saved at : {best_pt_dest}")
    print(f"   - Run Artifacts  : {base_dir / project_dir / experiment_name}")
    print("=" * 65)

if __name__ == "__main__":
    train_baseline()
