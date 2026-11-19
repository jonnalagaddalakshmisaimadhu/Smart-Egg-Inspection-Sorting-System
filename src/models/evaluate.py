"""
EggVision: Test Set Evaluation & Error Analysis Script (Phase 7)
Evaluates EggVision.pt on the untouched test split, extracts honest metrics,
identifies failure cases, and categorizes top error types.
"""

import os
import shutil
import cv2
from pathlib import Path
from ultralytics import YOLO

def evaluate_test_set():
    base_dir = Path(__file__).resolve().parent.parent.parent
    model_path = base_dir / "models" / "weights" / "EggVision.pt"
    data_yaml = base_dir / "data" / "datasets" / "v1" / "data.yaml"
    eval_dir = base_dir / "runs" / "detect" / "test_eval"
    failures_dir = eval_dir / "failures"

    print("=" * 65)
    print("🥚 EggVision: Phase 7 Honest Test Split Evaluation")
    print("=" * 65)
    print(f"📦 Model Weight : {model_path}")
    print(f"📊 Dataset Config: {data_yaml}")
    print("=" * 65)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

    # 1. Run Official Test Split Validation
    model = YOLO(str(model_path))
    test_metrics = model.val(
        data=str(data_yaml),
        split="test",
        project=str(base_dir / "runs" / "detect"),
        name="test_eval",
        device=0,
        exist_ok=True,
        verbose=True,
        plots=True
    )

    # 2. Detailed Per-Image Error Analysis
    test_img_dir = base_dir / "data" / "datasets" / "v1" / "images" / "test"
    test_lbl_dir = base_dir / "data" / "datasets" / "v1" / "labels" / "test"

    failures_dir.mkdir(parents=True, exist_ok=True)

    class_names = {0: "cracked", 1: "dirty", 2: "good"}
    
    total_test = 0
    correct_count = 0
    failures = []

    error_types = {
        "cracked_missed_as_good": 0,    # Critical Safety Defect
        "good_falsely_called_cracked": 0, # Production Waste
        "no_detection": 0
    }

    test_images = sorted([f for f in test_img_dir.iterdir() if f.suffix.lower() in [".jpg", ".png"]])
    
    for img_path in test_images:
        total_test += 1
        lbl_path = test_lbl_dir / (img_path.stem + ".txt")
        
        # Ground truth class
        gt_class = None
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                first_line = f.readline().strip()
                if first_line:
                    gt_class = int(first_line.split()[0])

        # Inference
        results = model.predict(source=str(img_path), device=0, conf=0.35, verbose=False)
        pred_boxes = results[0].boxes
        
        pred_class = None
        pred_conf = 0.0
        if len(pred_boxes) > 0:
            best_idx = pred_boxes.conf.argmax()
            pred_class = int(pred_boxes.cls[best_idx].item())
            pred_conf = float(pred_boxes.conf[best_idx].item())

        # Check accuracy
        if pred_class == gt_class:
            correct_count += 1
        else:
            gt_name = class_names.get(gt_class, "unknown")
            pred_name = class_names.get(pred_class, "none")
            
            # Categorize error
            if gt_class == 0 and pred_class == 2:
                err_category = "cracked_missed_as_good"
                error_types["cracked_missed_as_good"] += 1
            elif gt_class == 2 and pred_class == 0:
                err_category = "good_falsely_called_cracked"
                error_types["good_falsely_called_cracked"] += 1
            elif pred_class is None:
                err_category = "no_detection"
                error_types["no_detection"] += 1
            else:
                err_category = f"misc_{gt_name}_to_{pred_name}"
                error_types.setdefault(err_category, 0)
                error_types[err_category] += 1

            failures.append({
                "image": img_path.name,
                "ground_truth": gt_name,
                "predicted": pred_name,
                "confidence": pred_conf,
                "category": err_category
            })

            # Save visual failure image with annotated text
            ann_img = results[0].plot()
            h, w = ann_img.shape[:2]
            status_text = f"GT: {gt_name.upper()} | PRED: {pred_name.upper()} ({pred_conf:.2f})"
            cv2.rectangle(ann_img, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(ann_img, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            fail_save_path = failures_dir / f"fail_{err_category}_{img_path.name}"
            cv2.imwrite(str(fail_save_path), ann_img)

    # 3. Print Final Honest Report
    map50 = test_metrics.box.map50
    map50_95 = test_metrics.box.map
    accuracy = (correct_count / total_test) * 100 if total_test > 0 else 0

    print("\n" + "=" * 65)
    print("🎯 Phase 7: Honest Test Set Results Summary:")
    print("=" * 65)
    print(f"   - Total Test Images    : {total_test}")
    print(f"   - Classification Acc   : {accuracy:.2f}% ({correct_count}/{total_test} correct)")
    print(f"   - Test Set mAP @ 0.50  : {map50:.4f}")
    print(f"   - Test Set mAP @ 50:95 : {map50_95:.4f}")
    print("-" * 65)
    print("🔍 Error Analysis (Top Failure Types):")
    for err, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {err:<30}: {count} occurrences")
    print(f"\n🖼️  Visual failure images saved to: {failures_dir}")
    print("=" * 65)

if __name__ == "__main__":
    evaluate_test_set()
