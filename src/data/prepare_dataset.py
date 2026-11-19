"""
EggVision: Dataset Preparation & Standardization Script (Phase 5)
Combines raw egg datasets, auto-extracts normalized bounding boxes,
splits into Train (80%), Val (10%), Test (10%), and generates data.yaml.
"""

import os
import random
import shutil
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Phase 0 Defined Classes
CLASSES = {
    0: "cracked",
    1: "dirty",
    2: "good"
}

# Image resolution for training
TARGET_SIZE = (640, 640)

def extract_egg_bbox(img):
    """
    Computes normalized bounding box (x_center, y_center, width, height)
    using adaptive thresholding and contour analysis.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Otsu thresholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if corners are predominantly bright (white background)
    corners = [thresh[0, 0], thresh[0, -1], thresh[-1, 0], thresh[-1, -1]]
    if np.mean(corners) > 127:
        thresh = cv2.bitwise_not(thresh)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > (w * h * 0.05)]
    if valid_contours:
        c = max(valid_contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        # Clamp coordinates between 0.0 and 1.0
        xc = max(0.01, min(0.99, (x + bw / 2.0) / w))
        yc = max(0.01, min(0.99, (y + bh / 2.0) / h))
        nw = max(0.05, min(0.99, bw / w))
        nh = max(0.05, min(0.99, bh / h))
        return xc, yc, nw, nh
    else:
        # Default centered box
        return 0.5, 0.5, 0.85, 0.85

def collect_raw_samples(base_dir):
    """
    Collects image file paths and maps them to Phase 0 classes.
    """
    samples = []
    
    # 1. Dataset: eggs-damaged-or-not
    d1_path = base_dir / "data" / "raw" / "eggs-damaged-or-not" / "Eggs Classification"
    if d1_path.exists():
        # Damaged -> cracked (0)
        damaged_dir = d1_path / "Damaged"
        if damaged_dir.exists():
            for f in damaged_dir.iterdir():
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    samples.append((f, 0, "eggs-damaged-or-not"))
        
        # Not Damaged -> good (2)
        not_damaged_dir = d1_path / "Not Damaged"
        if not_damaged_dir.exists():
            for f in not_damaged_dir.iterdir():
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    samples.append((f, 2, "eggs-damaged-or-not"))

    # 2. Dataset: broken-eggs
    d2_path = base_dir / "data" / "raw" / "broken-eggs"
    if d2_path.exists():
        for split_name in ["train", "test"]:
            split_dir = d2_path / split_name
            if not split_dir.exists():
                continue
            
            # crack -> cracked (0)
            crack_dir = split_dir / "crack"
            if crack_dir.exists():
                for f in crack_dir.iterdir():
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        samples.append((f, 0, "broken-eggs"))
            
            # good -> good (2)
            good_dir = split_dir / "good"
            if good_dir.exists():
                for f in good_dir.iterdir():
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        samples.append((f, 2, "broken-eggs"))

    return samples

def prepare_v1_dataset():
    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "data" / "datasets" / "v1"
    
    print("=" * 60)
    print("🥚 EggVision: Building Standardized v1 Dataset")
    print("=" * 60)
    
    # Collect all raw samples
    samples = collect_raw_samples(base_dir)
    print(f"📊 Total Raw Images Found: {len(samples)}")
    
    if len(samples) == 0:
        print("❌ Error: No raw images found. Please verify data/raw directory.")
        return

    # Shuffle samples
    random.shuffle(samples)
    
    # Stratified split: Train 80%, Val 10%, Test 10%
    by_class = {}
    for img_path, class_id, source in samples:
        by_class.setdefault(class_id, []).append((img_path, class_id, source))
        
    train_set, val_set, test_set = [], [], []
    for class_id, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)
        
        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])
        
    print(f"📁 Split Distribution:")
    print(f"   - Train (80%): {len(train_set)} images")
    print(f"   - Val   (10%): {len(val_set)} images")
    print(f"   - Test  (10%): {len(test_set)} images")
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    # Process and save images and YOLO labels
    splits_data = [("train", train_set), ("val", val_set), ("test", test_set)]
    
    for split_name, dataset in splits_data:
        print(f"\n⚙️  Processing {split_name} split...")
        for idx, (img_path, class_id, source) in enumerate(tqdm(dataset, desc=split_name)):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Compute bbox before resizing
            xc, yc, nw, nh = extract_egg_bbox(img)
            
            # Resize image to target resolution
            resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            dest_name = f"egg_{source}_{idx:04d}_{class_id}.jpg"
            img_dest = output_dir / "images" / split_name / dest_name
            lbl_dest = output_dir / "labels" / split_name / (Path(dest_name).stem + ".txt")
            
            # Save resized image
            cv2.imwrite(str(img_dest), resized_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Save YOLO label file: <class_id> <x_center> <y_center> <width> <height>
            with open(lbl_dest, "w") as lf:
                lf.write(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                
    # Create data.yaml
    data_yaml_content = {
        "path": str(output_dir.as_posix()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "cracked",
            1: "dirty",
            2: "good"
        }
    }
    
    yaml_file = output_dir / "data.yaml"
    with open(yaml_file, "w") as yf:
        yaml.dump(data_yaml_content, yf, default_flow_style=False, sort_keys=False)
        
    print("\n" + "=" * 60)
    print(f"✅ Dataset v1 successfully built at: {output_dir}")
    print(f"📄 Config written to: {yaml_file}")
    print("=" * 60)

if __name__ == "__main__":
    prepare_v1_dataset()
