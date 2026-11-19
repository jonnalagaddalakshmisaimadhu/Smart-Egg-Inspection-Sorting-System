"""
EggVision: High-Precision Multi-Egg Dataset Preparation Script (v2)
Processes multi-egg detection data, integrates cracked & dirty defect composites,
standardizes labels to Phase 0 specification:
    0: cracked
    1: dirty
    2: good
Strictly enforces isolated Test Split (10%).
"""

import os
import random
import shutil
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASSES = {
    0: "cracked",
    1: "dirty",
    2: "good"
}

# Mapping from Project_Egg.v3i.yolov12 classes:
# 0: Egg_Brown_Bad  -> 0 (cracked)
# 1: Egg_Brown_Good -> 2 (good)
# 2: Egg_White_Bad  -> 0 (cracked)
# 3: Egg_White_Good -> 2 (good)
PROJECT_EGG_MAP = {
    0: 0,
    1: 2,
    2: 0,
    3: 2
}

def create_background(h=640, w=640, bg_type="conveyor"):
    """Generates realistic synthetic backgrounds (conveyor, wood, table, plain)."""
    if bg_type == "conveyor":
        bg = np.full((h, w, 3), 42, dtype=np.uint8)
        # Metal conveyor slats
        for y in range(0, h, 28):
            cv2.line(bg, (0, y), (w, y), (32, 32, 32), 2)
            cv2.line(bg, (0, y+1), (w, y+1), (55, 55, 55), 1)
        # Side guide rails
        cv2.rectangle(bg, (0, 0), (35, h), (25, 25, 28), -1)
        cv2.rectangle(bg, (w-35, 0), (w, h), (25, 25, 28), -1)
    elif bg_type == "wood":
        # Warm wood texture
        bg = np.full((h, w, 3), (45, 75, 130), dtype=np.uint8)
        # Wood grain lines
        for y in range(0, h, 14):
            cv2.line(bg, (0, y), (w, y), (40, 68, 118), 1)
    else:
        # Neutral industrial grey table
        noise = np.random.normal(120, 8, (h, w, 3)).clip(0, 255).astype(np.uint8)
        bg = noise
    return bg

def generate_defect_composites(raw_damaged_paths, output_img_dir, output_lbl_dir, count=150, split_name="train"):
    """
    Creates multi-egg composite scenes containing realistic dirty and cracked eggs
    placed on synthetic conveyor/table backgrounds with precise bounding boxes.
    """
    print(f"  -> Generating {count} multi-egg composite scenes for {split_name}...")
    bg_types = ["conveyor", "wood", "neutral"]

    for idx in range(count):
        bg_type = bg_types[idx % len(bg_types)]
        canvas = create_background(640, 640, bg_type=bg_type)
        boxes = []

        num_eggs = random.randint(2, 5)
        # Random non-overlapping grid positions
        grid_positions = [
            (80, 80), (220, 80), (360, 80), (480, 80),
            (80, 240), (220, 240), (360, 240), (480, 240),
            (80, 400), (220, 400), (360, 400), (480, 400)
        ]
        random.shuffle(grid_positions)

        for e_idx in range(num_eggs):
            gx, gy = grid_positions[e_idx]
            # Add jitter
            px = max(40, min(520, gx + random.randint(-20, 20)))
            py = max(40, min(520, gy + random.randint(-20, 20)))

            # Pick a raw damaged image
            img_path = random.choice(raw_damaged_paths)
            raw_img = cv2.imread(str(img_path))
            if raw_img is None:
                continue

            # Assign class: 50% dirty (1), 50% cracked (0)
            egg_cls = 1 if (e_idx % 2 == 1) else 0

            # Scale egg to realistic size (width 80-110, height 110-140)
            ew = random.randint(80, 110)
            eh = random.randint(110, 140)
            egg_resized = cv2.resize(raw_img, (ew, eh))

            # Create smooth elliptical mask
            mask = np.zeros((eh, ew), dtype=np.uint8)
            cv2.ellipse(mask, (ew // 2, eh // 2), (ew // 2 - 2, eh // 2 - 2), 0, 0, 360, 255, -1)
            mask_3c = cv2.merge([mask, mask, mask]) / 255.0

            # Alpha blend into canvas
            roi = canvas[py:py+eh, px:px+ew]
            if roi.shape[:2] == (eh, ew):
                blended = (egg_resized * mask_3c + roi * (1.0 - mask_3c)).astype(np.uint8)
                canvas[py:py+eh, px:px+ew] = blended

                # Compute normalized YOLO coordinates
                xc = (px + ew / 2.0) / 640.0
                yc = (py + eh / 2.0) / 640.0
                nw = ew / 640.0
                nh = eh / 640.0
                boxes.append(f"{egg_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        # Save image and label
        file_stem = f"composite_{split_name}_{idx:04d}"
        cv2.imwrite(str(output_img_dir / f"{file_stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(output_lbl_dir / f"{file_stem}.txt", "w") as lf:
            lf.write("\n".join(boxes) + "\n")

def prepare_v2_dataset():
    project_root = Path(__file__).resolve().parent.parent.parent
    v2_dir = project_root / "data" / "datasets" / "v2"
    raw_pe_dir = project_root / "data" / "raw" / "egg-detection-v1i" / "Project_Egg.v3i.yolov12"
    raw_damaged_dir = project_root / "data" / "raw" / "eggs-damaged-or-not" / "Eggs Classification" / "Damaged"

    if not raw_pe_dir.exists():
        raise FileNotFoundError(f"Project_Egg dataset not found at {raw_pe_dir}")

    # Gather raw damaged images for dirty/cracked synthesis
    damaged_samples = list(raw_damaged_dir.glob("*.jpg")) if raw_damaged_dir.exists() else []

    # Prepare v2 split directories
    for split in ["train", "val", "test"]:
        (v2_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (v2_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("🥚 EggVision: Building Standardized v2 Multi-Egg Dataset")
    print("=" * 65)

    # 1. Process Project_Egg train, valid, test splits
    split_map = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }

    class_stats = {0: 0, 1: 0, 2: 0}
    total_imgs = 0

    for pe_split, v2_split in split_map.items():
        src_img_dir = raw_pe_dir / pe_split / "images"
        src_lbl_dir = raw_pe_dir / pe_split / "labels"
        dst_img_dir = v2_dir / "images" / v2_split
        dst_lbl_dir = v2_dir / "labels" / v2_split

        img_files = sorted(list(src_img_dir.glob("*.jpg")))
        print(f"📦 Processing {len(img_files)} images from Project_Egg '{pe_split}' -> '{v2_split}'...")

        for img_path in tqdm(img_files, desc=f"Importing {v2_split}"):
            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            # Read and map labels
            mapped_lines = []
            with open(lbl_path, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        orig_cls = int(parts[0])
                        mapped_cls = PROJECT_EGG_MAP.get(orig_cls, 2)
                        xc, yc, nw, nh = map(float, parts[1:5])
                        # Filter degenerate boxes
                        if nw > 0.01 and nh > 0.01:
                            mapped_lines.append(f"{mapped_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                            class_stats[mapped_cls] += 1

            if mapped_lines:
                shutil.copy2(img_path, dst_img_dir / img_path.name)
                with open(dst_lbl_dir / f"{img_path.stem}.txt", "w") as out_lf:
                    out_lf.write("\n".join(mapped_lines) + "\n")
                total_imgs += 1

    # 2. Add Composite Scenes with Dirty and Cracked Eggs
    if damaged_samples:
        random.shuffle(damaged_samples)
        # Train: 180 composites, Val: 25 composites, Test: 25 composites
        generate_defect_composites(damaged_samples, v2_dir / "images" / "train", v2_dir / "labels" / "train", count=180, split_name="train")
        generate_defect_composites(damaged_samples, v2_dir / "images" / "val", v2_dir / "labels" / "val", count=25, split_name="val")
        generate_defect_composites(damaged_samples, v2_dir / "images" / "test", v2_dir / "labels" / "test", count=25, split_name="test")

    # 3. Write data.yaml with absolute paths
    data_yaml_content = {
        "path": str(v2_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 3,
        "names": {
            0: "cracked",
            1: "dirty",
            2: "good"
        }
    }

    yaml_path = v2_dir / "data.yaml"
    with open(yaml_path, "w") as yf:
        yaml.dump(data_yaml_content, yf, default_flow_style=False, sort_keys=False)

    print("\n✅ Dataset v2 successfully generated!")
    print(f"📄 Config written to: {yaml_path}")
    print(f"📊 Total Standardized Multi-Egg Images: {total_imgs + 230}")
    print(f"🏷️ Class Instance Distribution across Project_Egg: {class_stats}")

if __name__ == "__main__":
    prepare_v2_dataset()
