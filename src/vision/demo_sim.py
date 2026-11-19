"""
EggVision: Conveyor Simulation & Pipeline Verification Script
Simulates eggs moving across a conveyor belt to verify ByteTrack tracking,
virtual line crossing counts, and size grading.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.vision.pipeline import EggVisionPipeline


def run_conveyor_simulation():
    base_dir = Path(__file__).resolve().parent.parent.parent
    test_dir = base_dir / "data" / "datasets" / "v1" / "images" / "test"
    
    print("=" * 65)
    print("🥚 EggVision: Testing Core Vision Pipeline (Phases 9 & 10)")
    print("=" * 65)

    pipeline = EggVisionPipeline(
        model_path="models/weights/EggVision.pt",
        line_y=320,
        device=0
    )

    test_images = sorted(list(test_dir.glob("*.jpg")))[:6]
    if not test_images:
        print("❌ No test images found to simulate.")
        return

    canvas_h, canvas_w = 640, 640
    print(f"🎬 Simulating conveyor movement for {len(test_images)} test eggs...")

    saved_frame = None

    for egg_idx, img_path in enumerate(test_images):
        egg_raw = cv2.imread(str(img_path))
        if egg_raw is None:
            continue
        
        egg_resized = cv2.resize(egg_raw, (240, 240))
        
        # Simulate egg moving vertically from y=50 down to y=360 across counting line (y=320)
        for y_pos in range(50, 370, 30):
            frame = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8) # Dark conveyor background
            
            # Place egg at current position (bounded within 640x640)
            x_pos = 200
            frame[y_pos:y_pos+240, x_pos:x_pos+240] = egg_resized
            
            # Process frame through pipeline
            annotated_frame, stats = pipeline.process_frame(frame)
            saved_frame = annotated_frame

    # Save output demo frame
    output_path = base_dir / "data" / "demo_conveyor_frame.jpg"
    if saved_frame is not None:
        cv2.imwrite(str(output_path), saved_frame)
        print(f"\n🖼️  Saved verification frame to: {output_path}")

    # Print pipeline statistics
    print("\n" + "=" * 65)
    print("🎯 Pipeline Verification Statistics:")
    print(f"   - Total Eggs Crossed Line : {stats['total_passed']}")
    print(f"   - Good Eggs Counted       : {stats['good']}")
    print(f"   - Cracked Eggs Counted    : {stats['cracked']}")
    print(f"   - Size Breakdown (S/M/L)  : {stats['sizes']}")
    print(f"   - Real-time Pipeline FPS  : {stats['fps']:.1f}")
    print("=" * 65)

if __name__ == "__main__":
    run_conveyor_simulation()
