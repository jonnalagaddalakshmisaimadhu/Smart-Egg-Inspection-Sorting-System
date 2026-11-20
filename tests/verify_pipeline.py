import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
from src.vision.pipeline import EggVisionPipeline

pipe = EggVisionPipeline(
    model_path='models/weights/EggVision.pt',
    conf_threshold=0.42,
    line_y=320,
    enable_line_crossing=False,
    device=0
)
cap = cv2.VideoCapture('data/uploads/current_inspection.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for f_idx in range(total):
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    frame_resized = cv2.resize(frame, (960, int(h * (960 / w))))
    ann, stats = pipe.process_frame(frame_resized)

cap.release()
final_stats = pipe.tracker.get_summary()
itemized = pipe.tracker.get_itemized_records()

print("--- PIPELINE VERIFICATION RESULTS ---")
print("Total Passed Unique:", final_stats["total_passed"])
print("Good:", final_stats["good"])
print("Cracked:", final_stats["cracked"])
print("Dirty:", final_stats["dirty"])
print("Sizes:", final_stats["sizes"])
print("Confirmed Itemized Records:", len(itemized))
for egg in itemized:
    print(f"  Egg: {egg['id']} | Class: {egg['class_name']} | Grade: [{egg['grade']}] | Conf: {egg['confidence']} | Box: {egg['width_px']}x{egg['height_px']} | Frames: {egg['frames_seen']}")
