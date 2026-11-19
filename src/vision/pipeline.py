"""
EggVision: Core Vision Pipeline Engine (Phases 9 & 10 Integration)
Combines YOLO detection, ByteTrack tracking, Size Grading, and Live HUD Rendering.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.vision.calibrator import SizeCalibrator
from src.vision.tracker import EggTracker

class EggVisionPipeline:
    def __init__(
        self,
        model_path="models/weights/EggVision.pt",
        conf_threshold=0.35,
        line_y=320,
        pixels_per_metric=8.5,
        device=0
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        resolved_model = base_dir / model_path

        if not resolved_model.exists():
            raise FileNotFoundError(f"Model weight not found at {resolved_model}")

        # Load YOLO model
        self.model = YOLO(str(resolved_model))
        self.conf_threshold = conf_threshold
        self.line_y = line_y
        self.device = device

        # Initialize Calibrator and Tracker
        self.calibrator = SizeCalibrator(pixels_per_metric=pixels_per_metric)
        self.tracker = EggTracker(line_y=line_y)

        # Supervision Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=4
        )

        # FPS calculation variables
        self.prev_time = time.time()
        self.fps = 0.0

    def process_frame(self, frame: np.ndarray):
        """
        Processes a single video frame through detection, sizing, and tracking.
        Returns:
            annotated_frame (np.ndarray), current_stats (dict)
        """
        h, w = frame.shape[:2]
        
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.prev_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff) if self.fps > 0 else (1.0 / time_diff)
        self.prev_time = current_time

        # 1. YOLO ByteTrack Inference
        results = self.model.track(
            source=frame,
            device=self.device,
            conf=self.conf_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        # Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # 2. Size Grading & Labels
        size_grades = []
        labels = []
        
        if len(detections) > 0 and detections.tracker_id is not None:
            for xyxy, cls_id, conf, tracker_id in zip(
                detections.xyxy, detections.class_id, detections.confidence, detections.tracker_id
            ):
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                measurement = self.calibrator.measure_and_grade(bw, bh)
                grade = measurement["grade"]
                size_grades.append(grade)

                cls_name = self.tracker.class_names.get(int(cls_id), "unknown").upper()
                label_str = f"#{tracker_id} {cls_name} [{grade}] {conf:.2f}"
                labels.append(label_str)
        else:
            size_grades = ["M"] * len(detections)

        # 3. Update Tracker Registry & Check Virtual Line Crossing
        self.tracker.update_and_count(detections, size_grades)

        # 4. Visual Annotations
        annotated_frame = frame.copy()
        
        # Draw virtual counting line across the frame
        cv2.line(annotated_frame, (0, self.line_y), (w, self.line_y), (0, 255, 255), 2)
        cv2.putText(annotated_frame, "INSPECTION & COUNTING LINE", (w - 260, self.line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Draw boxes and labels
        if len(detections) > 0:
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
            if labels:
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )

        # 5. Render Live Heads-Up Display (HUD) Banner
        stats = self.tracker.get_summary()
        stats["fps"] = round(self.fps, 1)

        total = stats["total_passed"]
        cracked = stats["cracked"]
        good = stats["good"]
        s_cnt = stats["sizes"]["S"]
        m_cnt = stats["sizes"]["M"]
        l_cnt = stats["sizes"]["L"]
        good_pct = (good / total * 100) if total > 0 else 100.0

        # Draw top HUD bar
        cv2.rectangle(annotated_frame, (0, 0), (w, 55), (20, 20, 20), -1)
        
        hud_text_1 = f"TOTAL: {total} | GOOD: {good} ({good_pct:.0f}%) | CRACKED: {cracked} | S:{s_cnt} M:{m_cnt} L:{l_cnt}"
        hud_text_2 = f"FPS: {stats['fps']:.1f} | EGGVISION ENGINE"
        
        cv2.putText(annotated_frame, hud_text_1, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(annotated_frame, hud_text_2, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Alert banner if any cracked egg detected
        if cracked > 0:
            cv2.putText(annotated_frame, "DEFECT DETECTED", (w - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return annotated_frame, stats
