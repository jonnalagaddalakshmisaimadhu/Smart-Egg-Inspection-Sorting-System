"""
EggVision: High-Precision Vision Pipeline Engine (Phases 9 & 10 Integration)
Multi-egg object detection, ByteTrack tracking, Size Grading, and Live HUD Rendering.
Instantly displays tight bounding boxes, sizing, and metrics on all visible eggs.
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
        conf_threshold=0.42,
        line_y=320,
        enable_line_crossing=False,
        pixels_per_metric=2.6,
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
        self.tracker = EggTracker(line_y=line_y, enable_line_crossing=enable_line_crossing)

        # Supervision Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.52,
            text_thickness=1,
            text_padding=4
        )

        # FPS calculation variables
        self.prev_time = time.time()
        self.fps = 0.0

    def process_frame(self, frame: np.ndarray):
        """
        Processes any video frame with multi-egg localization and defect sorting.
        Returns:
            annotated_frame (np.ndarray), current_stats (dict)
        """
        h, w = frame.shape[:2]
        self.calibrator.adapt_to_resolution(w, h)
        
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

        # Filter out oversized, low-confidence, or degenerate bounding boxes (bowls, hands, shadows, tables)
        if len(detections) > 0 and detections.xyxy is not None:
            valid_indices = []
            frame_area = w * h
            for i in range(len(detections.xyxy)):
                xyxy = detections.xyxy[i]
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                box_area = bw * bh
                aspect = max(bw, bh) / max(min(bw, bh), 1.0)

                # Strict Egg Morphology Verification:
                # 1. Meets confidence threshold
                # 2. Maximum dimension < 35% of frame (prevents container/table bounding)
                # 3. Maximum area < 10% of frame area
                # 4. Aspect ratio <= 1.70 (eggs are ellipsoid, hands/arms/edges have high aspect)
                # 5. Minimum size >= 24px
                if (conf >= self.conf_threshold and
                    box_area < (0.10 * frame_area) and
                    (bw < 0.35 * w) and (bh < 0.35 * h) and
                    bw >= 24 and bh >= 24 and
                    aspect <= 1.70):
                    valid_indices.append(i)

            if len(valid_indices) < len(detections):
                detections = detections[valid_indices]

        # 2. Size Grading for all valid eggs
        size_grades = []
        if len(detections) > 0:
            for xyxy in detections.xyxy:
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                measurement = self.calibrator.measure_and_grade(bw, bh)
                size_grades.append(measurement["grade"])
        else:
            size_grades = []

        # 3. Update Tracker Registry & Obtain Confirmed Consensus Labels
        verified_labels = self.tracker.update_and_count(detections, size_grades)

        # 4. Visual Annotations on every confirmed egg
        annotated_frame = frame.copy()

        # Draw inspection datum line as reference
        datum_y = min(self.line_y, h - 30)
        cv2.line(annotated_frame, (0, datum_y), (w, datum_y), (50, 100, 180), 1)
        cv2.putText(annotated_frame, "INSPECTION DATUM LINE", (12, datum_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 140, 220), 1)

        # Draw boxes and verified labels on eggs
        if len(detections) > 0:
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
            if verified_labels and len(verified_labels) == len(detections):
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=verified_labels
                )

        # 5. Render Live Heads-Up Display (HUD) Banner
        stats = self.tracker.get_summary()
        stats["fps"] = round(self.fps, 1)

        total_unique = stats["total_passed"]
        cracked_unique = stats["cracked"]
        good_unique = stats["good"]
        
        active_in_frame = stats["active_in_frame"]
        active_cracked = stats["active_cracked"]
        
        s_cnt = stats["sizes"]["S"]
        m_cnt = stats["sizes"]["M"]
        l_cnt = stats["sizes"]["L"]

        # Draw top HUD bar
        cv2.rectangle(annotated_frame, (0, 0), (w, 55), (15, 18, 24), -1)
        
        hud_text_1 = f"IN FRAME: {active_in_frame} | TOTAL UNIQUE: {total_unique} | DEFECTS: {cracked_unique} | S:{s_cnt} M:{m_cnt} L:{l_cnt}"
        hud_text_2 = f"FPS: {stats['fps']:.1f} | EGGVISION NEURAL ENGINE"
        
        cv2.putText(annotated_frame, hud_text_1, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(annotated_frame, hud_text_2, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

        # Immediate visual alert if ANY cracked egg is in frame right now
        if active_cracked > 0:
            cv2.putText(annotated_frame, f"CRACKED DEFECT VISIBLE ({active_cracked})", 
                        (w - 290, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)

        return annotated_frame, stats
