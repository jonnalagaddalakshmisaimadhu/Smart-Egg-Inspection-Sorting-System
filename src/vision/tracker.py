"""
EggVision: Tracking Registry & Virtual Line Counter (Phase 9)
Tracks eggs across frames, handles virtual line crossing,
and enforces strict defect priority: Cracked (0) > Dirty (1) > Good (2).
"""

import numpy as np

class EggTracker:
    def __init__(self, line_y: int = 320):
        """
        Args:
            line_y: Y-coordinate of the horizontal virtual inspection line across the conveyor.
        """
        self.line_y = line_y

        # Persistent Egg Registry: tracker_id -> state dict
        self.egg_records = {}

        # Aggregate Statistics
        self.stats = {
            "total_passed": 0,
            "cracked": 0,
            "dirty": 0,
            "good": 0,
            "sizes": {"S": 0, "M": 0, "L": 0}
        }

        # Class names
        self.class_names = {0: "cracked", 1: "dirty", 2: "good"}

    def update_and_count(self, detections, size_grades: list):
        """
        Args:
            detections: Supervision Detections object containing tracker_id and class_id.
            size_grades: List of size strings ('S', 'M', 'L') matching detections.
        """
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            return

        for idx, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue
                
            cls_id = int(detections.class_id[idx])
            grade = size_grades[idx] if idx < len(size_grades) else "M"
            
            # Bounding box center Y
            bbox = detections.xyxy[idx]
            center_y = (bbox[1] + bbox[3]) / 2.0
            
            if tracker_id not in self.egg_records:
                self.egg_records[tracker_id] = {
                    "worst_class": cls_id,
                    "grade": grade,
                    "prev_y": center_y,
                    "counted": False
                }
            else:
                record = self.egg_records[tracker_id]
                prev_y = record["prev_y"]
                
                # Priority Resolver: Keep worst defect seen across all frames
                # 0 (cracked) > 1 (dirty) > 2 (good)
                if cls_id < record["worst_class"]:
                    record["worst_class"] = cls_id
                record["grade"] = grade

                # Check virtual line crossing (moving downwards across line_y)
                if not record["counted"] and prev_y < self.line_y and center_y >= self.line_y:
                    record["counted"] = True
                    worst_cls = record["worst_class"]

                    self.stats["total_passed"] += 1
                    if worst_cls == 0:
                        self.stats["cracked"] += 1
                    elif worst_cls == 1:
                        self.stats["dirty"] += 1
                    else:
                        self.stats["good"] += 1

                    if grade in self.stats["sizes"]:
                        self.stats["sizes"][grade] += 1

                record["prev_y"] = center_y

    def get_summary(self):
        return self.stats
