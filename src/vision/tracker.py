"""
EggVision: Universal Tracking & In-Frame Telemetry Engine (Phase 9)
Supports any video orientation (trays, horizontal/vertical conveyors, static setups).
Instant detection and tracking without requiring rigid line crossings.
"""

class EggTracker:
    def __init__(self, line_y: int = 320, enable_line_crossing: bool = False):
        """
        Args:
            line_y: Optional visual inspection guide line.
            enable_line_crossing: If True, only counts at line.
                                  If False (default), instantly tracks and counts any unique egg entering the frame.
        """
        self.line_y = line_y
        self.enable_line_crossing = enable_line_crossing

        # Persistent Egg Registry: tracker_id -> state dict
        self.egg_records = {}

        # Cumulative Batch Statistics
        self.stats = {
            "total_passed": 0,
            "cracked": 0,
            "dirty": 0,
            "good": 0,
            "sizes": {"S": 0, "M": 0, "L": 0},
            # Real-time In-Frame instant counters (currently visible on screen)
            "active_in_frame": 0,
            "active_cracked": 0,
            "active_good": 0
        }

        # Class mapping
        self.class_names = {0: "cracked", 1: "dirty", 2: "good"}

    def update_and_count(self, detections, size_grades: list):
        """
        Processes detections in the current frame.
        Instantly registers and tallies unique eggs without waiting for line crossings.
        """
        # Reset live in-frame instantaneous counters
        active_in_frame = 0
        active_cracked = 0
        active_good = 0

        if detections is None or detections.tracker_id is None or len(detections.tracker_id) == 0:
            self.stats["active_in_frame"] = 0
            self.stats["active_cracked"] = 0
            self.stats["active_good"] = 0
            return

        for idx, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue

            cls_id = int(detections.class_id[idx])
            grade = size_grades[idx] if idx < len(size_grades) else "M"
            
            # Active in-frame tallies
            active_in_frame += 1
            if cls_id == 0:
                active_cracked += 1
            else:
                active_good += 1

            bbox = detections.xyxy[idx]
            center_y = (bbox[1] + bbox[3]) / 2.0

            # 1. New unique egg detected in frame
            if tracker_id not in self.egg_records:
                self.egg_records[tracker_id] = {
                    "worst_class": cls_id,
                    "grade": grade,
                    "prev_y": center_y,
                    "counted": False,
                    "frames_seen": 1
                }
                
                # Universal Mode: Count immediately upon detection
                if not self.enable_line_crossing:
                    self._register_egg_count(tracker_id, cls_id, grade)

            else:
                # 2. Existing tracked egg
                record = self.egg_records[tracker_id]
                record["frames_seen"] += 1
                prev_y = record["prev_y"]

                # Defect Priority Resolver: If defect (0: cracked) seen in ANY frame, lock it!
                if cls_id < record["worst_class"]:
                    old_class = record["worst_class"]
                    record["worst_class"] = cls_id
                    
                    # If previously counted as good, upgrade to cracked!
                    if record["counted"]:
                        if old_class == 2 and cls_id == 0:
                            self.stats["good"] = max(0, self.stats["good"] - 1)
                            self.stats["cracked"] += 1

                record["grade"] = grade

                # Optional Line Crossing Mode check
                if self.enable_line_crossing and not record["counted"]:
                    if prev_y < self.line_y and center_y >= self.line_y:
                        self._register_egg_count(tracker_id, record["worst_class"], grade)

                record["prev_y"] = center_y

        self.stats["active_in_frame"] = active_in_frame
        self.stats["active_cracked"] = active_cracked
        self.stats["active_good"] = active_good

    def _register_egg_count(self, tracker_id, cls_id, grade):
        """Helper to increment cumulative counts once per unique egg."""
        record = self.egg_records[tracker_id]
        if not record["counted"]:
            record["counted"] = True
            self.stats["total_passed"] += 1
            
            if cls_id == 0:
                self.stats["cracked"] += 1
            elif cls_id == 1:
                self.stats["dirty"] += 1
            else:
                self.stats["good"] += 1

            if grade in self.stats["sizes"]:
                self.stats["sizes"][grade] += 1

    def get_summary(self):
        return self.stats
