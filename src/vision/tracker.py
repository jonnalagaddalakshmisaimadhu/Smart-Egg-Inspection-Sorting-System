"""
EggVision: Universal Tracking & In-Frame Telemetry Engine (Phase 9)
Supports any video orientation (trays, horizontal/vertical conveyors, static setups).
Instant detection, defect priority resolution, and itemized traceability registry.
"""

import time

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
            "active_dirty": 0,
            "active_good": 0
        }

        # Class mapping: 0: cracked, 1: dirty, 2: good
        self.class_names = {0: "cracked", 1: "dirty", 2: "good"}

    def update_and_count(self, detections, size_grades: list):
        """
        Processes detections in the current frame.
        Instantly registers and tallies unique eggs without waiting for line crossings.
        """
        active_in_frame = 0
        active_cracked = 0
        active_dirty = 0
        active_good = 0

        if detections is None or detections.tracker_id is None or len(detections.tracker_id) == 0:
            self.stats["active_in_frame"] = 0
            self.stats["active_cracked"] = 0
            self.stats["active_dirty"] = 0
            self.stats["active_good"] = 0
            return

        current_time_str = time.strftime("%H:%M:%S")

        for idx, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue

            cls_id = int(detections.class_id[idx])
            conf = float(detections.confidence[idx]) if detections.confidence is not None else 0.85
            grade = size_grades[idx] if idx < len(size_grades) else "M"
            bbox = detections.xyxy[idx]
            bw = float(bbox[2] - bbox[0])
            bh = float(bbox[3] - bbox[1])
            center_y = (bbox[1] + bbox[3]) / 2.0

            # Live in-frame tallies
            active_in_frame += 1
            if cls_id == 0:
                active_cracked += 1
            elif cls_id == 1:
                active_dirty += 1
            else:
                active_good += 1

            # 1. New unique egg detected in frame
            if tracker_id not in self.egg_records:
                self.egg_records[tracker_id] = {
                    "id": f"EG-{int(tracker_id):04d}",
                    "tracker_id": int(tracker_id),
                    "worst_class": cls_id,
                    "grade": grade,
                    "confidence": round(conf, 2),
                    "width_px": round(bw, 1),
                    "height_px": round(bh, 1),
                    "prev_y": center_y,
                    "counted": False,
                    "frames_seen": 1,
                    "first_seen": current_time_str,
                    "last_seen": current_time_str,
                    "status": "PASS" if cls_id == 2 else "REJECT",
                    "disposition": "ACCEPT (PACKING)" if cls_id == 2 else ("EJECT (CRACKED)" if cls_id == 0 else "DIVERT (DIRTY)")
                }
                
                # Universal Mode: Count immediately upon detection
                if not self.enable_line_crossing:
                    self._register_egg_count(tracker_id, cls_id, grade)

            else:
                # 2. Existing tracked egg
                record = self.egg_records[tracker_id]
                record["frames_seen"] += 1
                record["last_seen"] = current_time_str
                record["confidence"] = max(record["confidence"], round(conf, 2))
                record["width_px"] = round(bw, 1)
                record["height_px"] = round(bh, 1)
                prev_y = record["prev_y"]

                # Defect Priority Resolver: cracked (0) > dirty (1) > good (2)
                if cls_id < record["worst_class"]:
                    old_class = record["worst_class"]
                    record["worst_class"] = cls_id
                    record["status"] = "PASS" if cls_id == 2 else "REJECT"
                    record["disposition"] = "ACCEPT (PACKING)" if cls_id == 2 else ("EJECT (CRACKED)" if cls_id == 0 else "DIVERT (DIRTY)")
                    
                    # Update cumulative tallies if already counted
                    if record["counted"]:
                        if old_class == 2 and cls_id == 0:
                            self.stats["good"] = max(0, self.stats["good"] - 1)
                            self.stats["cracked"] += 1
                        elif old_class == 2 and cls_id == 1:
                            self.stats["good"] = max(0, self.stats["good"] - 1)
                            self.stats["dirty"] += 1
                        elif old_class == 1 and cls_id == 0:
                            self.stats["dirty"] = max(0, self.stats["dirty"] - 1)
                            self.stats["cracked"] += 1

                record["grade"] = grade

                # Optional Line Crossing Mode check
                if self.enable_line_crossing and not record["counted"]:
                    if prev_y < self.line_y and center_y >= self.line_y:
                        self._register_egg_count(tracker_id, record["worst_class"], grade)

                record["prev_y"] = center_y

        self.stats["active_in_frame"] = active_in_frame
        self.stats["active_cracked"] = active_cracked
        self.stats["active_dirty"] = active_dirty
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

    def get_itemized_records(self):
        """Returns sorted list of all unique egg inspection records."""
        records = list(self.egg_records.values())
        records.sort(key=lambda r: r["tracker_id"])
        return records
