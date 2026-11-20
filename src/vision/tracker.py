"""
EggVision: High-Precision Universal Tracking & Telemetry Engine (Phase 9)
Supports complex multi-egg scenes, stationary trays, hands/occlusions, and moving conveyors.
Features:
- Multi-frame confirmation (rejects single-frame noise flickers)
- Temporal consensus defect voting (prevents hand shadows from creating false crack alarms)
- Spatial track deduplication (merges tracks when eggs are temporarily occluded by hands)
- Itemized unit traceability registry for ISO/USDA compliance audits
"""

import time
import math

class EggTracker:
    def __init__(self, line_y: int = 320, enable_line_crossing: bool = False, min_confirm_frames: int = 4):
        """
        Args:
            line_y: Optional visual inspection guide line.
            enable_line_crossing: If True, only counts at line.
                                  If False (default), instantly tracks and counts unique eggs.
            min_confirm_frames: Number of consistent frames required before counting an egg.
        """
        self.line_y = line_y
        self.enable_line_crossing = enable_line_crossing
        self.min_confirm_frames = min_confirm_frames

        # Persistent Egg Registry: egg_id -> state dict
        self.egg_records = {}
        # Track ID alias mapping (for occluded eggs reassigned to original egg): new_tid -> master_tid
        self.track_aliases = {}

        # Cumulative Batch Statistics
        self.stats = {
            "total_passed": 0,
            "cracked": 0,
            "dirty": 0,
            "good": 0,
            "sizes": {"S": 0, "M": 0, "L": 0},
            # Real-time In-Frame instant counters
            "active_in_frame": 0,
            "active_cracked": 0,
            "active_dirty": 0,
            "active_good": 0
        }

        # Class mapping: 0: cracked, 1: dirty, 2: good
        self.class_names = {0: "cracked", 1: "dirty", 2: "good"}

    def update_and_count(self, detections, size_grades: list):
        """
        Processes detections in the current frame with multi-frame consensus
        and spatial occlusion recovery.
        Returns:
            verified_labels (list of str): Human-readable verified labels for drawing.
        """
        active_in_frame = 0
        active_cracked = 0
        active_dirty = 0
        active_good = 0
        verified_labels = []

        if detections is None or detections.tracker_id is None or len(detections.tracker_id) == 0:
            self.stats["active_in_frame"] = 0
            self.stats["active_cracked"] = 0
            self.stats["active_dirty"] = 0
            self.stats["active_good"] = 0
            return verified_labels

        current_time_str = time.strftime("%H:%M:%S")

        for idx, raw_tracker_id in enumerate(detections.tracker_id):
            if raw_tracker_id is None:
                continue

            raw_tid = int(raw_tracker_id)
            cls_id = int(detections.class_id[idx])
            conf = float(detections.confidence[idx]) if detections.confidence is not None else 0.85
            grade = size_grades[idx] if idx < len(size_grades) else "M"
            bbox = detections.xyxy[idx]
            bw = float(bbox[2] - bbox[0])
            bh = float(bbox[3] - bbox[1])
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0

            # 1. Resolve Tracker Alias (Check if this track was previously linked to an existing egg)
            master_tid = self.track_aliases.get(raw_tid, raw_tid)

            # 2. Adaptive Spatial Deduplication (re-associate occluded / nudged eggs)
            if master_tid not in self.egg_records:
                matched_existing = None
                min_dist = float("inf")
                max_allow_dist = max(85.0, bw * 0.70)
                for exist_tid, rec in self.egg_records.items():
                    dx = center_x - rec.get("center_x", center_x)
                    dy = center_y - rec.get("center_y", center_y)
                    dist = math.hypot(dx, dy)
                    if dist < max_allow_dist and dist < min_dist:
                        min_dist = dist
                        matched_existing = exist_tid

                if matched_existing is not None:
                    master_tid = matched_existing
                    self.track_aliases[raw_tid] = master_tid

            # 3. Create or Update Master Record
            initial_worst = 2 if (cls_id == 2 or conf < 0.75) else cls_id
            init_class_name = "SOUND (GRADE A)" if initial_worst == 2 else ("CRACKED SHELL" if initial_worst == 0 else "DIRTY SHELL")

            if master_tid not in self.egg_records:
                self.egg_records[master_tid] = {
                    "id": f"EG-{master_tid:04d}",
                    "tracker_id": master_tid,
                    "class_history": [cls_id],
                    "conf_history": [conf],
                    "worst_class": initial_worst,
                    "class_name": init_class_name,
                    "grade": grade,
                    "confidence": round(conf, 2),
                    "width_px": round(bw, 1),
                    "height_px": round(bh, 1),
                    "center_x": center_x,
                    "center_y": center_y,
                    "prev_y": center_y,
                    "counted": False,
                    "frames_seen": 1,
                    "first_seen": current_time_str,
                    "last_seen": current_time_str,
                    "status": "PASS",
                    "disposition": "ACCEPT (PACKING)"
                }
            else:
                record = self.egg_records[master_tid]
                record["frames_seen"] += 1
                record["last_seen"] = current_time_str
                record["confidence"] = max(record["confidence"], round(conf, 2))
                record["width_px"] = round(bw, 1)
                record["height_px"] = round(bh, 1)
                record["center_x"] = center_x
                record["center_y"] = center_y
                record["grade"] = grade

                # Record class history for temporal consensus
                record["class_history"].append(cls_id)
                record["conf_history"].append(conf)
                if len(record["class_history"]) > 60:
                    record["class_history"].pop(0)
                    record["conf_history"].pop(0)

                # Defect Temporal Consensus Evaluation:
                # Requires sustained multi-frame defect evidence to prevent transient shadows/hands from causing false alarms
                classes = record["class_history"]
                confs = record["conf_history"]
                cracked_hits = classes.count(0)
                dirty_hits = classes.count(1)
                total_hits = len(classes)

                old_class = record["worst_class"]
                new_class = 2  # Default sound egg

                # High-confidence crack across >= 2 frames OR persistent defect hits >= 3 frames with >= 25% ratio
                cracked_high_conf = sum(1 for c, cf in zip(classes, confs) if c == 0 and cf >= 0.70)
                dirty_high_conf = sum(1 for c, cf in zip(classes, confs) if c == 1 and cf >= 0.70)

                if (cracked_high_conf >= 2) or (cracked_hits >= 3 and (cracked_hits / total_hits) >= 0.25):
                    new_class = 0
                elif (dirty_high_conf >= 2) or (dirty_hits >= 3 and (dirty_hits / total_hits) >= 0.25):
                    new_class = 1

                # Update if class evolved
                if new_class != old_class:
                    record["worst_class"] = new_class
                    record["class_name"] = "SOUND (GRADE A)" if new_class == 2 else ("CRACKED SHELL" if new_class == 0 else "DIRTY SHELL")
                    record["status"] = "PASS" if new_class == 2 else "REJECT"
                    record["disposition"] = "ACCEPT (PACKING)" if new_class == 2 else ("EJECT (CRACKED)" if new_class == 0 else "DIVERT (DIRTY)")

                    # If already counted, adjust cumulative tallies smoothly
                    if record["counted"]:
                        if old_class == 2 and new_class == 0:
                            self.stats["good"] = max(0, self.stats["good"] - 1)
                            self.stats["cracked"] += 1
                        elif old_class == 2 and new_class == 1:
                            self.stats["good"] = max(0, self.stats["good"] - 1)
                            self.stats["dirty"] += 1
                        elif old_class == 1 and new_class == 0:
                            self.stats["dirty"] = max(0, self.stats["dirty"] - 1)
                            self.stats["cracked"] += 1

            record = self.egg_records[master_tid]
            
            # 4. Multi-Frame Confirmation Gate (Official registration only after min_confirm_frames)
            if not record["counted"] and record["frames_seen"] >= self.min_confirm_frames:
                if not self.enable_line_crossing:
                    self._register_egg_count(master_tid, record["worst_class"], record["grade"])
                else:
                    # Line crossing mode check
                    prev_y = record.get("prev_y", center_y)
                    if prev_y < self.line_y and center_y >= self.line_y:
                        self._register_egg_count(master_tid, record["worst_class"], record["grade"])

            record["prev_y"] = center_y

            # 5. Live in-frame tallies based on confirmed consensus class
            cur_class = record["worst_class"]
            active_in_frame += 1
            if cur_class == 0 and record["frames_seen"] >= self.min_confirm_frames:
                active_cracked += 1
            elif cur_class == 1 and record["frames_seen"] >= self.min_confirm_frames:
                active_dirty += 1
            else:
                active_good += 1

            # 6. Build verified master display label
            cls_disp = "CRACKED" if cur_class == 0 else ("DIRTY" if cur_class == 1 else "GOOD")
            label_str = f"#{master_tid} {cls_disp} [{record['grade']}] {conf:.2f}"
            verified_labels.append(label_str)

        self.stats["active_in_frame"] = active_in_frame
        self.stats["active_cracked"] = active_cracked
        self.stats["active_dirty"] = active_dirty
        self.stats["active_good"] = active_good
        return verified_labels

    def _register_egg_count(self, master_tid, cls_id, grade):
        """Helper to increment cumulative counts once per confirmed unique egg."""
        record = self.egg_records[master_tid]
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
        # Dynamically synchronize all cumulative counts with confirmed unique records
        confirmed = [r for r in self.egg_records.values() if r.get("counted")]
        total = len(confirmed)
        cracked = sum(1 for r in confirmed if r.get("worst_class") == 0)
        dirty = sum(1 for r in confirmed if r.get("worst_class") == 1)
        good = sum(1 for r in confirmed if r.get("worst_class") == 2)
        s_cnt = sum(1 for r in confirmed if r.get("grade") == "S")
        m_cnt = sum(1 for r in confirmed if r.get("grade") == "M")
        l_cnt = sum(1 for r in confirmed if r.get("grade") == "L")

        self.stats["total_passed"] = total
        self.stats["cracked"] = cracked
        self.stats["dirty"] = dirty
        self.stats["good"] = good
        self.stats["sizes"] = {"S": s_cnt, "M": m_cnt, "L": l_cnt}
        return self.stats

    def get_itemized_records(self):
        """Returns sorted list of all confirmed unique egg inspection records."""
        # Only include eggs that reached minimum confirmation
        records = [r for r in self.egg_records.values() if r.get("counted") or r.get("frames_seen", 0) >= self.min_confirm_frames]
        records.sort(key=lambda r: r["tracker_id"])
        return records
