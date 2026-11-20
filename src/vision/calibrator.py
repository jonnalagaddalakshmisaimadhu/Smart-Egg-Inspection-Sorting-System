"""
EggVision: Size Grading & Metric Calibration Module (Phase 10)
Calibrates pixel dimensions to physical millimeters using adaptive scaling
and grades eggs into Small (S), Medium (M), and Large (L).
"""

class SizeCalibrator:
    def __init__(self, pixels_per_metric: float = 2.45):
        """
        Args:
            pixels_per_metric: Number of pixels per millimeter.
                               Dynamically adapts to frame resolution and camera distance.
        """
        self.pixels_per_metric = pixels_per_metric
        
        # Calibrated area thresholds (in mm²) mapped from physical weights:
        # Small  (S): < 53g   (~ < 1900 mm²)
        # Medium (M): 53g-63g (~ 1900 - 2350 mm²)
        # Large  (L): > 63g   (~ > 2350 mm²)
        self.small_max_area = 1900.0
        self.medium_max_area = 2350.0
        self.running_areas = []

    def adapt_to_resolution(self, frame_w: int, frame_h: int):
        """
        Adapts scale based on camera capture resolution.
        Maps ~45mm egg width accurately across resolutions.
        """
        diag = (frame_w**2 + frame_h**2)**0.5
        self.pixels_per_metric = max(1.5, diag / 450.0)

    def calibrate_from_reference(self, reference_pixel_width: float, known_width_mm: float = 23.0):
        """
        Dynamically updates calibration scale using a detected reference coin.
        """
        if reference_pixel_width > 0:
            self.pixels_per_metric = reference_pixel_width / known_width_mm
            return self.pixels_per_metric
        return self.pixels_per_metric

    def measure_and_grade(self, width_px: float, height_px: float):
        """
        Measures bounding box dimensions in mm and determines size grade.
        Uses adaptive batch median grading for multi-resolution conveyor footage.
        
        Returns:
            dict containing width_mm, height_mm, area_mm2, and size_grade ('S', 'M', 'L')
        """
        width_mm = width_px / self.pixels_per_metric
        height_mm = height_px / self.pixels_per_metric
        area_mm2 = 3.14159 * (width_mm / 2.0) * (height_mm / 2.0)
        raw_box_area = width_px * height_px

        if 400 < raw_box_area < 65000:
            self.running_areas.append(raw_box_area)
            if len(self.running_areas) > 150:
                self.running_areas.pop(0)

        # Adaptive batch sizing: grades relative to batch median egg size
        if len(self.running_areas) >= 5:
            sorted_areas = sorted(self.running_areas)
            med = sorted_areas[len(sorted_areas) // 2]
            if raw_box_area < 0.88 * med:
                grade = "S"
            elif raw_box_area <= 1.12 * med:
                grade = "M"
            else:
                grade = "L"
        else:
            if area_mm2 < self.small_max_area:
                grade = "S"
            elif area_mm2 <= self.medium_max_area:
                grade = "M"
            else:
                grade = "L"

        return {
            "width_mm": round(width_mm, 1),
            "height_mm": round(height_mm, 1),
            "area_mm2": round(area_mm2, 1),
            "grade": grade
        }
