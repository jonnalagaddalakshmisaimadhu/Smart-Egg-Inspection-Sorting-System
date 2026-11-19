"""
EggVision: Size Grading & Metric Calibration Module (Phase 10)
Calibrates pixel dimensions to physical millimeters using a reference standard
and grades eggs into Small (S), Medium (M), and Large (L).
"""

class SizeCalibrator:
    def __init__(self, pixels_per_metric: float = 8.5):
        """
        Args:
            pixels_per_metric: Number of pixels per millimeter.
                               Can be calibrated using a reference coin (e.g., standard ₹5 coin = 23mm).
        """
        self.pixels_per_metric = pixels_per_metric
        
        # Calibrated area thresholds (in mm²) mapped from physical weights:
        # Small  (S): < 53g   (~ < 1950 mm²)
        # Medium (M): 53g-63g (~ 1950 - 2400 mm²)
        # Large  (L): > 63g   (~ > 2400 mm²)
        self.small_max_area = 1950.0
        self.medium_max_area = 2400.0

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
        
        Returns:
            dict containing width_mm, height_mm, area_mm2, and size_grade ('S', 'M', 'L')
        """
        width_mm = width_px / self.pixels_per_metric
        height_mm = height_px / self.pixels_per_metric
        
        # Approximate surface projection area (ellipse formula: pi * a * b)
        area_mm2 = 3.14159 * (width_mm / 2.0) * (height_mm / 2.0)
        
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
