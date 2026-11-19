"""
EggVision: Size Grading & Metric Calibration Module (Phase 10)
Calibrates pixel dimensions to physical millimeters using adaptive scaling
and grades eggs into Small (S), Medium (M), and Large (L).
"""

class SizeCalibrator:
    def __init__(self, pixels_per_metric: float = 2.6):
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

    def adapt_to_resolution(self, frame_w: int, frame_h: int):
        """
        Adapts scale based on camera capture resolution.
        """
        diag = (frame_w**2 + frame_h**2)**0.5
        # Standard industrial 1080p diagonal (~2200px) maps to ~2.6 px/mm
        self.pixels_per_metric = max(1.2, diag / 850.0)

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
