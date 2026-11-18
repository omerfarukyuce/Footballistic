"""
Pitch detection and boundary filtering module.

Segments the playing field from the camera feed using HSV colour filtering
and convex hull extraction, then provides:
  - is_inside_pitch()      -- point-in-polygon membership check
  - is_strictly_inside()   -- tighter check that rejects the marginal band
                              used by technical staff / coaches
  - filter_ball_candidates() -- size/shape/position filter for ball detections
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


class PitchDetector:
    """Detects the green football pitch area and filters out-of-bounds detections."""

    def __init__(self):
        # HSV range for soccer-green pitch grass
        self.lower_green = np.array([30, 35, 35])
        self.upper_green = np.array([85, 255, 255])
        self.last_pitch_contour: Optional[np.ndarray] = None
        # Eroded (inset) version of the pitch contour used for strict filtering
        self._strict_contour: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ mask

    def get_pitch_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Computes binary mask of the pitch area using HSV colour filtering
        and morphological operations to close field markings and player gaps.
        Also updates the eroded strict contour used for staff/coach rejection.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pitch_mask = np.zeros_like(raw_mask)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > frame.shape[0] * frame.shape[1] * 0.20:
                hull = cv2.convexHull(largest)
                cv2.drawContours(pitch_mask, [hull], -1, 255, -1)
                self.last_pitch_contour = hull

                # Build a slightly inset contour for strict boundary checking.
                # We draw the hull onto a temporary mask, erode it, then re-extract.
                tmp = np.zeros_like(raw_mask)
                cv2.drawContours(tmp, [hull], -1, 255, -1)
                erode_px = max(5, int(min(frame.shape[:2]) * 0.015))
                ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
                eroded = cv2.erode(tmp, ek, iterations=2)
                strict_cs, _ = cv2.findContours(
                    eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if strict_cs:
                    self._strict_contour = max(strict_cs, key=cv2.contourArea)
                else:
                    self._strict_contour = hull
                return pitch_mask

        # Fallback: reuse last known contour
        if self.last_pitch_contour is not None:
            cv2.drawContours(pitch_mask, [self.last_pitch_contour], -1, 255, -1)
            return pitch_mask

        # Last resort: central 85% of frame
        h, w = frame.shape[:2]
        pitch_mask[int(h * 0.1):int(h * 0.95), int(w * 0.05):int(w * 0.95)] = 255
        return pitch_mask

    # ------------------------------------------------------------------ membership

    def is_inside_pitch(self, point: Tuple[float, float], tolerance_px: float = 10.0) -> bool:
        """
        Returns True if the point (x, y) is within tolerance_px of the outer
        pitch convex hull.  Suitable for general player detection filtering.
        """
        if self.last_pitch_contour is None:
            return True
        dist = cv2.pointPolygonTest(
            self.last_pitch_contour, (float(point[0]), float(point[1])), True
        )
        return dist >= -tolerance_px

    def is_strictly_inside_pitch(self, point: Tuple[float, float]) -> bool:
        """
        Returns True only if the point is clearly inside the playing area
        (uses an eroded/inset contour that excludes the marginal zone where
        technical staff and coaches stand just outside the touch-line).

        Use this for the initial colour-calibration pass so coaches / staff
        at the touch-line are not included in the team-colour fit.
        """
        contour = self._strict_contour if self._strict_contour is not None else self.last_pitch_contour
        if contour is None:
            return True
        dist = cv2.pointPolygonTest(
            contour, (float(point[0]), float(point[1])), True
        )
        return dist >= 0.0  # Must be strictly inside

    # ------------------------------------------------------------------ ball

    def filter_ball_candidates(
        self,
        balls: List,
        max_size_px: int = 40,
        min_size_px: int = 4
    ) -> List:
        """
        Filters ball bounding boxes to only keep plausible detections:
        - Inside pitch
        - Plausible size (4 px - 40 px)
        - Approximately 1:1 aspect ratio
        """
        valid = []
        for ball in balls:
            x1, y1, x2, y2 = ball.bbox
            w = x2 - x1
            h = y2 - y1
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            if w > max_size_px or h > max_size_px or w < min_size_px or h < min_size_px:
                continue
            if not 0.5 <= w / max(1.0, h) <= 2.0:
                continue
            if not self.is_inside_pitch(center):
                continue
            valid.append(ball)
        return valid
