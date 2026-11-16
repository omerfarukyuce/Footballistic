"""
Pitch detection and boundary filtering module.
Segments the playing field from the camera feed and filters out false detections
in the stands, technical area, and advertising boards.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


class PitchDetector:
    """Detects the green football pitch area and filters out-of-bounds detections."""

    def __init__(self):
        # HSV range for soccer green pitch
        self.lower_green = np.array([30, 35, 35])
        self.upper_green = np.array([85, 255, 255])
        self.last_pitch_contour: Optional[np.ndarray] = None

    def get_pitch_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Computes binary mask of the pitch area using HSV color filtering
        and morphological operations to close field markings and players.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # Close gaps caused by white lines and players
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find largest contour (the playing pitch)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pitch_mask = np.zeros_like(raw_mask)
        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            # Ensure it is large enough to be the pitch
            if cv2.contourArea(largest_c) > (frame.shape[0] * frame.shape[1] * 0.25):
                hull = cv2.convexHull(largest_c)
                cv2.drawContours(pitch_mask, [hull], -1, 255, -1)
                self.last_pitch_contour = hull
                return pitch_mask

        # Fallback to last known contour if available
        if self.last_pitch_contour is not None:
            cv2.drawContours(pitch_mask, [self.last_pitch_contour], -1, 255, -1)
            return pitch_mask

        # If all else fails, assume central 80% is pitch
        h, w = frame.shape[:2]
        pitch_mask[int(h * 0.1):int(h * 0.95), int(w * 0.05):int(w * 0.95)] = 255
        return pitch_mask

    def is_inside_pitch(self, point: Tuple[float, float], tolerance_px: float = 10.0) -> bool:
        """
        Checks whether a point (x, y) is inside the detected pitch contour.
        """
        if self.last_pitch_contour is None:
            return True

        # pointPolygonTest returns positive if inside, negative if outside, 0 if on edge
        dist = cv2.pointPolygonTest(self.last_pitch_contour, (float(point[0]), float(point[1])), True)
        return dist >= -tolerance_px

    def filter_ball_candidates(
        self,
        balls: List,
        max_size_px: int = 40,
        min_size_px: int = 4
    ) -> List:
        """
        Filters ball bounding boxes to only keep plausible ball detections:
        - Must be inside the pitch contour
        - Must have plausible ball size (neither too huge nor sub-pixel noise)
        - Must have approximately 1:1 aspect ratio
        """
        valid_balls = []
        for ball in balls:
            x1, y1, x2, y2 = ball.bbox
            w = x2 - x1
            h = y2 - y1
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            # 1. Size check
            if w > max_size_px or h > max_size_px or w < min_size_px or h < min_size_px:
                continue

            # 2. Aspect ratio check (balls are round/square boxes)
            aspect_ratio = w / max(1.0, h)
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            # 3. Inside pitch check
            if not self.is_inside_pitch(center):
                continue

            valid_balls.append(ball)

        return valid_balls
