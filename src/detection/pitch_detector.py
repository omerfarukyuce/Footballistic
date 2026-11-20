"""
Pitch detection and boundary filtering module.

Segments the playing field from the camera feed using HSV grass colour filtering
and non-convex external contour extraction (so stadium dugouts, benches, and
advertising banners are NOT bridged into the pitch mask), then provides:
  - get_pitch_mask()           -- binary mask + dynamic vertical pitch bounds
  - get_pitch_crop_bounds()    -- tight ROI crop around pitch (drops stadium stands)
  - is_inside_pitch()          -- accurate point-in-pitch membership check
  - is_strictly_inside_pitch() -- inset check excluding touchline / technical area
  - filter_ball_candidates()   -- size / shape / pitch+airspace filter for balls
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


class PitchDetector:
    """Detects the green football pitch area and filters out-of-bounds detections."""

    def __init__(self):
        # HSV range for natural football grass turf (H<=58 excludes neon/emerald green LED ad boards!)
        self.lower_green = np.array([28, 45, 35])
        self.upper_green = np.array([58, 255, 235])

        self.last_pitch_contour: Optional[np.ndarray] = None
        self._strict_contour: Optional[np.ndarray] = None
        self.last_pitch_mask: Optional[np.ndarray] = None
        self.last_strict_mask: Optional[np.ndarray] = None
        self.pitch_y_bounds: Optional[Tuple[int, int]] = None  # (y_top, y_bottom)
        self.pitch_x_bounds: Optional[Tuple[int, int]] = None  # (x_left, x_right)
        self.top_edge_by_x: Optional[np.ndarray] = None        # top grass Y per image column

    # ------------------------------------------------------------------ mask

    def get_pitch_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Computes a clean binary mask of the playing field without bridging
        across stadium dugouts or green LED ad banners (avoids convexHull leaks).
        Also computes the eroded strict mask, per-column top edge, and pitch Y bounds.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # 1. Open first to sever thin links to stadium banners/signs
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_open, iterations=1)

        # 2. Horizontal-biased close seals white lines & player legs without bridging tall ad boards
        k_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close_h, iterations=1)
        k_close_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k_close_s, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pitch_mask = np.zeros_like(raw_mask)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > h * w * 0.16:
                # Fill external contour directly (NO convexHull, so dugout roofs stay excluded)
                cv2.drawContours(pitch_mask, [largest], -1, 255, -1)

                # Light closing to smooth small player notches along touchlines
                k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
                pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_CLOSE, k_smooth, iterations=1)

                clean_cs, _ = cv2.findContours(pitch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if clean_cs:
                    largest = max(clean_cs, key=cv2.contourArea)

                self.last_pitch_contour = largest
                self.last_pitch_mask = pitch_mask

                # Per-column top grass edge (for accurate far-touchline linesman rejection)
                has_grass = np.any(pitch_mask > 0, axis=0)
                top_y_col = np.argmax(pitch_mask > 0, axis=0).astype(np.int32)
                top_y_col[~has_grass] = 0
                self.top_edge_by_x = top_y_col

                # Compute dynamic vertical & horizontal bounds of the visible pitch
                pts = largest.reshape(-1, 2)
                y_top = int(np.percentile(pts[:, 1], 1))
                y_bot = int(np.percentile(pts[:, 1], 99))
                x_left = int(pts[:, 0].min())
                x_right = int(pts[:, 0].max())
                self.pitch_y_bounds = (y_top, y_bot)
                self.pitch_x_bounds = (x_left, x_right)

                # Build eroded strict mask/contour (excludes touchline coach band)
                erode_px = max(7, int(min(h, w) * 0.018))
                ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
                eroded = cv2.erode(pitch_mask, ek, iterations=2)
                # Also trim bottom 3.5% of frame when pitch extends to bottom edge
                # (technical area / coach zone directly beneath the near touchline)
                if y_bot >= h - 15:
                    eroded[int(h * 0.94):, :] = 0
                self.last_strict_mask = eroded

                strict_cs, _ = cv2.findContours(
                    eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if strict_cs:
                    self._strict_contour = max(strict_cs, key=cv2.contourArea)
                else:
                    self._strict_contour = largest
                return pitch_mask

        # Fallback: reuse last known mask
        if self.last_pitch_mask is not None and self.last_pitch_mask.shape == raw_mask.shape:
            return self.last_pitch_mask.copy()

        # Last resort: central playing region
        pitch_mask[int(h * 0.20):int(h * 0.95), int(w * 0.02):int(w * 0.98)] = 255
        self.last_pitch_mask = pitch_mask
        self.pitch_y_bounds = (int(h * 0.20), int(h * 0.95))
        self.pitch_x_bounds = (int(w * 0.02), int(w * 0.98))
        return pitch_mask

    def get_pitch_crop_bounds(
        self,
        frame: np.ndarray,
        top_pad: int = 55,
        bottom_pad: int = 8
    ) -> Tuple[int, int, int, int]:
        """
        Returns tight (x1, y1, x2, y2) bounding box around the playing field so
        YOLO does not waste resolution or compute on stadium crowds.
        Adds `top_pad` above the far touchline so heads of distant players and
        lobbed balls remain inside the detection crop.
        """
        h, w = frame.shape[:2]
        if self.pitch_y_bounds is None:
            self.get_pitch_mask(frame)

        if self.pitch_y_bounds is not None:
            py1, py2 = self.pitch_y_bounds
            y1 = max(0, py1 - top_pad)
            y2 = min(h, py2 + bottom_pad)
        else:
            y1, y2 = int(h * 0.15), h

        # Keep full width (football pitch usually spans full horizontal broadcast width)
        return 0, y1, w, y2

    # ------------------------------------------------------------------ membership

    def get_dist_below_top_edge(self, point: Tuple[float, float]) -> float:
        """Returns vertical pixel distance below the top grass edge at point's X column."""
        if self.top_edge_by_x is None:
            return 999.0
        ix = int(np.clip(round(float(point[0])), 0, len(self.top_edge_by_x) - 1))
        top_y = int(self.top_edge_by_x[ix])
        if top_y <= 0:
            return 999.0
        return float(point[1]) - float(top_y)

    def is_inside_pitch(self, point: Tuple[float, float], tolerance_px: float = 0.0) -> bool:
        """
        Returns True if the point (x, y) is inside the non-convex pitch mask
        (or within tolerance_px of the pitch contour).
        """
        px, py = float(point[0]), float(point[1])

        # Fast O(1) lookup when tolerance_px == 0 and mask exists
        if tolerance_px == 0.0 and self.last_pitch_mask is not None:
            h, w = self.last_pitch_mask.shape[:2]
            ix = int(round(px))
            iy = int(round(py))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                return False
            # Also guard against coaches standing at the extreme bottom frame edge
            if iy >= h - 18:
                return False
            return bool(self.last_pitch_mask[iy, ix] > 0)

        if self.last_pitch_contour is None:
            return True
        dist = cv2.pointPolygonTest(self.last_pitch_contour, (px, py), True)
        return dist >= -tolerance_px

    def is_strictly_inside_pitch(self, point: Tuple[float, float]) -> bool:
        """
        Returns True only if the point is clearly inside the playing area
        (uses the eroded mask that excludes the marginal zone where technical
        staff, coaches, and photographers stand along touchlines).
        """
        px, py = float(point[0]), float(point[1])
        if self.last_strict_mask is not None:
            h, w = self.last_strict_mask.shape[:2]
            ix, iy = int(round(px)), int(round(py))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                return False
            return bool(self.last_strict_mask[iy, ix] > 0)

        contour = self._strict_contour if self._strict_contour is not None else self.last_pitch_contour
        if contour is None:
            return True
        dist = cv2.pointPolygonTest(contour, (px, py), True)
        return dist >= 0.0

    # ------------------------------------------------------------------ ball

    def filter_ball_candidates(
        self,
        balls: List,
        max_size_px: int = 44,
        min_size_px: int = 4,
        top_air_margin_px: int = 95
    ) -> List:
        """
        Filters ball bounding boxes to keep plausible detections:
        - Inside pitch OR in the airspace slightly above the far touchline (lobbed balls)
        - Plausible ball size (4 px - 44 px)
        - Approximately circular aspect ratio (0.45 - 2.2)
        """
        valid = []
        y_top_limit = (self.pitch_y_bounds[0] - top_air_margin_px) if self.pitch_y_bounds else 0

        for ball in balls:
            x1, y1, x2, y2 = ball.bbox
            w = x2 - x1
            h = y2 - y1
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            if w > max_size_px or h > max_size_px or w < min_size_px or h < min_size_px:
                continue
            if not 0.45 <= w / max(1.0, h) <= 2.2:
                continue

            # Accept if inside pitch (small tolerance) OR airborne above far touchline
            in_pitch = self.is_inside_pitch(center, tolerance_px=15.0)
            in_airspace = (
                self.pitch_y_bounds is not None
                and y_top_limit <= center[1] <= self.pitch_y_bounds[0] + 25
            )
            if not (in_pitch or in_airspace):
                continue

            valid.append(ball)
        return valid
