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

        # Near (bottom) touchline model y = a*x + b, detected from the painted white line.
        # Grass usually continues beyond the touchline (apron / technical area), so the
        # grass mask alone cannot separate coaches standing just outside the line.
        self.near_touchline: Optional[Tuple[float, float]] = None
        self._touchline_age: int = 999
        self._rng = np.random.default_rng(7)

    # ------------------------------------------------------------------ touchline

    def detect_near_touchline(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detects the near (bottom) touchline as a straight, thin, near-horizontal white
        line that has grass directly above it and spans a large part of the frame width.

        Method: per-column scan of thin white runs in the lower part of the image,
        followed by RANSAC line fitting. Lines covering < 40% of sampled columns are
        rejected, which filters out watermarks, ad-board lettering and short penalty-box
        edges. If several long lines exist (e.g. a technical-area line below), the
        topmost strong one is the touchline.
        """
        h, w = frame.shape[:2]
        y0 = int(h * 0.55)
        hsv = cv2.cvtColor(frame[max(0, y0 - 8):, :], cv2.COLOR_BGR2HSV)
        off = y0 - max(0, y0 - 8)
        white = (hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 160)
        grass = (
            (hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 58) &
            (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 35)
        )

        step = 6
        xs_pts: List[float] = []
        ys_pts: List[float] = []
        for x in range(0, w, step):
            col = white[off:, x].astype(np.int8)
            if not col.any():
                continue
            d = np.diff(np.concatenate(([0], col, [0])))
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]
            for s, e in zip(starts, ends):
                run_len = e - s
                if run_len < 1 or run_len > 10:
                    continue
                ys_local = s + off  # index in hsv crop
                above = ys_local - 4
                if above < 0 or not (grass[above, x] or grass[max(0, above - 2), x]):
                    continue
                xs_pts.append(float(x))
                ys_pts.append(float(y0 + s + run_len * 0.5))

        n = len(xs_pts)
        if n < 30:
            return None

        xs = np.asarray(xs_pts)
        ys = np.asarray(ys_pts)
        n_cols = w // step
        max_slope = np.tan(np.radians(12.0))

        found: List[Tuple[float, float, int]] = []
        remaining = np.ones(n, dtype=bool)
        for _ in range(3):  # extract up to 3 distinct long lines
            idx_pool = np.where(remaining)[0]
            if len(idx_pool) < 30:
                break
            best = None
            best_cov = 0
            for _it in range(220):
                i, j = self._rng.choice(idx_pool, 2, replace=False)
                dx = xs[j] - xs[i]
                if abs(dx) < 200:
                    continue
                a = (ys[j] - ys[i]) / dx
                if abs(a) > max_slope:
                    continue
                b = ys[i] - a * xs[i]
                inl = remaining & (np.abs(ys - (a * xs + b)) < 3.5)
                cov = len(np.unique(xs[inl]))
                if cov > best_cov:
                    best_cov = cov
                    best = inl
            if best is None or best_cov < 0.40 * n_cols:
                break
            a_fit, b_fit = np.polyfit(xs[best], ys[best], 1)
            if abs(a_fit) > max_slope:
                break
            found.append((float(a_fit), float(b_fit), best_cov))
            # remove a band around this line before searching the next one
            remaining &= np.abs(ys - (a_fit * xs + b_fit)) > 8.0

        if not found:
            return None

        strongest = max(f[2] for f in found)
        strong = [f for f in found if f[2] >= 0.65 * strongest]
        cx = w * 0.5
        a_best, b_best, _ = min(strong, key=lambda f: f[0] * cx + f[1])  # topmost
        return a_best, b_best

    def update_near_touchline(self, frame: np.ndarray) -> None:
        """Updates the cached touchline model; keeps the last one for a few frames on misses."""
        line = self.detect_near_touchline(frame)
        if line is not None:
            self.near_touchline = line
            self._touchline_age = 0
        else:
            self._touchline_age += 1
            if self._touchline_age > 12:
                self.near_touchline = None

    def is_outside_near_touchline(self, foot: Tuple[float, float], bbox_h: float) -> bool:
        """True if a person's feet are clearly below (outside) the near touchline."""
        if self.near_touchline is None:
            return False
        a, b = self.near_touchline
        line_y = a * float(foot[0]) + b
        margin = max(6.0, 0.08 * float(bbox_h))
        return float(foot[1]) > line_y + margin

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
