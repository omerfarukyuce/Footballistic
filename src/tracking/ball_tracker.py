"""
Dedicated Broadcast Football Ball Tracker.

Combines four complementary signals for high-recall, glitch-free ball tracking:
  1. High-resolution pitch-cropped YOLO detections (class 32, low conf >= 0.06).
  2. GMC-Compensated Constant-Velocity Kalman Filter [x, y, vx, vy] that warps
     the ball state with the camera pan every frame.
  3. 1:1 Native-Resolution ROI Zoom (`detect_ball_in_roi`) around the predicted
     ball position whenever full-frame YOLO misses the ball.
  4. Fast Local GMC-Compensated Motion + White-Blob Recovery inside a tight
     search window when fast kicks cause motion blur that hides the ball from YOLO.
  5. Global Post-Pass Trajectory Filtering & GMC-Compensated Smooth Interpolation
     across short occlusions (e.g. ball passing behind players' legs) before rendering
     the pure WHITE broadcast ball trail.
"""

import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.tracking.football_tracker import warp_point, project_point_across_frames


class _BallKalman:
    """2D constant-velocity Kalman filter with camera-motion warp support."""

    def __init__(self):
        # State: [x, y, vx, vy]
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 500.0
        self.F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.94, 0.0],  # mild velocity damping (grass friction / air drag)
            [0.0, 0.0, 0.0, 0.94],
        ], dtype=np.float64)
        self.H_meas = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)
        self.Q = np.diag([6.0, 6.0, 35.0, 35.0])
        self.R = np.diag([16.0, 16.0])
        self.initialized = False

    def init(self, cx: float, cy: float, vx: float = 0.0, vy: float = 0.0):
        self.x = np.array([cx, cy, vx, vy], dtype=np.float64)
        self.P = np.diag([25.0, 25.0, 225.0, 225.0])
        self.initialized = True

    def apply_gmc(self, warp: Optional[np.ndarray]):
        """Warps Kalman position and velocity by the 2x3 camera motion matrix."""
        if not self.initialized or warp is None:
            return
        R = warp[:2, :2].astype(np.float64)
        t = warp[:2, 2].astype(np.float64)
        self.x[:2] = R @ self.x[:2] + t
        self.x[2:4] = R @ self.x[2:4]

    def predict(self) -> Tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, cx: float, cy: float, meas_noise_scale: float = 1.0):
        z = np.array([cx, cy], dtype=np.float64)
        R = self.R * meas_noise_scale
        y = z - self.H_meas @ self.x
        S = self.H_meas @ self.P @ self.H_meas.T + R
        K = self.P @ self.H_meas.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H_meas) @ self.P


class BallTracker:
    """Online + Global Post-Pass Ball Tracker for football broadcast footage."""

    def __init__(self):
        self.kf = _BallKalman()
        self.frames_since_seen: int = 999
        self.last_radius: float = 7.0
        self.prev_gray: Optional[np.ndarray] = None

        # Per-frame confirmed/recovered ball observations:
        # {frame_idx: (cx, cy, radius, confidence, source)}
        self.observations: Dict[int, Tuple[float, float, float, float, str]] = {}
        # All filtered raw YOLO candidates per frame for global post-linking:
        # {frame_idx: [(cx, cy, radius, conf), ...]}
        self.all_candidates: Dict[int, List[Tuple[float, float, float, float]]] = {}

    # ------------------------------------------------------------------ online step

    def _detect_motion_blob_ball(
        self,
        frame: np.ndarray,
        curr_gray: np.ndarray,
        pred_x: float,
        pred_y: float,
        warp_matrix: np.ndarray,
        player_bboxes: List[List[float]],
        search_radius: int = 68
    ) -> Optional[Tuple[float, float, float]]:
        """
        Fast local visual fallback: inside a compact window around the Kalman
        prediction, looks for a small moving bright/white circular blob that is
        not inside a player's upper torso.
        """
        if self.prev_gray is None:
            return None

        fh, fw = frame.shape[:2]
        x1 = max(0, int(round(pred_x)) - search_radius)
        y1 = max(0, int(round(pred_y)) - search_radius)
        x2 = min(fw, int(round(pred_x)) + search_radius)
        y2 = min(fh, int(round(pred_y)) + search_radius)
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None

        # Warp ONLY the local window via inverse shift of the affine matrix
        # Build local-to-full -> inverse warp -> sample prev_gray
        H_full = np.vstack([warp_matrix.astype(np.float32), [0.0, 0.0, 1.0]])
        T_roi = np.array([[1.0, 0.0, -x1], [0.0, 1.0, -y1], [0.0, 0.0, 1.0]], dtype=np.float32)
        M_local = (T_roi @ H_full)[:2, :]

        prev_warped_roi = cv2.warpAffine(
            self.prev_gray, M_local, (x2 - x1, y2 - y1),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        curr_roi_gray = curr_gray[y1:y2, x1:x2]
        diff = cv2.absdiff(curr_roi_gray, prev_warped_roi)
        motion_mask = (diff >= 22).astype(np.uint8) * 255

        # Bright/white low-saturation ball pixels (footballs are predominantly bright/white)
        roi_bgr = frame[y1:y2, x1:x2]
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        bright_mask = cv2.inRange(roi_hsv, np.array([0, 0, 165]), np.array([180, 105, 255]))

        combined = cv2.bitwise_and(motion_mask, bright_mask)

        # Mask out upper 72% of player bounding boxes (white jerseys/arms),
        # while keeping the feet zone open so dribbled balls stay detectable!
        for pb in player_bboxes:
            px1, py1, px2, py2 = pb
            torso_y2 = py1 + (py2 - py1) * 0.72
            rx1 = max(0, int(px1) - x1)
            ry1 = max(0, int(py1) - y1)
            rx2 = min(x2 - x1, int(px2) - x1)
            ry2 = min(y2 - y1, int(torso_y2) - y1)
            if rx2 > rx1 and ry2 > ry1:
                combined[ry1:ry2, rx1:rx2] = 0

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_dist = float(search_radius)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 14.0 or area > 360.0:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim <= 0:
                continue
            circ = 4.0 * math.pi * area / (perim * perim)
            if circ < 0.38:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if not (0.5 <= bw / max(1.0, bh) <= 2.0):
                continue

            cx = x1 + bx + bw * 0.5
            cy = y1 + by + bh * 0.5
            d = math.hypot(cx - pred_x, cy - pred_y)
            if d < best_dist:
                best_dist = d
                best = (float(cx), float(cy), float(max(4.5, (bw + bh) * 0.25)))

        return best

    def update(
        self,
        frame: np.ndarray,
        frame_idx: int,
        valid_balls: List,
        warp_matrix: np.ndarray,
        detector=None,
        pitch_detector=None,
        player_bboxes: Optional[List[List[float]]] = None
    ) -> Optional[Tuple[float, float, float]]:
        """
        Updates online ball state for `frame_idx`.
        Returns (cx, cy, radius) if ball is confidently observed on this frame, else None.
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert full-frame YOLO candidates to (cx, cy, radius, conf)
        cand_list: List[Tuple[float, float, float, float]] = []
        for b in valid_balls:
            x1, y1, x2, y2 = b.bbox
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            rad = max(5.0, min(14.0, ((x2 - x1) + (y2 - y1)) * 0.25))
            cand_list.append((float(cx), float(cy), float(rad), float(b.confidence)))

        # Step 1: GMC-warp & predict Kalman state
        pred_pos: Optional[Tuple[float, float]] = None
        if self.kf.initialized:
            self.kf.apply_gmc(warp_matrix)
            pred_pos = self.kf.predict()

        # Step 2: Associate with full-frame YOLO candidates
        chosen: Optional[Tuple[float, float, float, float, str]] = None

        if pred_pos is not None and self.frames_since_seen <= 22:
            gate = min(290.0, 75.0 + 30.0 * self.frames_since_seen)
            best_score = -1e9
            for cx, cy, rad, conf in cand_list:
                d = math.hypot(cx - pred_pos[0], cy - pred_pos[1])
                if d <= gate:
                    score = conf * 1.4 - (d / gate)
                    if score > best_score:
                        best_score = score
                        chosen = (cx, cy, rad, conf, "yolo")

        # Step 3: Native 1:1 ROI Zoom Recovery when full-frame YOLO missed the ball
        if (
            chosen is None
            and pred_pos is not None
            and self.frames_since_seen <= 10
            and detector is not None
        ):
            roi_balls = detector.detect_ball_in_roi(
                frame, pred_pos[0], pred_pos[1], roi_size=320, conf=0.05
            )
            if pitch_detector is not None:
                roi_balls = pitch_detector.filter_ball_candidates(roi_balls)

            gate_roi = min(155.0, 65.0 + 18.0 * self.frames_since_seen)
            best_score = -1e9
            for b in roi_balls:
                x1, y1, x2, y2 = b.bbox
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                rad = max(5.0, min(14.0, ((x2 - x1) + (y2 - y1)) * 0.25))
                cand_list.append((float(cx), float(cy), float(rad), float(b.confidence)))
                d = math.hypot(cx - pred_pos[0], cy - pred_pos[1])
                if d <= gate_roi:
                    score = float(b.confidence) * 1.4 - (d / gate_roi)
                    if score > best_score:
                        best_score = score
                        chosen = (float(cx), float(cy), float(rad), float(b.confidence), "roi_yolo")

        # Step 4: Fast Local GMC Motion + Bright-Blob Recovery (for blurred fast passes)
        if (
            chosen is None
            and pred_pos is not None
            and self.frames_since_seen <= 5
        ):
            blob = self._detect_motion_blob_ball(
                frame, curr_gray, pred_pos[0], pred_pos[1],
                warp_matrix, player_bboxes or []
            )
            if blob is not None:
                chosen = (blob[0], blob[1], blob[2], 0.12, "motion_blob")

        # Step 5: Re-acquire if uninitialized or lost for a long time
        if chosen is None and (not self.kf.initialized or self.frames_since_seen > 22):
            if cand_list:
                # Pick highest-confidence candidate (prefer >= 0.15 to avoid random noise)
                best_c = max(cand_list, key=lambda c: c[3])
                if best_c[3] >= 0.15:
                    chosen = (best_c[0], best_c[1], best_c[2], best_c[3], "yolo_reacq")

        self.all_candidates[frame_idx] = cand_list
        self.prev_gray = curr_gray

        if chosen is not None:
            cx, cy, rad, conf, src = chosen
            if not self.kf.initialized or self.frames_since_seen > 22:
                self.kf.init(cx, cy)
            else:
                noise_scale = 2.5 if src == "motion_blob" else 1.0
                self.kf.update(cx, cy, meas_noise_scale=noise_scale)
            self.frames_since_seen = 0
            self.last_radius = 0.7 * self.last_radius + 0.3 * rad
            self.observations[frame_idx] = (cx, cy, self.last_radius, conf, src)
            return cx, cy, self.last_radius

        self.frames_since_seen += 1
        return None

    # ------------------------------------------------------------------ global post-processing

    def finalize_trajectory(
        self,
        total_frames: int,
        frame_warps: Dict[int, np.ndarray],
        max_interp_gap: int = 24
    ) -> Dict[int, Tuple[float, float, float, bool]]:
        """
        Global post-pass cleanup and smoothing before video rendering:
          1. Recovers any valid ball detections in `all_candidates` that bridge
             gaps between confirmed observations (e.g. low-conf detections along
             the true flight line!).
          2. Prunes isolated 1-frame low-confidence false-positive blips.
          3. Interpolates missed occlusion gaps (up to `max_interp_gap` frames)
             in GMC-compensated pitch space so the ball glides smoothly with
             camera pans instead of drifting off the grass.
          4. Lightly smooths 1-2px box jitter while preserving sharp kicks.

        Returns:
            {frame_idx: (cx, cy, radius, is_interpolated)}
        """
        obs = dict(self.observations)

        # 1. Prune isolated low-conf single-frame outliers that jump > 220px from neighbours
        sorted_f = sorted(obs.keys())
        keep: Dict[int, Tuple[float, float, float, float, str]] = {}

        for idx, f in enumerate(sorted_f):
            cx, cy, rad, conf, src = obs[f]
            if conf >= 0.40:
                keep[f] = obs[f]
                continue

            supported = False
            # Check previous & next few observations
            for n_idx in range(max(0, idx - 3), min(len(sorted_f), idx + 4)):
                if n_idx == idx:
                    continue
                nf = sorted_f[n_idx]
                gap = abs(nf - f)
                if gap > 25:
                    continue
                ncx, ncy, _, _, _ = obs[nf]
                if nf > f:
                    proj = project_point_across_frames((cx, cy), f, nf, frame_warps)
                    dist = math.hypot(ncx - proj[0], ncy - proj[1])
                else:
                    proj = project_point_across_frames((ncx, ncy), nf, f, frame_warps)
                    dist = math.hypot(cx - proj[0], cy - proj[1])
                if dist <= 65.0 + 36.0 * gap:
                    supported = True
                    break

            if supported or conf >= 0.28:
                keep[f] = obs[f]

        # 2. Try to rescue missed low-conf YOLO candidates that lie RIGHT ON the
        #    GMC-interpolated flight path between consecutive confirmed observations!
        fkeys = sorted(keep.keys())
        rescued: Dict[int, Tuple[float, float, float, float, str]] = dict(keep)
        for k in range(len(fkeys) - 1):
            fa, fb = fkeys[k], fkeys[k + 1]
            gap = fb - fa
            if gap <= 1 or gap > 45:
                continue
            pa = (keep[fa][0], keep[fa][1])
            pb = (keep[fb][0], keep[fb][1])
            pa_at_fb = project_point_across_frames(pa, fa, fb, frame_warps)
            rel_dx = pb[0] - pa_at_fb[0]
            rel_dy = pb[1] - pa_at_fb[1]
            if math.hypot(rel_dx, rel_dy) > 42.0 * gap + 90.0:
                continue

            cur_w = pa
            for f in range(fa + 1, fb):
                H = frame_warps.get(f)
                if H is not None:
                    cur_w = warp_point(cur_w, H)
                t = (f - fa) / float(gap)
                exp_x = cur_w[0] + t * rel_dx
                exp_y = cur_w[1] + t * rel_dy

                cands = self.all_candidates.get(f, [])
                best_c = None
                best_d = 65.0
                for cx, cy, rad, conf in cands:
                    d = math.hypot(cx - exp_x, cy - exp_y)
                    if d < best_d:
                        best_d = d
                        best_c = (cx, cy, rad, conf, "rescued_yolo")
                if best_c is not None:
                    rescued[f] = best_c

        # 3. GMC-compensated smooth interpolation across short remaining gaps
        final_pos: Dict[int, Tuple[float, float, float, bool]] = {
            f: (v[0], v[1], v[2], False) for f, v in rescued.items()
        }
        fkeys = sorted(rescued.keys())
        for k in range(len(fkeys) - 1):
            fa, fb = fkeys[k], fkeys[k + 1]
            gap = fb - fa
            if 1 < gap <= max_interp_gap:
                pa = (rescued[fa][0], rescued[fa][1])
                pb = (rescued[fb][0], rescued[fb][1])
                ra, rb = rescued[fa][2], rescued[fb][2]

                pa_at_fb = project_point_across_frames(pa, fa, fb, frame_warps)
                rel_dx = pb[0] - pa_at_fb[0]
                rel_dy = pb[1] - pa_at_fb[1]

                # Guard against interpolating across unrelated teleport segments
                if math.hypot(rel_dx, rel_dy) > 42.0 * gap + 90.0:
                    continue

                cur_w = pa
                for f in range(fa + 1, fb):
                    H = frame_warps.get(f)
                    if H is not None:
                        cur_w = warp_point(cur_w, H)
                    t = (f - fa) / float(gap)
                    ix = cur_w[0] + t * rel_dx
                    iy = cur_w[1] + t * rel_dy
                    ir = ra + t * (rb - ra)
                    final_pos[f] = (float(ix), float(iy), float(ir), True)

        # 4. Gentle 3-tap jitter smoothing on consecutive frames (skipping sharp kicks)
        smoothed = dict(final_pos)
        for f in sorted(final_pos.keys()):
            if (f - 1) in final_pos and (f + 1) in final_pos:
                p_prev = final_pos[f - 1]
                p_curr = final_pos[f]
                p_next = final_pos[f + 1]

                # Warp prev -> curr and compare velocity change
                H_c = frame_warps.get(f)
                prev_w = warp_point((p_prev[0], p_prev[1]), H_c) if H_c is not None else (p_prev[0], p_prev[1])
                H_n = frame_warps.get(f + 1)
                curr_w = warp_point((p_curr[0], p_curr[1]), H_n) if H_n is not None else (p_curr[0], p_curr[1])

                v1x, v1y = p_curr[0] - prev_w[0], p_curr[1] - prev_w[1]
                v2x, v2y = p_next[0] - curr_w[0], p_next[1] - curr_w[1]
                accel = math.hypot(v2x - v1x, v2y - v1y)

                # Only smooth small jitter (do NOT round off hard kicks/bounces!)
                if accel < 18.0 and math.hypot(v1x, v1y) < 85.0 and math.hypot(v2x, v2y) < 85.0:
                    # Simple 0.2 / 0.6 / 0.2 spatial smoothing around current frame
                    sx = 0.2 * prev_w[0] + 0.6 * p_curr[0] + 0.2 * (p_curr[0] + v2x)
                    sy = 0.2 * prev_w[1] + 0.6 * p_curr[1] + 0.2 * (p_curr[1] + v2y)
                    smoothed[f] = (float(sx), float(sy), p_curr[2], p_curr[3])

        return smoothed
