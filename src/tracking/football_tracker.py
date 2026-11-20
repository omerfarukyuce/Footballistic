"""
Camera-Motion-Compensated Persistent Football Player Tracker & Global Track Consolidator.

Solves the two core multi-object tracking failures in wide-angle broadcast football:
  1. Camera pans causing ID resets:
     Uses BoT-SORT's Global Motion Compensation (GMC via sparse optical flow) to estimate
     the frame-to-frame 2x3 camera affine warp `H` and warp ALL active & lost Kalman
     boxes BEFORE association. Also exports `H` so ball tracking and visual trails share
     the exact same camera motion model.
  2. Distant / small players losing IDs when IoU drops to 0:
     Extends BoT-SORT with a Stage-3 Foot-Center Distance + Lab Jersey Colour Re-ID
     association hook (`_post_first_association`), matching unmatched tracks even when
     two small bounding boxes have zero pixel overlap, while preventing cross-team ID
     swaps during player crossings.
  3. Post-pass Global Consolidation & Gap Interpolation:
     After Pass 1 completes, projects lost track endpoints forward across gaps using the
     recorded frame-to-frame GMC warps, stitches fragmented tracklets with matching Lab
     jersey appearance, and linearly interpolates short occlusion gaps (up to 12 frames).
"""

import math
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils import matching
from ultralytics.trackers.utils.stracks import multi_gmc
from ultralytics.utils import YAML, IterableSimpleNamespace


class _DetBatch:
    """Lightweight numpy-backed detection container compatible with Ultralytics BOTSORT."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)
        if len(self.xyxy) > 0:
            w = self.xyxy[:, 2] - self.xyxy[:, 0]
            h = self.xyxy[:, 3] - self.xyxy[:, 1]
            cx = self.xyxy[:, 0] + w * 0.5
            cy = self.xyxy[:, 1] + h * 0.5
            self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_DetBatch":
        return _DetBatch(self.xyxy[idx], self.conf[idx], self.cls[idx])


def _is_valid_warp(H: np.ndarray) -> bool:
    """Sanity-checks a 2x3 affine warp to guard against sudden broadcast camera cuts."""
    if H is None or H.shape != (2, 3) or not np.all(np.isfinite(H)):
        return False
    scale = float(np.hypot(H[0, 0], H[1, 0]))
    trans = float(np.hypot(H[0, 2], H[1, 2]))
    return (0.85 <= scale <= 1.15) and (trans <= 240.0)


def warp_point(pt: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """Transforms a single 2D point by a 2x3 affine matrix H."""
    x, y = pt
    nx = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    ny = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    return float(nx), float(ny)


def project_point_across_frames(
    pt: Tuple[float, float],
    start_frame: int,
    end_frame: int,
    frame_warps: Dict[int, np.ndarray]
) -> Tuple[float, float]:
    """Projects a camera point from `start_frame` forward to `end_frame` via GMC warps."""
    cur = pt
    for f in range(start_frame + 1, end_frame + 1):
        H = frame_warps.get(f)
        if H is not None:
            cur = warp_point(cur, H)
    return cur


class FootballBoTSORT(BOTSORT):
    """
    BoT-SORT subclass tailored for football player tracking:
      - Pitch-only detections (zero stadium spectators entering the tracker)
      - Camera Motion Compensation (GMC) via sparse optical flow, exposed as `last_warp`
      - Stage-3 Foot-Center Distance + Lab Jersey Appearance matching for tiny/fast players
    """

    def __init__(self, config_path: str = "configs/football_botsort.yaml"):
        if os.path.exists(config_path):
            cfg = YAML.load(config_path)
        else:
            cfg = {
                "tracker_type": "botsort",
                "track_high_thresh": 0.22,
                "track_low_thresh": 0.08,
                "new_track_thresh": 0.25,
                "track_buffer": 120,
                "match_thresh": 0.85,
                "fuse_score": True,
                "gmc_method": "sparseOptFlow",
                "proximity_thresh": 0.5,
                "appearance_thresh": 0.8,
                "with_reid": False,
                "model": "auto",
            }
        args = IterableSimpleNamespace(**cfg)
        super().__init__(args)

        self.last_warp: np.ndarray = np.eye(2, 3, dtype=np.float32)
        # Per-track EMA Lab jersey color for appearance-aware Stage-3 matching
        self.track_lab_ema: Dict[int, np.ndarray] = {}
        self._curr_det_labs: List[Optional[np.ndarray]] = []

    def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high) -> None:
        """Computes GMC affine camera warp, stores it, and warps all tracks before matching."""
        warp = np.eye(2, 3, dtype=np.float32)
        if hasattr(self, "gmc") and self.gmc.method is not None and img is not None:
            try:
                est = self.gmc.apply(img, results_high.xyxy if len(results_high) > 0 else None)
                if _is_valid_warp(est):
                    warp = est.astype(np.float32)
            except Exception:
                warp = np.eye(2, 3, dtype=np.float32)

        self.last_warp = warp
        multi_gmc(strack_pool, warp)
        multi_gmc(unconfirmed, warp)

    def _post_first_association(
        self,
        strack_pool,
        detections,
        u_track: List[int],
        u_detection: List[int],
        activated,
        refind
    ) -> Tuple[List[int], List[int]]:
        """
        Stage-3 Football Re-Association:
        Matches tracks and detections that had IoU == 0 (e.g. small distant players or
        fast sprints / brief occlusions) using GMC-warped foot distance + box scale +
        Lab jersey color similarity. Prevents cross-team ID swaps!
        """
        if len(u_track) == 0 or len(u_detection) == 0:
            return u_track, u_detection

        rem_tracks = [strack_pool[int(i)] for i in u_track]
        rem_dets = [detections[int(j)] for j in u_detection]

        cost = np.ones((len(rem_tracks), len(rem_dets)), dtype=np.float32)

        for ti, trk in enumerate(rem_tracks):
            t_xyxy = trk.xyxy
            t_fx = (t_xyxy[0] + t_xyxy[2]) * 0.5
            t_fy = t_xyxy[3]
            t_h = max(12.0, float(t_xyxy[3] - t_xyxy[1]))
            dt = max(1, int(self.frame_id - trk.frame_id))

            # Allowed physical travel radius grows smoothly with frames lost
            max_rad = min(230.0, max(45.0, t_h * 1.35) + 11.0 * min(dt, 16))
            t_lab = self.track_lab_ema.get(trk.track_id)

            for di, det in enumerate(rem_dets):
                d_xyxy = det.xyxy
                d_fx = (d_xyxy[0] + d_xyxy[2]) * 0.5
                d_fy = d_xyxy[3]
                d_h = max(12.0, float(d_xyxy[3] - d_xyxy[1]))

                # Scale consistency gate (distant player shouldn't match foreground giant box)
                h_ratio = max(t_h, d_h) / max(1.0, min(t_h, d_h))
                if h_ratio > 1.70:
                    continue

                dist_px = math.hypot(d_fx - t_fx, d_fy - t_fy)
                if dist_px > max_rad:
                    continue

                # Appearance (Lab jersey color) gate to forbid cross-team swaps
                d_lab = (
                    self._curr_det_labs[det.idx]
                    if (0 <= det.idx < len(self._curr_det_labs))
                    else None
                )
                color_cost = 0.25
                if t_lab is not None and d_lab is not None:
                    lab_dist = float(np.linalg.norm(t_lab - d_lab))
                    # Strict cross-team guard: reject clearly different jersey colours
                    if lab_dist > 34.0:
                        continue
                    color_cost = min(1.0, lab_dist / 38.0)

                spatial_cost = dist_px / max_rad
                cost[ti, di] = 0.65 * spatial_cost + 0.35 * color_cost

        matches, u_t_local, u_d_local = matching.linear_assignment(cost, thresh=0.78)
        for ti, di in matches:
            self._apply_match(rem_tracks[ti], rem_dets[di], activated, refind)

        new_u_track = np.array([int(u_track[int(i)]) for i in u_t_local], dtype=int)
        new_u_det = np.array([int(u_detection[int(j)]) for j in u_d_local], dtype=int)
        return new_u_track, new_u_det

    def update_players(
        self,
        frame: np.ndarray,
        bboxes: List[List[float]],
        confs: List[float],
        det_labs: Optional[List[Optional[np.ndarray]]] = None
    ) -> Tuple[List[Tuple[int, List[float], float, int]], np.ndarray]:
        """
        Updates the tracker with pitch-filtered player detections for one frame.

        Returns:
            tracked_list: List of (track_id, [x1, y1, x2, y2], score, orig_det_idx)
            warp_matrix:  2x3 GMC affine matrix mapping previous frame -> current frame
        """
        self._curr_det_labs = det_labs if det_labs is not None else [None] * len(bboxes)

        if len(bboxes) == 0:
            batch = _DetBatch(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)))
        else:
            batch = _DetBatch(
                np.asarray(bboxes, dtype=np.float32),
                np.asarray(confs, dtype=np.float32),
                np.zeros((len(bboxes),), dtype=np.float32),
            )

        out = super().update(batch, img=frame)

        tracked: List[Tuple[int, List[float], float, int]] = []
        if out is not None and len(out) > 0:
            for row in out:
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                tid = int(row[4])
                score = float(row[5])
                det_idx = int(row[7]) if len(row) > 7 else -1

                # Update track's EMA Lab jersey color
                if 0 <= det_idx < len(self._curr_det_labs):
                    lab = self._curr_det_labs[det_idx]
                    if lab is not None:
                        prev = self.track_lab_ema.get(tid)
                        if prev is None:
                            self.track_lab_ema[tid] = lab.copy()
                        else:
                            self.track_lab_ema[tid] = (0.82 * prev + 0.18 * lab).astype(np.float32)

                tracked.append((tid, [x1, y1, x2, y2], score, det_idx))

        return tracked, self.last_warp.copy()


def consolidate_player_tracks(
    raw_tracks: Dict[int, Dict[int, List[float]]],
    track_colors_bgr: Dict[int, List[np.ndarray]],
    frame_warps: Dict[int, np.ndarray],
    max_frame_gap: int = 95,
    max_lab_dist: float = 28.0
) -> Tuple[Dict[int, Dict[int, List[float]]], Dict[int, List[np.ndarray]]]:
    """
    Global Post-Pass Tracklet Stitcher (Camera-Motion-Compensated + Appearance-Aware).

    Merges fragmented player tracklets caused by long occlusions, brief frame exits,
    or aggressive camera pans. Because endpoints are projected forward through the
    exact frame-to-frame GMC affine warps `frame_warps`, camera pans of 400+ pixels
    do not break re-association!

    Args:
        raw_tracks:       {tid: {frame_idx: [x1, y1, x2, y2]}}
        track_colors_bgr: {tid: [bgr_sample, ...]}
        frame_warps:      {frame_idx: 2x3 GMC affine matrix}
        max_frame_gap:    Max frame gap to bridge (~3.8s at 25fps)
        max_lab_dist:     Max CIE Lab distance between median jersey colours

    Returns:
        (merged_tracks, merged_colors_bgr)
    """
    import cv2

    if not raw_tracks:
        return {}, {}

    # Compute robust median Lab color per raw tracklet
    def _median_lab(bgr_list: List[np.ndarray]) -> Optional[np.ndarray]:
        if not bgr_list:
            return None
        arr = np.clip(np.asarray(bgr_list, dtype=np.float32), 0, 255).astype(np.uint8)
        labs = cv2.cvtColor(arr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        return np.median(labs, axis=0)

    tracks = {tid: dict(fmap) for tid, fmap in raw_tracks.items() if fmap}
    colors = {tid: list(track_colors_bgr.get(tid, [])) for tid in tracks}
    med_labs = {tid: _median_lab(colors[tid]) for tid in tracks}

    # Iteratively merge best non-overlapping (tail -> head) pair until convergence
    while True:
        tids = sorted(tracks.keys(), key=lambda t: min(tracks[t].keys()))
        best_pair: Optional[Tuple[int, int]] = None
        best_cost = float("inf")

        # Precompute start/end metadata for speed
        meta = {}
        for tid in tids:
            fkeys = sorted(tracks[tid].keys())
            f_start, f_end = fkeys[0], fkeys[-1]
            b_start, b_end = tracks[tid][f_start], tracks[tid][f_end]
            meta[tid] = (
                f_start,
                f_end,
                ((b_start[0] + b_start[2]) * 0.5, b_start[3]),
                ((b_end[0] + b_end[2]) * 0.5, b_end[3]),
                max(10.0, b_start[3] - b_start[1]),
                max(10.0, b_end[3] - b_end[1]),
            )

        for i, id_b in enumerate(tids):
            f_start_b, _, pos_start_b, _, h_start_b, _ = meta[id_b]
            lab_b = med_labs.get(id_b)

            for id_a in tids[:i]:
                _, f_end_a, _, pos_end_a, _, h_end_a = meta[id_a]
                gap = f_start_b - f_end_a
                if gap < 1 or gap > max_frame_gap:
                    continue

                # Scale sanity gate
                h_ratio = max(h_end_a, h_start_b) / max(1.0, min(h_end_a, h_start_b))
                if h_ratio > 1.65:
                    continue

                # Appearance similarity gate
                lab_a = med_labs.get(id_a)
                c_dist = 14.0
                if lab_a is not None and lab_b is not None:
                    c_dist = float(np.linalg.norm(lab_a - lab_b))
                    if c_dist > max_lab_dist:
                        continue

                # Project Track A's last known foot position forward across the camera pan!
                proj_end_a = project_point_across_frames(pos_end_a, f_end_a, f_start_b, frame_warps)
                dist_px = math.hypot(pos_start_b[0] - proj_end_a[0], pos_start_b[1] - proj_end_a[1])

                # Max physical displacement in pixels (accounts for running speed + small GMC drift)
                max_dist = min(380.0, 48.0 + 10.5 * gap)
                if dist_px > max_dist:
                    continue

                # Combined matching score (lower is better)
                cost = (dist_px / max_dist) + 0.45 * (c_dist / max_lab_dist) + 0.15 * (gap / max_frame_gap)
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (id_a, id_b)

        if best_pair is None or best_cost > 1.15:
            break

        id_a, id_b = best_pair
        # Merge id_b into id_a
        tracks[id_a].update(tracks[id_b])
        colors[id_a].extend(colors[id_b])
        med_labs[id_a] = _median_lab(colors[id_a])
        del tracks[id_b]
        del colors[id_b]
        del med_labs[id_b]

    return tracks, colors


def interpolate_player_gaps(
    tracks: Dict[int, Dict[int, List[float]]],
    frame_warps: Dict[int, np.ndarray],
    max_gap: int = 12
) -> Dict[int, Dict[int, List[float]]]:
    """
    Smoothly fills short occlusion gaps (up to `max_gap` frames) per player using
    GMC-compensated linear interpolation so players do not flicker off-and-on.
    """
    filled: Dict[int, Dict[int, List[float]]] = {}
    for tid, fmap in tracks.items():
        if not fmap:
            continue
        out_fmap = dict(fmap)
        fkeys = sorted(fmap.keys())
        for k in range(len(fkeys) - 1):
            fa, fb = fkeys[k], fkeys[k + 1]
            gap = fb - fa
            if 1 < gap <= max_gap:
                ba = fmap[fa]
                bb = fmap[fb]
                ca = ((ba[0] + ba[2]) * 0.5, ba[3])
                cb = ((bb[0] + bb[2]) * 0.5, bb[3])
                wa, ha = ba[2] - ba[0], ba[3] - ba[1]
                wb, hb = bb[2] - bb[0], bb[3] - bb[1]

                # Project start point forward to fb via GMC so we interpolate true pitch motion
                ca_at_fb = project_point_across_frames(ca, fa, fb, frame_warps)
                rel_dx = cb[0] - ca_at_fb[0]
                rel_dy = cb[1] - ca_at_fb[1]

                # Guard against interpolating huge unnatural teleports
                if math.hypot(rel_dx, rel_dy) > 15.0 * gap + 35.0:
                    continue

                cur_warped = ca
                for f in range(fa + 1, fb):
                    H = frame_warps.get(f)
                    if H is not None:
                        cur_warped = warp_point(cur_warped, H)
                    t = (f - fa) / float(gap)
                    ix = cur_warped[0] + t * rel_dx
                    iy = cur_warped[1] + t * rel_dy
                    iw = wa + t * (wb - wa)
                    ih = ha + t * (hb - ha)
                    out_fmap[f] = [ix - iw * 0.5, iy - ih, ix + iw * 0.5, iy]
        filled[tid] = out_fmap
    return filled
