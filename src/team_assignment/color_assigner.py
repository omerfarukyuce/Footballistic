"""
Robust Team Classification, Grass-Excluded Jersey Sampling, Goalkeeper
Detection, and Non-Player (Referee / Coach) Rejection.

Key design principles:
  1. Grass-excluded torso color extraction: masks out green pitch pixels inside
     the player's upper-torso crop before computing the median color, so lean /
     distant / sprinting players retain pure jersey colors.
  2. Track-level classification: aggregates Lab color samples across ALL frames
     of each consolidated track (hundreds of samples) instead of making brittle
     per-frame decisions that drop players on a single shadowed frame.
  3. Iterative trimmed KMeans (k=2): supports ANY kit colors — including dark
     navy or black kits — while preventing the single referee and 1-2 goalkeepers
     from skewing the two outfield team clusters.
  4. Position-aware Goalkeeper assignment: outlier-colored tracks positioned on
     the defensive left/right extremes are identified as Goalkeepers and assigned
     to the team defending that half (the team whose outfielders are closer to
     that goal). Midfield color outliers are rejected as Referees.
"""

from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans


# Default fallback threshold for online single-frame classification
_OUTLIER_DIST_THRESHOLD = 36.0


class TeamColorAssigner:
    """
    Classifies football tracks into Team 0, Team 1, or Non-Player (-1: referee/staff),
    and explicitly identifies each team's Goalkeeper by defensive extremity.
    """

    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters
        self.team_kmeans: Optional[KMeans] = None
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.cluster_centers_lab: Optional[np.ndarray] = None
        self._inter_cluster_dist: float = 1.0
        self._cluster_spread: Tuple[float, float] = (12.0, 12.0)

    # ------------------------------------------------------------------ colour extraction

    def extract_jersey_color(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Returns the median BGR jersey colour from the upper-torso region
        (16%-48% height, 22%-78% width) after masking out green grass pixels.
        Works reliably for both close and small distant players (down to 10x18 px).
        """
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1

        if h < 14 or w < 6:
            return None

        fh, fw = frame.shape[:2]
        cy1 = max(0, y1 + int(h * 0.16))
        cy2 = min(fh, y1 + int(h * 0.48))
        cx1 = max(0, x1 + int(w * 0.22))
        cx2 = min(fw, x2 - int(w * 0.22))

        if cy2 <= cy1 + 1 or cx2 <= cx1 + 1:
            return None

        chest = frame[cy1:cy2, cx1:cx2]
        if chest.size == 0:
            return None

        # Mask out green grass background pixels inside the torso box
        hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
        grass_mask = (
            (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 85) &
            (hsv[:, :, 1] >= 42) & (hsv[:, :, 2] >= 35)
        )
        non_grass = chest[~grass_mask]

        # Use non-grass pixels if sufficient pixels remain (otherwise jersey itself may be green)
        if len(non_grass) >= max(8, int(chest.shape[0] * chest.shape[1] * 0.15)):
            return np.median(non_grass, axis=0).astype(np.float32)

        # Fallback for green jerseys: sample tight central core of torso
        ty1 = max(0, y1 + int(h * 0.20))
        ty2 = min(fh, y1 + int(h * 0.42))
        tx1 = max(0, x1 + int(w * 0.32))
        tx2 = min(fw, x2 - int(w * 0.32))
        core = frame[ty1:ty2, tx1:tx2]
        if core.size == 0:
            return np.median(chest, axis=(0, 1)).astype(np.float32)
        return np.median(core, axis=(0, 1)).astype(np.float32)

    @staticmethod
    def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
        """Converts a single BGR colour vector to CIE Lab (float32)."""
        bgr_pixel = np.uint8([[np.clip(bgr, 0, 255)]])
        return cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)

    @staticmethod
    def bgr_list_to_lab(bgr_list: List[np.ndarray]) -> np.ndarray:
        """Vectorised conversion of a list of BGR vectors to (N, 3) float32 CIE Lab."""
        arr = np.clip(np.asarray(bgr_list, dtype=np.float32), 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(arr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
        return lab.reshape(-1, 3).astype(np.float32)

    # ------------------------------------------------------------------ fitting

    def fit_teams(self, player_bgr_colors: List[np.ndarray]):
        """
        Fits two outfield team clusters in CIE Lab space using two-stage
        trimmed KMeans so isolated outliers (referees / goalkeepers) do not
        distort the team cluster centres, while preserving dark/black kits.
        """
        if len(player_bgr_colors) < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} player colour samples; "
                f"got {len(player_bgr_colors)}."
            )

        lab_samples = self.bgr_list_to_lab(player_bgr_colors)

        # Stage 1: Initial 2-cluster fit
        km1 = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        labels1 = km1.fit_predict(lab_samples)
        centers1 = km1.cluster_centers_

        # Stage 2: Trim the furthest 12% outlier samples (refs/GKs) and refit cleanly
        if len(lab_samples) >= 12:
            dists1 = np.array([
                float(np.linalg.norm(lab_samples[i] - centers1[labels1[i]]))
                for i in range(len(lab_samples))
            ])
            cutoff = float(np.percentile(dists1, 88))
            inliers = lab_samples[dists1 <= max(cutoff, 20.0)]
            if len(inliers) >= self.n_clusters * 2:
                km2 = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
                km2.fit(inliers)
                self.team_kmeans = km2
                self.cluster_centers_lab = km2.cluster_centers_
            else:
                self.team_kmeans = km1
                self.cluster_centers_lab = centers1
        else:
            self.team_kmeans = km1
            self.cluster_centers_lab = centers1

        self._finalize_clusters(lab_samples)

    def _finalize_clusters(self, lab_samples: np.ndarray):
        """Orders cluster centres deterministically and computes cluster spreads."""
        # Sort clusters deterministically by lightness (darker = Team 0, lighter = Team 1)
        # so Team 0 / Team 1 stay consistent across runs on the same video
        order = np.argsort(self.cluster_centers_lab[:, 0])
        self.cluster_centers_lab = self.cluster_centers_lab[order]
        if self.team_kmeans is not None:
            self.team_kmeans.cluster_centers_ = self.cluster_centers_lab.copy()

        self._inter_cluster_dist = float(
            np.linalg.norm(self.cluster_centers_lab[0] - self.cluster_centers_lab[1])
        )

        # Compute per-cluster robust spread (IQR/MAD-like) for adaptive outlier gating
        spreads = []
        d0_all = np.linalg.norm(lab_samples - self.cluster_centers_lab[0], axis=1)
        d1_all = np.linalg.norm(lab_samples - self.cluster_centers_lab[1], axis=1)
        assigned = np.where(d0_all < d1_all, 0, 1)
        for cid in (0, 1):
            c_dists = d0_all[assigned == 0] if cid == 0 else d1_all[assigned == 1]
            if len(c_dists) >= 3:
                # 75th percentile distance captures real lighting/angle variation
                spreads.append(float(np.percentile(c_dists, 75)))
            else:
                spreads.append(14.0)
        self._cluster_spread = (spreads[0], spreads[1])

        # Map cluster centres back to BGR for downstream visualisation
        for team_id, lab_center in enumerate(self.cluster_centers_lab):
            lab_px = np.uint8([[np.clip(lab_center, 0, 255)]])
            bgr_center = cv2.cvtColor(lab_px, cv2.COLOR_LAB2BGR)[0][0]
            self.team_colors[team_id] = tuple(map(int, bgr_center))

    # ------------------------------------------------------------------ track-level classification

    def classify_tracks(
        self,
        track_colors: Dict[int, List[np.ndarray]],
        track_positions_2d: Dict[int, List[Tuple[float, float]]],
        field_x1: float = 40.0,
        field_w: float = 970.0,
        field_y1: float = 40.0,
        field_h: float = 600.0
    ) -> Tuple[Dict[int, int], Set[int]]:
        """
        Robust track-level classification across the full clip.

        1. Computes each track's median Lab jersey colour across all its frames.
        2. Refits the 2 outfield team clusters on long tracks (weighted by duration)
           with iterative outlier trimming so referees/GKs cannot skew clusters.
        3. Assigns outfield tracks matching either cluster to Team 0 or Team 1.
        4. For outlier-coloured tracks (referees / goalkeepers):
           - Determines which team defends left vs right from outfield team centroids.
           - If the outlier track sits at the far-left or far-right defensive extreme
             (inside the goalkeeper corridor), assigns it as that team's Goalkeeper!
           - Otherwise marks it as team_id = -1 (Referee / non-player -> dropped).

        Returns:
            (track_teams, goalkeeper_ids)
        """
        track_median_lab: Dict[int, np.ndarray] = {}
        weighted_lab_pool: List[np.ndarray] = []

        for tid, bgr_list in track_colors.items():
            if not bgr_list:
                continue
            labs = self.bgr_list_to_lab(bgr_list)
            med_lab = np.median(labs, axis=0).astype(np.float32)
            track_median_lab[tid] = med_lab

            # Weight tracks with more frames into the KMeans calibration pool (capped per track
            # so one super-long track cannot dominate)
            if len(bgr_list) >= 5:
                reps = min(10, max(1, len(bgr_list) // 15))
                for _ in range(reps):
                    weighted_lab_pool.append(med_lab)

        # Refit team clusters from clean track medians if we have enough tracks
        if len(weighted_lab_pool) >= 6:
            pool_arr = np.asarray(weighted_lab_pool, dtype=np.float32)
            km1 = KMeans(n_clusters=2, n_init=15, random_state=42)
            lbl1 = km1.fit_predict(pool_arr)
            c1 = km1.cluster_centers_

            # Iterative trim: drop isolated outliers (referee + GKs, furthest ~15%)
            dists1 = np.array([
                float(np.linalg.norm(pool_arr[i] - c1[lbl1[i]]))
                for i in range(len(pool_arr))
            ])
            trim_thresh = max(18.0, float(np.percentile(dists1, 86)))
            inliers = pool_arr[dists1 <= trim_thresh]
            if len(inliers) >= 4:
                km2 = KMeans(n_clusters=2, n_init=15, random_state=42)
                km2.fit(inliers)
                self.team_kmeans = km2
                self.cluster_centers_lab = km2.cluster_centers_
                self._finalize_clusters(inliers)
            else:
                self.team_kmeans = km1
                self.cluster_centers_lab = c1
                self._finalize_clusters(pool_arr)

        if self.cluster_centers_lab is None:
            return {tid: 0 for tid in track_positions_2d}, set()

        # Pass A: Classify clear outfielders vs colour outliers
        track_teams: Dict[int, int] = {}
        outlier_tids: List[int] = []
        goalkeeper_ids: Set[int] = set()

        c0, c1 = self.cluster_centers_lab[0], self.cluster_centers_lab[1]
        # Adaptive outlier threshold per cluster: generous enough for shadowed/far
        # outfielders, strict enough to catch yellow/neon referees & red/orange GKs
        t0 = float(np.clip(max(28.0, self._cluster_spread[0] * 2.4 + 12.0), 28.0, 48.0))
        t1 = float(np.clip(max(28.0, self._cluster_spread[1] * 2.4 + 12.0), 28.0, 48.0))
        # Also cap relative to inter-cluster distance so closely spaced kits still separate refs
        inter_cap = max(26.0, self._inter_cluster_dist * 0.65)
        t0 = min(t0, inter_cap)
        t1 = min(t1, inter_cap)

        for tid in track_positions_2d.keys():
            med_lab = track_median_lab.get(tid)
            if med_lab is None:
                outlier_tids.append(tid)
                continue

            d0 = float(np.linalg.norm(med_lab - c0))
            d1 = float(np.linalg.norm(med_lab - c1))

            # Check if track is a colour outlier (far from both team centres, OR
            # high chromaticity mismatch such as bright yellow ref / red GK when teams
            # are low-saturation white & dark navy)
            nearest = 0 if d0 < d1 else 1
            nearest_d = d0 if nearest == 0 else d1
            thresh = t0 if nearest == 0 else t1

            # Extra chromaticity (a*, b* plane) check: catches yellow/red/green refs & GKs
            # even if their lightness L* happens to sit between white and dark navy!
            ab_d0 = float(np.linalg.norm(med_lab[1:] - c0[1:]))
            ab_d1 = float(np.linalg.norm(med_lab[1:] - c1[1:]))
            min_ab_d = min(ab_d0, ab_d1)

            if nearest_d > thresh or min_ab_d > 24.0:
                outlier_tids.append(tid)
            else:
                track_teams[tid] = nearest

        # Pass B: Determine which team defends Left vs Right based on outfielders' mean X
        team_xs = {0: [], 1: []}
        all_outfield_xm = []
        for tid, team_id in track_teams.items():
            pts = track_positions_2d.get(tid, [])
            if pts:
                avg_xm = float(np.mean([(p[0] - field_x1) / max(1.0, field_w) * 105.0 for p in pts]))
                team_xs[team_id].append(avg_xm)
                all_outfield_xm.append(avg_xm)

        mean_x0 = float(np.mean(team_xs[0])) if team_xs[0] else 45.0
        mean_x1 = float(np.mean(team_xs[1])) if team_xs[1] else 60.0
        left_defending_team = 0 if mean_x0 <= mean_x1 else 1
        right_defending_team = 1 - left_defending_team

        # Outfield X envelope (to recognise a keeper standing behind their defensive line)
        min_out_x = float(np.percentile(all_outfield_xm, 5)) if len(all_outfield_xm) >= 4 else 22.0
        max_out_x = float(np.percentile(all_outfield_xm, 95)) if len(all_outfield_xm) >= 4 else 83.0

        # Pass C: Distinguish Goalkeepers from Referees among colour-outlier tracks
        left_gk_candidates: List[Tuple[float, int]] = []
        right_gk_candidates: List[Tuple[float, int]] = []

        for tid in outlier_tids:
            pts = track_positions_2d.get(tid, [])
            if not pts or len(pts) < 5:
                track_teams[tid] = -1
                continue

            xs_m = [(p[0] - field_x1) / max(1.0, field_w) * 105.0 for p in pts]
            ys_m = [(p[1] - field_y1) / max(1.0, field_h) * 68.0 for p in pts]
            avg_xm = float(np.mean(xs_m))
            avg_ym = float(np.mean(ys_m))
            spread_x = float(np.std(xs_m))

            # Goalkeepers stay in the goal corridor (central Y: 10m..58m), do not roam
            # box-to-box like a referee (low X spread), and sit on the extreme left or right
            in_goal_y = 10.0 <= avg_ym <= 58.0
            is_left_gk = in_goal_y and (avg_xm <= max(24.0, min_out_x + 2.0)) and spread_x < 16.0
            is_right_gk = in_goal_y and (avg_xm >= min(81.0, max_out_x - 2.0)) and spread_x < 16.0

            if is_left_gk:
                left_gk_candidates.append((avg_xm, tid))
            elif is_right_gk:
                right_gk_candidates.append((-avg_xm, tid))
            else:
                # Midfield / roaming outlier -> Referee or non-player!
                track_teams[tid] = -1

        # Pick the single most extreme left/right GK candidate (any extras stay -1)
        if left_gk_candidates:
            left_gk_candidates.sort()
            _, best_left_gk = left_gk_candidates[0]
            track_teams[best_left_gk] = left_defending_team
            goalkeeper_ids.add(best_left_gk)
            for _, other_tid in left_gk_candidates[1:]:
                track_teams[other_tid] = -1

        if right_gk_candidates:
            right_gk_candidates.sort()
            _, best_right_gk = right_gk_candidates[0]
            track_teams[best_right_gk] = right_defending_team
            goalkeeper_ids.add(best_right_gk)
            for _, other_tid in right_gk_candidates[1:]:
                track_teams[other_tid] = -1

        return track_teams, goalkeeper_ids

    # ------------------------------------------------------------------ single-frame prediction

    def predict_team(self, dominant_bgr: np.ndarray) -> int:
        """Returns team ID (0 or 1) or -1 if colour is a clear non-player outlier."""
        team_id, _ = self.predict_team_with_confidence(dominant_bgr)
        return team_id

    def predict_team_with_confidence(
        self, dominant_bgr: np.ndarray
    ) -> Tuple[int, float]:
        """Returns (team_id, confidence); team_id = -1 for colour outliers."""
        if self.team_kmeans is None or self.cluster_centers_lab is None:
            return 0, 0.5

        lab = self.bgr_to_lab(dominant_bgr)
        d0 = float(np.linalg.norm(lab - self.cluster_centers_lab[0]))
        d1 = float(np.linalg.norm(lab - self.cluster_centers_lab[1]))
        min_d = min(d0, d1)

        adaptive_threshold = min(
            _OUTLIER_DIST_THRESHOLD + 0.18 * self._inter_cluster_dist,
            52.0
        )
        ab_d0 = float(np.linalg.norm(lab[1:] - self.cluster_centers_lab[0][1:]))
        ab_d1 = float(np.linalg.norm(lab[1:] - self.cluster_centers_lab[1][1:]))
        if min_d > adaptive_threshold or min(ab_d0, ab_d1) > 26.0:
            return -1, 0.0

        total_d = d0 + d1 + 1e-6
        if d0 < d1:
            return 0, float(1.0 - d0 / total_d)
        return 1, float(1.0 - d1 / total_d)

    # ------------------------------------------------------------------ goalkeeper helper

    def infer_goalkeeper_team(
        self,
        pos_2d_x_px: float,
        field_x1: float = 40.0,
        field_w: float = 970.0
    ) -> int:
        """Fallback positional team inference for a goalkeeper (left -> 0, right -> 1)."""
        x_norm = (pos_2d_x_px - field_x1) / max(1.0, field_w)
        return 0 if x_norm < 0.5 else 1
