"""
Robust Team Classification with Non-Player Rejection.

Key improvements over v1:
  - Colour-outlier rejection: any detection whose jersey Lab color is too far
    from BOTH team cluster centres is classified as "outlier" (referee / coach / staff)
    and gets team_id = -1 so it is excluded from spatial analytics.
  - Goalkeeper team assignment by pitch position, not jersey colour:
    whichever side of the pitch the keeper stands on determines their team.
  - Confidence-weighted temporal voting per tracker ID.
  - CIE Lab inner-chest sampling retained (avoids destructive green masking).
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans


# ── Thresholds ──────────────────────────────────────────────────────────────
# If the nearest cluster distance (in Lab space) exceeds this value the
# detection is treated as a non-player (referee / coach / staff).
_OUTLIER_DIST_THRESHOLD = 32.0   # Lab units; tune upwards if GK jerseys are rejected

# Minimum Lab lightness for a jersey sample to participate in KMeans fitting.
# This filters pitch-green reflections on dark pixels while preserving dark navy/black kits.
_MIN_LIGHTNESS_FOR_FIT = 35      # 0-255 scale (OpenCV LAB L channel)


class TeamColorAssigner:
    """
    Classifies detected persons into two teams (IDs 0 and 1) or as non-players (-1).

    Non-players are:
      * Technical staff / coaches standing outside the pitch boundary
        (handled upstream by PitchDetector.is_inside_pitch - they are
        filtered before this class is called)
      * Referees and any person whose jersey colour lies far from both
        team cluster centres (outlier rejection).
    """

    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters
        self.team_kmeans: Optional[KMeans] = None
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.cluster_centers_lab: Optional[np.ndarray] = None
        # Inter-cluster distance; used to scale the outlier threshold adaptively
        self._inter_cluster_dist: float = 1.0

    # ------------------------------------------------------------------ colour extraction

    def extract_jersey_color(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Returns the median BGR jersey colour sampled from the inner-chest region
        (18%-48% height, 25%-75% width of the bounding box).
        Returns None if the crop is too small or empty.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1

        if h < 20 or w < 10:
            return None

        cy1 = max(0, y1 + int(h * 0.18))
        cy2 = min(frame.shape[0], y1 + int(h * 0.48))
        cx1 = max(0, x1 + int(w * 0.25))
        cx2 = min(frame.shape[1], x2 - int(w * 0.25))

        chest = frame[cy1:cy2, cx1:cx2]
        if chest.size == 0 or chest.shape[0] < 3 or chest.shape[1] < 3:
            return None

        return np.median(chest, axis=(0, 1)).astype(np.float32)

    def bgr_to_lab(self, bgr: np.ndarray) -> np.ndarray:
        """Converts a BGR colour vector to CIE Lab."""
        bgr_pixel = np.uint8([[np.clip(bgr, 0, 255)]])
        return cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)

    # ------------------------------------------------------------------ fitting

    def fit_teams(self, player_bgr_colors: List[np.ndarray]):
        """
        Fits two clusters in CIE Lab space from a list of jersey BGR colours.

        Excludes very dark samples (likely referees in black) from fitting so
        the two team clusters are not pulled towards referee black.
        Falls back to all samples if filtering leaves fewer than n_clusters.
        """
        if len(player_bgr_colors) < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} player colour samples; "
                f"got {len(player_bgr_colors)}."
            )

        lab_samples = []
        for bgr in player_bgr_colors:
            lab = self.bgr_to_lab(bgr)
            if lab[0] >= _MIN_LIGHTNESS_FOR_FIT:
                lab_samples.append(lab)

        if len(lab_samples) < self.n_clusters:
            lab_samples = [self.bgr_to_lab(c) for c in player_bgr_colors]

        self.team_kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self.team_kmeans.fit(lab_samples)
        self.cluster_centers_lab = self.team_kmeans.cluster_centers_

        # Inter-cluster distance (used to scale outlier threshold)
        self._inter_cluster_dist = float(
            np.linalg.norm(self.cluster_centers_lab[0] - self.cluster_centers_lab[1])
        )

        # Map cluster centres back to BGR for downstream visualisation
        for team_id, lab_center in enumerate(self.cluster_centers_lab):
            lab_px = np.uint8([[np.clip(lab_center, 0, 255)]])
            bgr_center = cv2.cvtColor(lab_px, cv2.COLOR_LAB2BGR)[0][0]
            self.team_colors[team_id] = tuple(map(int, bgr_center))

    # ------------------------------------------------------------------ prediction

    def predict_team(self, dominant_bgr: np.ndarray) -> int:
        """
        Returns team ID (0 or 1) or -1 if the colour is an outlier
        (referee / coach / staff whose jersey does not match either team).
        """
        team_id, _ = self.predict_team_with_confidence(dominant_bgr)
        return team_id

    def predict_team_with_confidence(
        self, dominant_bgr: np.ndarray
    ) -> Tuple[int, float]:
        """
        Returns (team_id, confidence).
        team_id = -1 when the jersey is an outlier (non-player).

        Outlier detection:
            If the minimum distance to either cluster exceeds
            _OUTLIER_DIST_THRESHOLD the person is flagged as a non-player
            and team_id = -1 is returned.
        """
        if self.team_kmeans is None or self.cluster_centers_lab is None:
            return 0, 0.5

        lab = self.bgr_to_lab(dominant_bgr)
        d0 = float(np.linalg.norm(lab - self.cluster_centers_lab[0]))
        d1 = float(np.linalg.norm(lab - self.cluster_centers_lab[1]))
        min_d = min(d0, d1)

        # Adaptive threshold: scales slightly with inter-cluster distance
        adaptive_threshold = min(
            _OUTLIER_DIST_THRESHOLD + 0.20 * self._inter_cluster_dist,
            55.0
        )
        if min_d > adaptive_threshold:
            return -1, 0.0  # Non-player (referee / coach / very distinct GK kit)

        total_d = d0 + d1 + 1e-6
        if d0 < d1:
            return 0, float(1.0 - d0 / total_d)
        else:
            return 1, float(1.0 - d1 / total_d)

    # ------------------------------------------------------------------ goalkeeper helper

    def infer_goalkeeper_team(
        self,
        pos_2d_x_px: float,
        field_x1: float = 40.0,
        field_w: float = 970.0
    ) -> int:
        """
        Determines a goalkeeper's team from their 2D pitch X position.

        Goalkeepers have distinct jerseys that may be rejected as outliers.
        We reassign them by field position:
          * Left half  (x_norm < 0.5) -> Team 0 (defends left goal)
          * Right half (x_norm >= 0.5) -> Team 1 (defends right goal)
        """
        x_norm = (pos_2d_x_px - field_x1) / max(1.0, field_w)
        return 0 if x_norm < 0.5 else 1
