"""
Robust Team Classification based on jersey color clustering in CIE Lab space.
Avoids destructive pitch-masking that corrupts green-jersey teams.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamColorAssigner:
    """
    Classifies players into two opposing teams based on jersey color clustering.
    Uses perceptually uniform CIE Lab color space and inner-chest sampling.
    """

    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters
        self.team_kmeans: Optional[KMeans] = None
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.cluster_centers_lab: Optional[np.ndarray] = None

    def extract_jersey_color(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Extracts dominant jersey BGR color from the inner chest region of a player.
        The inner chest (height 18%-48%, width 25%-75%) is almost exclusively jersey fabric,
        preventing background pitch bleed without needing destructive green masks.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1

        if h < 20 or w < 10:
            return None

        # Sample inner chest
        crop_y1 = max(0, y1 + int(h * 0.18))
        crop_y2 = min(frame.shape[0], y1 + int(h * 0.48))
        crop_x1 = max(0, x1 + int(w * 0.25))
        crop_x2 = min(frame.shape[1], x2 - int(w * 0.25))

        chest = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if chest.size == 0 or chest.shape[0] < 3 or chest.shape[1] < 3:
            return None

        # Robust median BGR color across chest patch
        median_bgr = np.median(chest, axis=(0, 1)).astype(np.float32)
        return median_bgr

    def bgr_to_lab(self, bgr: np.ndarray) -> np.ndarray:
        """Converts a BGR color vector to CIE Lab space."""
        bgr_pixel = np.uint8([[np.clip(bgr, 0, 255)]])
        lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0][0]
        return lab_pixel.astype(np.float32)

    def fit_teams(self, player_bgr_colors: List[np.ndarray]):
        """
        Fits 2 clusters (Team 0 and Team 1) in CIE Lab color space.
        Filters out extreme dark outliers (referees/goalkeepers) during fitting.
        """
        if len(player_bgr_colors) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} player colors to fit teams.")

        lab_samples = []
        valid_bgrs = []
        for bgr in player_bgr_colors:
            lab = self.bgr_to_lab(bgr)
            # Filter out black/extremely dark referee samples (L < 40 in 0..255 scale)
            if lab[0] >= 40:
                lab_samples.append(lab)
                valid_bgrs.append(bgr)

        # Fallback if too many samples were filtered
        if len(lab_samples) < self.n_clusters:
            lab_samples = [self.bgr_to_lab(c) for c in player_bgr_colors]

        self.team_kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self.team_kmeans.fit(lab_samples)
        self.cluster_centers_lab = self.team_kmeans.cluster_centers_

        # Map cluster centers back to BGR for visualization
        for team_id, lab_center in enumerate(self.cluster_centers_lab):
            lab_px = np.uint8([[np.clip(lab_center, 0, 255)]])
            bgr_center = cv2.cvtColor(lab_px, cv2.COLOR_LAB2BGR)[0][0]
            self.team_colors[team_id] = tuple(map(int, bgr_center))

    def predict_team(self, dominant_bgr: np.ndarray) -> int:
        """Predicts team ID (0 or 1) in CIE Lab color space."""
        if self.team_kmeans is None:
            raise RuntimeError("TeamColorAssigner must be fitted with fit_teams() first.")

        lab = self.bgr_to_lab(dominant_bgr)
        team_id = int(self.team_kmeans.predict([lab])[0])
        return team_id

    def predict_team_with_confidence(self, dominant_bgr: np.ndarray) -> Tuple[int, float]:
        """
        Predicts team ID and provides confidence based on relative Lab Euclidean distance.
        """
        if self.team_kmeans is None or self.cluster_centers_lab is None:
            return 0, 0.5

        lab = self.bgr_to_lab(dominant_bgr)
        d0 = np.linalg.norm(lab - self.cluster_centers_lab[0])
        d1 = np.linalg.norm(lab - self.cluster_centers_lab[1])
        
        total_d = d0 + d1 + 1e-6
        if d0 < d1:
            team_id = 0
            conf = 1.0 - (d0 / total_d)
        else:
            team_id = 1
            conf = 1.0 - (d1 / total_d)
        return team_id, float(conf)
