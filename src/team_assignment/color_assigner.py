"""
Team classification based on player jersey dominant colors using K-Means clustering.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamColorAssigner:
    """Classifies players into two teams based on jersey color clustering."""

    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters
        self.team_kmeans: Optional[KMeans] = None
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.player_team_dict: Dict[int, int] = {}

    def extract_jersey_color(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Extracts dominant jersey color from the upper torso region of a player bounding box.
        bbox format: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1

        if h <= 0 or w <= 0:
            return None

        # Focus on upper torso (avoid head/hair, shorts, and green grass)
        crop_y1 = max(0, y1 + int(h * 0.15))
        crop_y2 = min(frame.shape[0], y1 + int(h * 0.55))
        crop_x1 = max(0, x1 + int(w * 0.20))
        crop_x2 = min(frame.shape[1], x2 - int(w * 0.20))

        torso = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if torso.size == 0 or torso.shape[0] < 5 or torso.shape[1] < 5:
            return None

        # Filter out green grass pixels in HSV space
        hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        grass_mask = cv2.inRange(hsv_torso, lower_green, upper_green)
        
        non_grass_pixels = torso[grass_mask == 0]
        if len(non_grass_pixels) < 25:
            # Fallback to entire torso if grass filtering removes too much
            non_grass_pixels = torso.reshape(-1, 3)

        # Get dominant color via K-Means (k=1)
        kmeans = KMeans(n_clusters=1, n_init=3, random_state=42)
        kmeans.fit(non_grass_pixels)
        dominant_bgr = kmeans.cluster_centers_[0]
        return dominant_bgr

    def fit_teams(self, player_colors: List[np.ndarray]):
        """
        Fits 2 clusters (Team 0 and Team 1) on extracted player colors.
        """
        if len(player_colors) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} player colors to fit teams.")

        self.team_kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self.team_kmeans.fit(player_colors)
        
        for team_id, center in enumerate(self.team_kmeans.cluster_centers_):
            # Convert BGR float to int tuple
            self.team_colors[team_id] = tuple(map(int, center))

    def predict_team(self, dominant_bgr: np.ndarray) -> int:
        """Predicts team ID (0 or 1) for a given jersey BGR color."""
        if self.team_kmeans is None:
            raise RuntimeError("TeamColorAssigner must be fitted with fit_teams() first.")
        team_id = int(self.team_kmeans.predict([dominant_bgr])[0])
        return team_id
