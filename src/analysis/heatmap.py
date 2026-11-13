"""
Heatmap and average position spatial analytics module.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class SpatialAnalytics:
    """Computes player movement heatmaps and average pitch positions."""

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        self.width = frame_width
        self.height = frame_height
        # Mapping from tracker_id to list of (x, y) bottom-center coordinates
        self.player_tracks: Dict[int, List[Tuple[float, float]]] = {}
        # Mapping from tracker_id to team_id
        self.player_teams: Dict[int, int] = {}

    def add_position(self, tracker_id: int, bbox: List[float], team_id: int):
        """
        Adds player's bottom-center coordinate (feet location) for the current frame.
        """
        x1, y1, x2, y2 = bbox
        bottom_center_x = (x1 + x2) / 2.0
        bottom_center_y = y2
        
        if tracker_id not in self.player_tracks:
            self.player_tracks[tracker_id] = []
        self.player_tracks[tracker_id].append((bottom_center_x, bottom_center_y))
        self.player_teams[tracker_id] = team_id

    def get_average_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Calculates average (mean) (x, y) position for each player.
        """
        avg_positions = {}
        for tracker_id, positions in self.player_tracks.items():
            if len(positions) > 0:
                coords = np.array(positions)
                avg_positions[tracker_id] = (float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1])))
        return avg_positions

    def generate_heatmap(
        self,
        tracker_id: Optional[int] = None,
        team_id: Optional[int] = None,
        sigma: float = 25.0
    ) -> np.ndarray:
        """
        Generates a 2D Gaussian density heatmap for a specific player or an entire team.
        Returns a colored BGR heatmap image.
        """
        density = np.zeros((self.height, self.width), dtype=np.float32)

        coords_to_include = []
        for p_id, positions in self.player_tracks.items():
            if tracker_id is not None and p_id != tracker_id:
                continue
            if team_id is not None and self.player_teams.get(p_id) != team_id:
                continue
            coords_to_include.extend(positions)

        if not coords_to_include:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Accumulate coordinate hits
        for x, y in coords_to_include:
            ix, iy = int(x), int(y)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                density[iy, ix] += 1.0

        # Apply 2D Gaussian blur
        density = gaussian_filter(density, sigma=sigma)

        # Normalize to [0, 255]
        max_val = np.max(density)
        if max_val > 0:
            density = (density / max_val * 255).astype(np.uint8)
        else:
            density = density.astype(np.uint8)

        # Apply colormap
        heatmap_bgr = cv2.applyColorMap(density, cv2.COLORMAP_JET)
        
        # Zero out areas with negligible density
        mask = density < 10
        heatmap_bgr[mask] = 0
        return heatmap_bgr
