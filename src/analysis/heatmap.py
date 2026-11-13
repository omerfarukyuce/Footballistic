"""
Heatmap, trajectory, and average position spatial analytics module.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class SpatialAnalytics:
    """Computes player movement heatmaps, trajectories, and average pitch positions."""

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        self.width = frame_width
        self.height = frame_height
        # Mapping from tracker_id to list of (x, y) bottom-center coordinates
        self.player_tracks: Dict[int, List[Tuple[float, float]]] = {}
        # Mapping from tracker_id to team_id
        self.player_teams: Dict[int, int] = {}
        self.total_processed_frames: int = 0

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

    def set_total_frames(self, frames: int):
        self.total_processed_frames = max(1, frames)

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

    def get_player_summary(self) -> List[Dict]:
        """
        Returns structured summary table of all tracked players.
        """
        summary = []
        avg_pos = self.get_average_positions()
        total_f = max(1, self.total_processed_frames)

        for p_id in sorted(self.player_tracks.keys()):
            positions = self.player_tracks[p_id]
            team_id = self.player_teams.get(p_id, 0)
            avg_x, avg_y = avg_pos.get(p_id, (0.0, 0.0))
            
            summary.append({
                "Player ID": f"#{p_id}",
                "Raw ID": p_id,
                "Team": f"Team {team_id}",
                "Team ID": team_id,
                "Frames Active": len(positions),
                "Presence (%)": round((len(positions) / total_f) * 100, 1),
                "Avg X (px)": round(avg_x, 1),
                "Avg Y (px)": round(avg_y, 1),
            })
        return summary

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
        mask = density < 8
        heatmap_bgr[mask] = 0
        return heatmap_bgr

    def generate_player_overlay(
        self,
        tracker_id: int,
        background_frame: np.ndarray,
        sigma: float = 20.0
    ) -> np.ndarray:
        """
        Generates a heatmap for a single player overlaid onto the match background frame.
        """
        heatmap_bgr = self.generate_heatmap(tracker_id=tracker_id, sigma=sigma)
        overlay = background_frame.copy()
        nonzero_mask = np.any(heatmap_bgr > 0, axis=-1)

        if np.any(nonzero_mask):
            overlay[nonzero_mask] = cv2.addWeighted(
                overlay[nonzero_mask], 0.35,
                heatmap_bgr[nonzero_mask], 0.65,
                0
            )
            
            # Also draw average position point on top of the heatmap
            avg_positions = self.get_average_positions()
            if tracker_id in avg_positions:
                avg_x, avg_y = avg_positions[tracker_id]
                center = (int(avg_x), int(avg_y))
                cv2.circle(overlay, center, 14, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(overlay, center, 11, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, center, 7, (0, 0, 255), -1, cv2.LINE_AA)

        return overlay

    def generate_player_trajectory(
        self,
        tracker_id: int,
        background_frame: np.ndarray
    ) -> np.ndarray:
        """
        Draws the movement trajectory trail of a specific player over time.
        """
        canvas = background_frame.copy()
        positions = self.player_tracks.get(tracker_id, [])
        if len(positions) < 2:
            return canvas

        team_id = self.player_teams.get(tracker_id, 0)
        trail_color = (255, 120, 0) if team_id == 0 else (0, 220, 100)

        pts = np.array(positions, np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts], isClosed=False, color=trail_color, thickness=3, lineType=cv2.LINE_AA)

        # Draw Start Point (Green circle)
        start_pt = (int(positions[0][0]), int(positions[0][1]))
        cv2.circle(canvas, start_pt, 8, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, start_pt, 9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Start", (start_pt[0] + 10, start_pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw End Point (Red circle)
        end_pt = (int(positions[-1][0]), int(positions[-1][1]))
        cv2.circle(canvas, end_pt, 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, end_pt, 9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Current", (end_pt[0] + 10, end_pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        return canvas
