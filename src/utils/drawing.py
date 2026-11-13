"""
Visualization and drawing utilities for tactical pitch analysis.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class TacticalVisualizer:
    """Draws player foot ellipses, ID tags, ball markers, and average positions."""

    # Pre-defined team theme colors (BGR)
    TEAM_COLORS = {
        0: (220, 50, 50),     # Team 0 (Blue-ish / Red-ish)
        1: (40, 200, 40),     # Team 1 (Green-ish)
        -1: (20, 200, 220)    # Referee / Unknown (Yellow)
    }

    @staticmethod
    def draw_player_ellipse(
        frame: np.ndarray,
        bbox: List[float],
        color: Tuple[int, int, int],
        tracker_id: Optional[int] = None
    ):
        """Draws a neat tactical ellipse under the player's feet with a subtle shadow."""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2.0)
        bottom_y = int(y2)
        width = int((x2 - x1) * 0.8)
        axes = (max(5, width // 2), max(3, width // 4))

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.ellipse(overlay, (center_x, bottom_y), axes, 0, 0, 360, color, -1, cv2.LINE_AA)
        cv2.ellipse(overlay, (center_x, bottom_y), axes, 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw tracker ID label if available
        if tracker_id is not None:
            label = f"#{tracker_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            label_x = center_x - text_w // 2
            label_y = bottom_y + text_h + 8
            
            # Badge background
            cv2.rectangle(
                frame,
                (label_x - 3, label_y - text_h - 2),
                (label_x + text_w + 3, label_y + baseline),
                (20, 20, 20),
                -1
            )
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

    @staticmethod
    def draw_ball_pointer(frame: np.ndarray, bbox: List[float]):
        """Draws a modern downward pointing triangle above the ball."""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2.0)
        top_y = int(y1) - 4

        triangle_pts = np.array([
            [center_x, top_y],
            [center_x - 7, top_y - 12],
            [center_x + 7, top_y - 12]
        ], np.int32)

        cv2.drawContours(frame, [triangle_pts], 0, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.drawContours(frame, [triangle_pts], 0, (0, 0, 0), 1, cv2.LINE_AA)

    @staticmethod
    def draw_average_positions(
        background_frame: np.ndarray,
        avg_positions: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, int]
    ) -> np.ndarray:
        """
        Draws average player positions onto a frame/pitch.
        """
        canvas = background_frame.copy()
        for p_id, (avg_x, avg_y) in avg_positions.items():
            team_id = player_teams.get(p_id, 0)
            color = TacticalVisualizer.TEAM_COLORS.get(team_id, (200, 200, 200))
            center = (int(avg_x), int(avg_y))

            # Outer ring & filled circle
            cv2.circle(canvas, center, 14, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, center, 12, color, -1, cv2.LINE_AA)

            # Player ID text
            label = str(p_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            cv2.putText(
                canvas,
                label,
                (center[0] - tw // 2, center[1] + th // 2),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        return canvas

