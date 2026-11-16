"""
Visualization and drawing utilities for tactical pitch analysis.
Includes persistent tactical ellipses, ID badges, ball pointers,
and clean, glitch-free motion trails for players and the ball.
"""

from collections import deque
import math
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np


class TacticalVisualizer:
    """Draws player foot ellipses, ID tags, ball markers, and motion trails."""

    # High-contrast Team Colors (BGR format):
    # Team 0: Vibrant Red
    # Team 1: Vibrant Royal Blue
    # Referee: Vibrant Yellow
    TEAM_COLORS = {
        0: (35, 45, 235),     # Team 0: Red
        1: (235, 130, 25),    # Team 1: Royal Blue
        -1: (20, 220, 245)    # Referee / Official: Yellow
    }

    @staticmethod
    def draw_player_ellipse(
        frame: np.ndarray,
        bbox: List[float],
        color: Tuple[int, int, int],
        tracker_id: Optional[int] = None
    ):
        """Draws a tactical ellipse under the player's feet with a shadow."""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2.0)
        bottom_y = int(y2)
        width = int((x2 - x1) * 0.8)
        axes = (max(5, width // 2), max(3, width // 4))

        # Semi-transparent ellipse overlay
        overlay = frame.copy()
        cv2.ellipse(overlay, (center_x, bottom_y), axes, 0, 0, 360, color, -1, cv2.LINE_AA)
        cv2.ellipse(overlay, (center_x, bottom_y), axes, 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Draw tracker ID badge if available
        if tracker_id is not None:
            label = f"#{tracker_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            label_x = center_x - text_w // 2
            label_y = bottom_y + text_h + 8
            
            # Dark badge background
            cv2.rectangle(
                frame,
                (label_x - 3, label_y - text_h - 2),
                (label_x + text_w + 3, label_y + baseline),
                (15, 15, 15),
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
    def draw_player_trails(
        frame: np.ndarray,
        player_trails: Dict[int, deque],
        player_teams: Dict[int, int],
        active_ids: Set[int],
        max_jump_px: float = 70.0
    ):
        """
        Draws dynamic movement trails ONLY for currently active players.
        Suppresses false jump lines (outliers) and dead ghost trails.
        """
        for p_id in active_ids:
            trail = player_trails.get(p_id)
            if not trail or len(trail) < 2:
                continue

            team_id = player_teams.get(p_id, 0)
            color = TacticalVisualizer.TEAM_COLORS.get(team_id, (200, 200, 200))
            pts = list(trail)

            for i in range(1, len(pts)):
                pt1 = pts[i - 1]
                pt2 = pts[i]

                # Outlier check: Do not connect jumps larger than max_jump_px
                dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                if dist > max_jump_px:
                    continue

                # Comet thickness effect (thinner at tail, thicker near feet)
                thickness = int(np.clip(2.5 * (i / len(pts)) + 1, 1, 3))
                p1 = (int(pt1[0]), int(pt1[1]))
                p2 = (int(pt2[0]), int(pt2[1]))

                # Subtle dark outline for high contrast
                cv2.line(frame, p1, p2, (15, 15, 15), thickness + 2, cv2.LINE_AA)
                cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_ball_trail(
        frame: np.ndarray,
        ball_trail: deque,
        max_jump_px: float = 140.0
    ):
        """
        Draws dynamic movement trail for the ball.
        Line color is pure WHITE with black shadow border and outlier jump suppression.
        """
        if len(ball_trail) < 2:
            return

        pts = list(ball_trail)
        for i in range(1, len(pts)):
            pt1 = pts[i - 1]
            pt2 = pts[i]

            # Outlier check: Do not connect teleporting ball jumps
            dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
            if dist > max_jump_px:
                continue

            thickness = int(np.clip(3.0 * (i / len(pts)) + 1, 2, 4))
            p1 = (int(pt1[0]), int(pt1[1]))
            p2 = (int(pt2[0]), int(pt2[1]))

            # Black contrast outline
            cv2.line(frame, p1, p2, (10, 10, 10), thickness + 2, cv2.LINE_AA)
            # Pure white ball trajectory line
            cv2.line(frame, p1, p2, (255, 255, 255), thickness, cv2.LINE_AA)

    @staticmethod
    def draw_average_positions(
        background_frame: np.ndarray,
        avg_positions: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, int]
    ) -> np.ndarray:
        """Draws average player positions on the camera background."""
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
