"""
Visualization and drawing utilities for tactical pitch analysis.
Includes persistent tactical ellipses, goalkeeper halos, ID badges,
ball ring + pointer, and smooth camera-motion-compensated trails
for players and the ball (pure WHITE ball trajectory trail).
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
    def warp_trail_points(pts: deque, warp_matrix: Optional[np.ndarray]) -> None:
        """
        Warps existing trail points in place by the frame-to-frame 2x3 GMC affine
        matrix so trails stay pinned to the pitch grass during camera pans.
        """
        if warp_matrix is None or len(pts) == 0:
            return
        R = warp_matrix[:, :2]
        t = warp_matrix[:, 2]
        arr = np.asarray(pts, dtype=np.float64)
        warped = (arr @ R.T) + t
        pts.clear()
        for x, y in warped:
            pts.append((float(x), float(y)))

    @staticmethod
    def draw_player_ellipse(
        frame: np.ndarray,
        bbox: List[float],
        color: Tuple[int, int, int],
        tracker_id: Optional[int] = None,
        is_goalkeeper: bool = False
    ):
        """Draws a tactical ellipse under the player's feet with ID badge."""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2.0)
        bottom_y = int(y2)
        width = int((x2 - x1) * 0.85)
        axes = (max(6, width // 2), max(3, width // 4))

        # Semi-transparent ellipse overlay
        overlay = frame.copy()
        cv2.ellipse(overlay, (center_x, bottom_y), axes, 0, 0, 360, color, -1, cv2.LINE_AA)
        border_col = (30, 220, 255) if is_goalkeeper else (0, 0, 0)
        cv2.ellipse(overlay, (center_x, bottom_y), axes, 0, 0, 360, border_col, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Draw tracker ID badge if available
        if tracker_id is not None:
            label = f"#{tracker_id} GK" if is_goalkeeper else f"#{tracker_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.44
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            label_x = center_x - text_w // 2
            label_y = bottom_y + text_h + 7

            # Dark badge background
            cv2.rectangle(
                frame,
                (label_x - 3, label_y - text_h - 2),
                (label_x + text_w + 3, label_y + baseline),
                (15, 15, 15),
                -1
            )
            text_col = (80, 235, 255) if is_goalkeeper else (255, 255, 255)
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                font,
                font_scale,
                text_col,
                thickness,
                cv2.LINE_AA
            )

    @staticmethod
    def draw_ball_pointer(
        frame: np.ndarray,
        ball_center: Tuple[float, float],
        radius: int = 8,
        interpolated: bool = False
    ):
        """
        Draws a crisp white highlight ring around the tracked football and
        a downward-pointing tactical marker triangle above it.
        """
        cx, cy = int(round(ball_center[0])), int(round(ball_center[1]))
        r = max(6, min(14, int(radius)))

        # Crisp dark-outlined white ring around the ball
        cv2.circle(frame, (cx, cy), r + 2, (15, 15, 15), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2, cv2.LINE_AA)

        # Downward pointer triangle above the ball
        top_y = cy - r - 5
        triangle_pts = np.array([
            [cx, top_y],
            [cx - 7, top_y - 12],
            [cx + 7, top_y - 12]
        ], np.int32)

        fill_col = (225, 225, 225) if interpolated else (255, 255, 255)
        cv2.drawContours(frame, [triangle_pts], 0, fill_col, -1, cv2.LINE_AA)
        cv2.drawContours(frame, [triangle_pts], 0, (15, 15, 15), 2, cv2.LINE_AA)

    @staticmethod
    def draw_player_trails(
        frame: np.ndarray,
        player_trails: Dict[int, deque],
        player_teams: Dict[int, int],
        active_ids: Set[int],
        max_jump_px: float = 85.0
    ):
        """
        Draws smooth tapering movement trails ONLY for currently active players.
        Suppresses false jump lines (outliers) and dead ghost trails.
        """
        for p_id in active_ids:
            trail = player_trails.get(p_id)
            if not trail or len(trail) < 2:
                continue

            team_id = player_teams.get(p_id, 0)
            color = TacticalVisualizer.TEAM_COLORS.get(team_id, (200, 200, 200))
            pts = list(trail)
            n = len(pts)

            for i in range(1, n):
                pt1 = pts[i - 1]
                pt2 = pts[i]

                dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                if dist > max_jump_px:
                    continue

                ratio = i / n
                thickness = int(np.clip(2.6 * ratio + 1, 1, 3))
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))

                cv2.line(frame, p1, p2, (15, 15, 15), thickness + 2, cv2.LINE_AA)
                cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_ball_trail(
        frame: np.ndarray,
        ball_trail: deque,
        max_jump_px: float = 190.0
    ):
        """
        Draws a smooth, high-visibility pure WHITE comet trajectory trail
        behind the football with a subtle dark border for contrast against both
        green turf and white pitch lines.
        """
        if len(ball_trail) < 2:
            return

        pts = list(ball_trail)
        n = len(pts)

        # Render subtle tapered white glow on overlay for broadcast look
        overlay = frame.copy()
        for i in range(1, n):
            pt1 = pts[i - 1]
            pt2 = pts[i]
            dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
            if dist > max_jump_px:
                continue

            ratio = i / n
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            p2 = (int(round(pt2[0])), int(round(pt2[1])))
            glow_w = int(np.clip(5.0 * ratio + 2, 2, 7))
            cv2.line(overlay, p1, p2, (255, 255, 255), glow_w, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

        # Crisp core white trajectory line with dark contrast edge
        for i in range(1, n):
            pt1 = pts[i - 1]
            pt2 = pts[i]
            dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
            if dist > max_jump_px:
                continue

            ratio = i / n
            thickness = int(np.clip(3.2 * ratio + 1.2, 1, 4))
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            p2 = (int(round(pt2[0])), int(round(pt2[1])))

            cv2.line(frame, p1, p2, (20, 20, 20), thickness + 2, cv2.LINE_AA)
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

            cv2.circle(canvas, center, 14, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, center, 12, color, -1, cv2.LINE_AA)

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
