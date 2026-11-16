"""
2D Top-Down (Bird's Eye View) Football Pitch Mapping & Spatial Analytics.
Renders flat rectangular tactical pitch diagrams, heatmaps, and average formations.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class PitchView2D:
    """Renders 2D top-down tactical soccer pitch, heatmaps, and formations."""

    def __init__(
        self,
        pitch_width: int = 1050,
        pitch_height: int = 680,
        padding: int = 40
    ):
        self.width = pitch_width
        self.height = pitch_height
        self.padding = padding

        # Pitch playable dimensions inside padding
        self.field_x1 = padding
        self.field_y1 = padding
        self.field_x2 = pitch_width - padding
        self.field_y2 = pitch_height - padding
        self.field_w = self.field_x2 - self.field_x1
        self.field_h = self.field_y2 - self.field_y1

        # Base 2D pitch template
        self.base_pitch = self._create_base_pitch()

    def _create_base_pitch(self) -> np.ndarray:
        """Draws a clean, professional flat 2D soccer pitch template."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Field base green
        img[:] = (35, 125, 45)

        # Alternating mowing stripes (10 stripes along length)
        num_stripes = 10
        stripe_w = self.field_w / num_stripes
        for i in range(num_stripes):
            if i % 2 == 1:
                sx1 = int(self.field_x1 + i * stripe_w)
                sx2 = int(self.field_x1 + (i + 1) * stripe_w)
                img[self.field_y1:self.field_y2, sx1:sx2] = (30, 115, 40)

        line_color = (245, 245, 245)
        thickness = 2

        # 1. Outer boundary
        cv2.rectangle(img, (self.field_x1, self.field_y1), (self.field_x2, self.field_y2), line_color, thickness, cv2.LINE_AA)

        # 2. Halfway line & center spot
        mid_x = (self.field_x1 + self.field_x2) // 2
        mid_y = (self.field_y1 + self.field_y2) // 2
        cv2.line(img, (mid_x, self.field_y1), (mid_x, self.field_y2), line_color, thickness, cv2.LINE_AA)
        cv2.circle(img, (mid_x, mid_y), 4, line_color, -1, cv2.LINE_AA)

        # 3. Center circle (radius ~ 9.15m scaled)
        center_radius = int(self.field_w * (9.15 / 105.0))
        cv2.circle(img, (mid_x, mid_y), center_radius, line_color, thickness, cv2.LINE_AA)

        # 4. Penalty areas (16.5m length, 40.3m height)
        pen_len = int(self.field_w * (16.5 / 105.0))
        pen_h = int(self.field_h * (40.3 / 68.0))
        pen_y1 = mid_y - pen_h // 2
        pen_y2 = mid_y + pen_h // 2

        # Left Penalty Box
        cv2.rectangle(img, (self.field_x1, pen_y1), (self.field_x1 + pen_len, pen_y2), line_color, thickness, cv2.LINE_AA)
        # Right Penalty Box
        cv2.rectangle(img, (self.field_x2 - pen_len, pen_y1), (self.field_x2, pen_y2), line_color, thickness, cv2.LINE_AA)

        # 5. Goal areas (5.5m length, 18.3m height)
        goal_len = int(self.field_w * (5.5 / 105.0))
        goal_h = int(self.field_h * (18.3 / 68.0))
        goal_y1 = mid_y - goal_h // 2
        goal_y2 = mid_y + goal_h // 2

        cv2.rectangle(img, (self.field_x1, goal_y1), (self.field_x1 + goal_len, goal_y2), line_color, thickness, cv2.LINE_AA)
        cv2.rectangle(img, (self.field_x2 - goal_len, goal_y1), (self.field_x2, goal_y2), line_color, thickness, cv2.LINE_AA)

        # 6. Penalty spots (11m)
        pen_spot_dist = int(self.field_w * (11.0 / 105.0))
        cv2.circle(img, (self.field_x1 + pen_spot_dist, mid_y), 3, line_color, -1, cv2.LINE_AA)
        cv2.circle(img, (self.field_x2 - pen_spot_dist, mid_y), 3, line_color, -1, cv2.LINE_AA)

        # 7. Penalty arcs
        cv2.ellipse(img, (self.field_x1 + pen_spot_dist, mid_y), (center_radius, center_radius), 0, -53, 53, line_color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (self.field_x2 - pen_spot_dist, mid_y), (center_radius, center_radius), 0, 127, 233, line_color, thickness, cv2.LINE_AA)

        # 8. Corner arcs (1m)
        corner_r = int(self.field_w * (1.5 / 105.0))
        cv2.ellipse(img, (self.field_x1, self.field_y1), (corner_r, corner_r), 0, 0, 90, line_color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (self.field_x1, self.field_y2), (corner_r, corner_r), 0, 270, 360, line_color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (self.field_x2, self.field_y1), (corner_r, corner_r), 0, 90, 180, line_color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (self.field_x2, self.field_y2), (corner_r, corner_r), 0, 180, 270, line_color, thickness, cv2.LINE_AA)

        return img

    def map_camera_to_2d(
        self,
        cam_x: float,
        cam_y: float,
        cam_w: int = 1920,
        cam_h: int = 1080
    ) -> Tuple[float, float]:
        """
        Maps perspective camera coordinates to the flat 2D top-down rectangular pitch.
        Compensates for perspective trapezoid compression (top of field is narrower in pixels).
        """
        # Tactical camera bounding bounds in pixel space (Bundesliga / DFL wide cam)
        y_top = cam_h * 0.28
        y_bottom = cam_h * 0.95
        
        y_norm = np.clip((cam_y - y_top) / max(1.0, (y_bottom - y_top)), 0.0, 1.0)

        # Perspective trapezoid width expansion from top to bottom
        w_top = cam_w * 0.88
        w_bottom = cam_w * 1.15
        current_w = w_top + y_norm * (w_bottom - w_top)
        center_x = cam_w * 0.50

        x_norm = 0.5 + (cam_x - center_x) / max(1.0, current_w)
        x_norm = np.clip(x_norm, 0.0, 1.0)

        pitch_2d_x = self.field_x1 + x_norm * self.field_w
        pitch_2d_y = self.field_y1 + y_norm * self.field_h
        return float(pitch_2d_x), float(pitch_2d_y)

    def generate_2d_heatmap(
        self,
        positions_2d: List[Tuple[float, float]],
        sigma: float = 28.0
    ) -> np.ndarray:
        """
        Computes 2D Gaussian density heatmap on the flat rectangular pitch.
        Blends seamlessly with the pitch lines.
        """
        pitch_canvas = self.base_pitch.copy()
        if len(positions_2d) < 3:
            return pitch_canvas

        density = np.zeros((self.height, self.width), dtype=np.float32)
        for x, y in positions_2d:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < self.width and 0 <= iy < self.height:
                density[iy, ix] += 1.0

        density = gaussian_filter(density, sigma=sigma)
        max_val = np.max(density)
        if max_val > 0:
            norm_density = (density / max_val * 255).astype(np.uint8)
        else:
            return pitch_canvas

        # Heatmap colormap
        heatmap_color = cv2.applyColorMap(norm_density, cv2.COLORMAP_JET)

        # Blend non-trivial density onto pitch
        mask = norm_density > 15
        if np.any(mask):
            pitch_canvas[mask] = cv2.addWeighted(
                pitch_canvas[mask], 0.35,
                heatmap_color[mask], 0.65,
                0
            )

        return pitch_canvas

    def draw_2d_formations(
        self,
        avg_positions_2d: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, int]
    ) -> np.ndarray:
        """
        Draws tactical formation circles for each player on the flat 2D pitch.
        Red = Team 0, Royal Blue = Team 1.
        """
        canvas = self.base_pitch.copy()

        team_colors_bgr = {
            0: (35, 45, 235),     # Team 0: Red
            1: (235, 130, 25),    # Team 1: Royal Blue
            -1: (20, 220, 245)    # Referee: Yellow
        }

        for p_id, (px, py) in avg_positions_2d.items():
            team_id = player_teams.get(p_id, 0)
            color = team_colors_bgr.get(team_id, (200, 200, 200))
            center = (int(round(px)), int(round(py)))

            # Outer ring & filled circle
            cv2.circle(canvas, center, 15, (15, 15, 15), 2, cv2.LINE_AA)
            cv2.circle(canvas, center, 13, color, -1, cv2.LINE_AA)

            # Player ID label
            label = str(p_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
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

    def draw_2d_player_trajectory(
        self,
        trail_2d: List[Tuple[float, float]],
        team_id: int
    ) -> np.ndarray:
        """
        Draws the movement trail of a specific player on the flat 2D pitch.
        """
        canvas = self.base_pitch.copy()
        if len(trail_2d) < 2:
            return canvas

        color = (35, 45, 235) if team_id == 0 else (235, 130, 25)

        pts = np.array(trail_2d, np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=(10, 10, 10), thickness=5, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)

        # Start marker (Green)
        start_pt = (int(trail_2d[0][0]), int(trail_2d[0][1]))
        cv2.circle(canvas, start_pt, 7, (0, 230, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, start_pt, 8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Start", (start_pt[0] + 8, start_pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # End marker (Red)
        end_pt = (int(trail_2d[-1][0]), int(trail_2d[-1][1]))
        cv2.circle(canvas, end_pt, 7, (0, 0, 240), -1, cv2.LINE_AA)
        cv2.circle(canvas, end_pt, 8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "End", (end_pt[0] + 8, end_pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        return canvas
