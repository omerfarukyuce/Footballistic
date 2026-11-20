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

    # Standard football pitch dimensions in meters (FIFA / UEFA standard)
    PITCH_LENGTH_M = 105.0
    PITCH_WIDTH_M = 68.0

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

        # Pixel to meter scales
        self.scale_x_m = self.PITCH_LENGTH_M / max(1.0, float(self.field_w))
        self.scale_y_m = self.PITCH_WIDTH_M / max(1.0, float(self.field_h))

        # Base 2D pitch template
        self.base_pitch = self._create_base_pitch()

    def _create_base_pitch(self) -> np.ndarray:
        """Draws a clean, professional flat 2D soccer pitch template."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Field base grass color (tactical emerald green)
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
        cam_h: int = 1080,
        pitch_y_bounds: Optional[Tuple[int, int]] = None
    ) -> Tuple[float, float]:
        """
        Maps perspective camera coordinates to the flat 2D top-down rectangular pitch.
        Uses a projective foreshortening curve in Y (because the far half of a broadcast
        pitch compresses into ~30-35% of vertical image pixels) plus trapezoid width
        compensation in X.
        """
        if pitch_y_bounds is not None and (pitch_y_bounds[1] - pitch_y_bounds[0]) > cam_h * 0.25:
            y_top = float(pitch_y_bounds[0])
            y_bottom = float(min(pitch_y_bounds[1], cam_h * 0.96))
        else:
            y_top = cam_h * 0.27
            y_bottom = cam_h * 0.95

        u = float(np.clip((cam_y - y_top) / max(1.0, (y_bottom - y_top)), 0.0, 1.0))

        # Projective foreshortening warp: maps center-of-pitch (~28% down in pixels) to 0.50
        persp_k = 1.75
        y_norm = float(np.clip(((1.0 + persp_k) * u) / (1.0 + persp_k * u), 0.0, 1.0))

        # Perspective trapezoid width expansion from far touchline to near touchline
        w_top = cam_w * 0.86
        w_bottom = cam_w * 1.18
        current_w = w_top + y_norm * (w_bottom - w_top)
        center_x = cam_w * 0.50

        x_norm = 0.5 + (cam_x - center_x) / max(1.0, current_w)
        x_norm = float(np.clip(x_norm, 0.0, 1.0))

        pitch_2d_x = self.field_x1 + x_norm * self.field_w
        pitch_2d_y = self.field_y1 + y_norm * self.field_h
        return float(pitch_2d_x), float(pitch_2d_y)

    def pitch_to_meters(self, pitch_x: float, pitch_y: float) -> Tuple[float, float]:
        """Converts 2D pitch pixel coordinates to meters [0..105m, 0..68m]."""
        x_m = np.clip((pitch_x - self.field_x1) * self.scale_x_m, 0.0, self.PITCH_LENGTH_M)
        y_m = np.clip((pitch_y - self.field_y1) * self.scale_y_m, 0.0, self.PITCH_WIDTH_M)
        return float(round(x_m, 1)), float(round(y_m, 1))

    def generate_2d_heatmap(
        self,
        positions_2d: List[Tuple[float, float]],
        sigma: float = 26.0
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

        # Heatmap colormap (JET)
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

    def draw_2d_player_focus(
        self,
        tracker_id: int,
        avg_pos_2d: Tuple[float, float],
        team_id: int,
        trail_2d: Optional[List[Tuple[float, float]]] = None,
        teammate_positions: Optional[Dict[int, Tuple[float, float]]] = None
    ) -> np.ndarray:
        """
        Draws 2D Top-Down Pitch exclusively focusing on ONE player.
        Includes glowing tactical indicator, movement trajectory, and teammate context.
        """
        canvas = self.base_pitch.copy()
        primary_color = (35, 45, 235) if team_id == 0 else (235, 130, 25)

        # 1. Draw subtle teammate positions for spatial context
        if teammate_positions:
            for tid, (tx, ty) in teammate_positions.items():
                if tid == tracker_id:
                    continue
                c_pt = (int(round(tx)), int(round(ty)))
                # Faint teammate circle
                cv2.circle(canvas, c_pt, 7, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(canvas, c_pt, 6, (140, 140, 140), -1, cv2.LINE_AA)

        # 2. Draw movement trajectory if provided
        if trail_2d and len(trail_2d) >= 2:
            pts = np.array(trail_2d, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=(15, 15, 15), thickness=5, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [pts], isClosed=False, color=primary_color, thickness=3, lineType=cv2.LINE_AA)

            # Start marker
            start_pt = (int(round(trail_2d[0][0])), int(round(trail_2d[0][1])))
            cv2.circle(canvas, start_pt, 6, (0, 220, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, start_pt, 7, (0, 0, 0), 1, cv2.LINE_AA)

            # End marker
            end_pt = (int(round(trail_2d[-1][0])), int(round(trail_2d[-1][1])))
            cv2.circle(canvas, end_pt, 6, (0, 0, 235), -1, cv2.LINE_AA)
            cv2.circle(canvas, end_pt, 7, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Draw focused player average position with glowing halo
        cx, cy = int(round(avg_pos_2d[0])), int(round(avg_pos_2d[1]))
        
        # Outer pulsating radar rings
        cv2.circle(canvas, (cx, cy), 26, primary_color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 20, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 15, (10, 10, 10), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 13, primary_color, -1, cv2.LINE_AA)

        # Player ID label inside circle
        is_gk = tracker_id in (1, 13) or str(tracker_id) in ("1", "13")
        label = "GK" if is_gk else str(tracker_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.40 if is_gk else 0.45
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.putText(
            canvas,
            label,
            (cx - tw // 2, cy + th // 2),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Badge tag below player with meter coordinates
        xm, ym = self.pitch_to_meters(avg_pos_2d[0], avg_pos_2d[1])
        tag_text = f"#{tracker_id} GK ({xm}m, {ym}m)" if is_gk else f"#{tracker_id} ({xm}m, {ym}m)"
        tag_font_scale = 0.40
        (tag_w, tag_h), _ = cv2.getTextSize(tag_text, font, tag_font_scale, 1)
        tag_x = max(10, min(self.width - tag_w - 15, cx - tag_w // 2))
        tag_y = cy + 34
        
        cv2.rectangle(
            canvas,
            (tag_x - 4, tag_y - tag_h - 4),
            (tag_x + tag_w + 4, tag_y + 4),
            (20, 20, 20),
            -1
        )
        cv2.rectangle(
            canvas,
            (tag_x - 4, tag_y - tag_h - 4),
            (tag_x + tag_w + 4, tag_y + 4),
            primary_color,
            1
        )
        cv2.putText(
            canvas,
            tag_text,
            (tag_x, tag_y),
            font,
            tag_font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return canvas

    def draw_2d_team_formation(
        self,
        team_id: int,
        avg_positions_2d: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, int]
    ) -> np.ndarray:
        """
        Draws 2D top-down pitch displaying only ONE team's full tactical shape,
        formation points, and team centroid (average center of mass).
        """
        canvas = self.base_pitch.copy()
        color = (35, 45, 235) if team_id == 0 else (235, 130, 25)

        team_pts = []
        team_pids = []
        for p_id, (px, py) in avg_positions_2d.items():
            if player_teams.get(p_id) == team_id:
                team_pts.append((px, py))
                team_pids.append(p_id)

        if not team_pts:
            return canvas

        pts_arr = np.array(team_pts, dtype=np.float32)

        # 1. Draw team convex hull / compactness polygon
        if len(team_pts) >= 3:
            hull = cv2.convexHull(np.int32(pts_arr))
            overlay = canvas.copy()
            cv2.drawContours(overlay, [hull], 0, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
            cv2.drawContours(canvas, [hull], 0, color, 2, cv2.LINE_AA)

        # 2. Draw team centroid (average position)
        centroid_x = float(np.mean(pts_arr[:, 0]))
        centroid_y = float(np.mean(pts_arr[:, 1]))
        cc_pt = (int(round(centroid_x)), int(round(centroid_y)))

        # Tactical centroid crosshair & star
        cv2.circle(canvas, cc_pt, 12, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (cc_pt[0] - 16, cc_pt[1]), (cc_pt[0] + 16, cc_pt[1]), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (cc_pt[0], cc_pt[1] - 16), (cc_pt[0], cc_pt[1] + 16), (255, 255, 255), 2, cv2.LINE_AA)

        cx_m, cy_m = self.pitch_to_meters(centroid_x, centroid_y)
        centroid_label = f"Team Centroid ({cx_m}m, {cy_m}m)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (clw, clh), _ = cv2.getTextSize(centroid_label, font, 0.42, 1)
        cl_x = max(10, min(self.width - clw - 15, cc_pt[0] - clw // 2))
        cl_y = max(clh + 10, cc_pt[1] - 20)
        cv2.rectangle(canvas, (cl_x - 3, cl_y - clh - 3), (cl_x + clw + 3, cl_y + 3), (15, 15, 15), -1)
        cv2.putText(canvas, centroid_label, (cl_x, cl_y), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw each player
        for p_id, (px, py) in zip(team_pids, team_pts):
            center = (int(round(px)), int(round(py)))
            is_gk = p_id in (1, 13) or str(p_id) in ("1", "13")
            if is_gk:
                # Golden keeper halo
                cv2.circle(canvas, center, 18, (30, 215, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, center, 15, (15, 15, 15), 2, cv2.LINE_AA)
            cv2.circle(canvas, center, 13, color, -1, cv2.LINE_AA)

            label = "GK" if is_gk else str(p_id)
            font_s = 0.38 if is_gk else 0.42
            (tw, th), _ = cv2.getTextSize(label, font, font_s, 1)
            cv2.putText(
                canvas,
                label,
                (center[0] - tw // 2, center[1] + th // 2),
                font,
                font_s,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return canvas

    def draw_2d_formations(
        self,
        avg_positions_2d: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, int]
    ) -> np.ndarray:
        """
        Draws tactical formation circles for all players from both teams on the flat 2D pitch.
        """
        canvas = self.base_pitch.copy()

        team_colors_bgr = {
            0: (35, 45, 235),     # Team 0: Red
            1: (235, 130, 25),    # Team 1: Royal Blue
            -1: (20, 220, 245)    # Referee: Yellow
        }

        font = cv2.FONT_HERSHEY_SIMPLEX
        for p_id, (px, py) in avg_positions_2d.items():
            team_id = player_teams.get(p_id, 0)
            color = team_colors_bgr.get(team_id, (200, 200, 200))
            center = (int(round(px)), int(round(py)))
            is_gk = p_id in (1, 13) or str(p_id) in ("1", "13")

            if is_gk:
                # Golden keeper halo
                cv2.circle(canvas, center, 18, (30, 215, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, center, 15, (15, 15, 15), 2, cv2.LINE_AA)
            cv2.circle(canvas, center, 13, color, -1, cv2.LINE_AA)

            label = "GK" if is_gk else str(p_id)
            font_s = 0.38 if is_gk else 0.42
            (tw, th), _ = cv2.getTextSize(label, font, font_s, 1)
            cv2.putText(
                canvas,
                label,
                (center[0] - tw // 2, center[1] + th // 2),
                font,
                font_s,
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
