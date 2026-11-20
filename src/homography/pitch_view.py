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

    def draw_2d_passing_network(
        self,
        team_id: int,
        network_data: Dict,
        goalkeeper_ids: Optional[set] = None
    ) -> np.ndarray:
        """
        Draws a 2D Bird's-Eye Passing Network for `team_id`:
          - Edges: weighted passing lanes between teammates (thickness scaled by pass count,
            highlighted in gold/cyan when containing progressive passes, with directional arrowheads).
          - Nodes: player average positions scaled by ball touches / involvement.
        """
        canvas = self.base_pitch.copy()
        primary_color = (35, 45, 235) if team_id == 0 else (235, 130, 25)
        edge_base_color = (120, 145, 255) if team_id == 0 else (255, 195, 90)
        prog_edge_color = (40, 225, 255)  # Gold/yellow highlight for progressive links

        nodes = network_data.get("nodes", {})
        edges = network_data.get("edges", [])
        if not nodes:
            return canvas

        gk_set = goalkeeper_ids or {1, 13}

        # 1. Draw subtle convex hull background of team shape
        pts_arr = np.array([n["pos_2d"] for n in nodes.values()], dtype=np.float32)
        if len(pts_arr) >= 3:
            hull = cv2.convexHull(np.int32(pts_arr))
            overlay = canvas.copy()
            cv2.drawContours(overlay, [hull], 0, primary_color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)

        # 2. Draw Passing Network Edges (passing lanes)
        link_overlay = canvas.copy()
        for ed in edges:
            u, v = ed["pair"]
            if u not in nodes or v not in nodes:
                continue
            p1 = (int(round(nodes[u]["pos_2d"][0])), int(round(nodes[u]["pos_2d"][1])))
            p2 = (int(round(nodes[v]["pos_2d"][0])), int(round(nodes[v]["pos_2d"][1])))
            cnt = ed["count"]
            is_prog = ed.get("progressive", 0) > 0

            thickness = int(np.clip(2 + cnt * 1.6, 2, 10))
            col = prog_edge_color if is_prog else edge_base_color

            # Dark contrast casing + glowing link
            cv2.line(link_overlay, p1, p2, (15, 15, 15), thickness + 3, cv2.LINE_AA)
            cv2.line(link_overlay, p1, p2, col, thickness, cv2.LINE_AA)

            # Directional arrowhead toward dominant receiver along the link
            src, dst = (p1, p2) if ed["a_to_b"] >= ed["b_to_a"] else (p2, p1)
            mx = int(round(src[0] + 0.62 * (dst[0] - src[0])))
            my = int(round(src[1] + 0.62 * (dst[1] - src[1])))
            ang = np.arctan2(dst[1] - src[1], dst[0] - src[0])
            ah = 7 + min(5, cnt)
            a1 = (int(round(mx - ah * np.cos(ang - 0.55))), int(round(my - ah * np.sin(ang - 0.55))))
            a2 = (int(round(mx - ah * np.cos(ang + 0.55))), int(round(my - ah * np.sin(ang + 0.55))))
            tri = np.array([[mx, my], a1, a2], dtype=np.int32)
            cv2.drawContours(link_overlay, [tri], 0, (15, 15, 15), 2, cv2.LINE_AA)
            cv2.drawContours(link_overlay, [tri], 0, col, -1, cv2.LINE_AA)

            # Pass count pill on multi-pass links
            if cnt >= 2:
                cx_m = (p1[0] + p2[0]) // 2
                cy_m = (p1[1] + p2[1]) // 2
                cv2.circle(link_overlay, (cx_m, cy_m), 9, (18, 22, 30), -1, cv2.LINE_AA)
                cv2.circle(link_overlay, (cx_m, cy_m), 9, col, 1, cv2.LINE_AA)
                lbl = str(cnt)
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                cv2.putText(
                    link_overlay, lbl, (cx_m - lw // 2, cy_m + lh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA
                )

        cv2.addWeighted(link_overlay, 0.85, canvas, 0.15, 0, canvas)

        # 3. Draw Player Nodes (radius scaled by touch involvement)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for pid, info in nodes.items():
            center = (int(round(info["pos_2d"][0])), int(round(info["pos_2d"][1])))
            touches = info.get("touches", 1)
            passes_made = info.get("passes_made", 0)
            radius = int(np.clip(12 + touches * 1.1 + passes_made * 0.8, 13, 22))
            is_gk = (pid in gk_set) or (pid in (1, 13))

            if is_gk:
                cv2.circle(canvas, center, radius + 4, (30, 215, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, center, radius + 2, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, radius, primary_color, -1, cv2.LINE_AA)
            if passes_made > 0:
                cv2.circle(canvas, center, radius, (245, 245, 245), 1, cv2.LINE_AA)

            label = "GK" if is_gk else str(pid)
            font_s = 0.38 if is_gk else 0.42
            (tw, th), _ = cv2.getTextSize(label, font, font_s, 1)
            cv2.putText(
                canvas, label, (center[0] - tw // 2, center[1] + th // 2),
                font, font_s, (255, 255, 255), 1, cv2.LINE_AA
            )

        return canvas

    def draw_2d_events_map(
        self,
        passes: List[Dict],
        interceptions: List[Dict],
        carries: Optional[List[Dict]] = None,
        shots: Optional[List[Dict]] = None,
        team_filter: Optional[int] = None
    ) -> np.ndarray:
        """
        Draws all match action vectors on the 2D pitch:
          - Completed Passes: directional arrows (Gold/Cyan for Progressive / Box Entry passes)
          - Progressive / Box Carries: dashed/accented movement vectors
          - Interceptions & Ball Recoveries: crisp diamond/ring markers at recovery point
          - Shots: bold white/gold goal-bound arrows
        """
        canvas = self.base_pitch.copy()
        overlay = canvas.copy()

        team_colors = {
            0: (45, 65, 245),     # Team 0: Red
            1: (245, 145, 35),    # Team 1: Blue
        }
        prog_color = (40, 230, 255)  # Bright Gold for progressive / key actions
        rec_color = (120, 255, 120)  # Bright Emerald for ball recoveries

        # 1. Draw Completed Pass Vectors
        for p in passes:
            tid = p["team_id"]
            if team_filter is not None and tid != team_filter:
                continue
            p1 = (int(round(p["start_2d"][0])), int(round(p["start_2d"][1])))
            p2 = (int(round(p["end_2d"][0])), int(round(p["end_2d"][1])))
            is_key = p.get("is_progressive") or p.get("is_box_entry") or p.get("is_final_third_entry")
            col = prog_color if is_key else team_colors.get(tid, (220, 220, 220))
            thick = 3 if is_key else 2

            cv2.arrowedLine(overlay, p1, p2, (15, 15, 15), thick + 2, cv2.LINE_AA, tipLength=0.14)
            cv2.arrowedLine(overlay, p1, p2, col, thick, cv2.LINE_AA, tipLength=0.14)
            cv2.circle(overlay, p1, 4, col, -1, cv2.LINE_AA)

        # 2. Draw Progressive / Box Carries
        for c in (carries or []):
            tid = c["team_id"]
            if team_filter is not None and tid != team_filter:
                continue
            p1 = (int(round(c["start_2d"][0])), int(round(c["start_2d"][1])))
            p2 = (int(round(c["end_2d"][0])), int(round(c["end_2d"][1])))
            cv2.arrowedLine(overlay, p1, p2, (20, 20, 20), 3, cv2.LINE_AA, tipLength=0.16)
            cv2.arrowedLine(overlay, p1, p2, (255, 220, 180), 2, cv2.LINE_AA, tipLength=0.16)

        # 3. Draw Shots
        for sh in (shots or []):
            tid = sh["team_id"]
            if team_filter is not None and tid != team_filter:
                continue
            p1 = (int(round(sh["start_2d"][0])), int(round(sh["start_2d"][1])))
            p2 = (int(round(sh["end_2d"][0])), int(round(sh["end_2d"][1])))
            cv2.arrowedLine(overlay, p1, p2, (10, 10, 10), 6, cv2.LINE_AA, tipLength=0.18)
            cv2.arrowedLine(overlay, p1, p2, (255, 255, 255), 4, cv2.LINE_AA, tipLength=0.18)

        # 4. Draw Interceptions & Ball Recoveries
        font = cv2.FONT_HERSHEY_SIMPLEX
        for r in interceptions:
            tid = r["team_id"]
            if team_filter is not None and tid != team_filter:
                continue
            pt = (int(round(r["end_2d"][0])), int(round(r["end_2d"][1])))
            t_col = team_colors.get(tid, rec_color)

            # Diamond recovery marker
            d_sz = 9
            diamond = np.array([
                [pt[0], pt[1] - d_sz],
                [pt[0] + d_sz, pt[1]],
                [pt[0], pt[1] + d_sz],
                [pt[0] - d_sz, pt[1]]
            ], dtype=np.int32)
            cv2.drawContours(overlay, [diamond], 0, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.drawContours(overlay, [diamond], 0, rec_color, -1, cv2.LINE_AA)
            cv2.circle(overlay, pt, 4, t_col, -1, cv2.LINE_AA)

            lbl = f"#{r['player_id']}"
            cv2.putText(overlay, lbl, (pt[0] + 10, pt[1] + 4), font, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        return canvas

    def draw_2d_single_pass(self, pass_ev: Dict) -> np.ndarray:
        """
        Renders a crystal-clear, self-explanatory 2D Bird's-Eye Freeze-Frame Visual
        dedicated to a SINGLE pass:
          - Sleek Broadcast Info Banner at top:
            Pass #, Timestamp, Team, From -> To, COMPLETED / INCOMPLETE badge,
            Pass Type, Length (m), Avg Speed (km/h), Flight Duration (s).
          - Full 2D pitch with ALL teammates and ALL opponents at the exact kick frame.
          - Highlighted Passer halo + Receiver/Interceptor halo + Receiver run dotted arrow.
          - Actual frame-by-frame curved ball flight path (`ball_path_2d`):
              * COMPLETED: solid glowing emerald/cyan arrow + floating length/speed tag.
              * INCOMPLETE: dashed red/orange arrow + bold 'X' interception marker.
          - Compact legend along the bottom border.
        """
        header_h = 84
        footer_h = 34
        total_h = header_h + self.height + footer_h
        full_img = np.zeros((total_h, self.width, 3), dtype=np.uint8)
        full_img[:] = (18, 22, 31)  # Deep slate-navy broadcast frame (BGR)

        pitch = self.base_pitch.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        team_colors = {
            0: (45, 55, 235),     # Team 0: Vibrant Red (BGR)
            1: (235, 135, 30),    # Team 1: Vibrant Royal Blue (BGR)
        }
        team_id = int(pass_ev["team_id"])
        passer_id = int(pass_ev["player_id"])
        target_id = int(pass_ev["target_id"])
        is_ok = bool(pass_ev["is_completed"])
        players_snap = pass_ev.get("players_at_kick", {})
        ball_path = pass_ev.get("ball_path_2d", [pass_ev["start_2d"], pass_ev["end_2d"]])
        recv_pos_final = pass_ev.get("receiver_pos_at_receive", pass_ev["end_2d"])

        # ------------------------------------------------------- 1. DRAW BALL TRAJECTORY ON PITCH
        pass_col = (110, 245, 85) if is_ok else (60, 75, 255)   # Green (Completed) / Red-Orange (Incomplete)
        glow_col = (160, 255, 140) if is_ok else (90, 120, 255)

        pts_list = [(int(round(p[0])), int(round(p[1]))) for p in ball_path]
        if len(pts_list) >= 2:
            # Subtle outer glow along the actual ball path
            glow_layer = pitch.copy()
            if is_ok:
                pts_arr = np.array(pts_list, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(glow_layer, [pts_arr], False, glow_col, 9, cv2.LINE_AA)
                cv2.addWeighted(glow_layer, 0.30, pitch, 0.70, 0, pitch)
                cv2.polylines(pitch, [pts_arr], False, (15, 15, 15), 6, cv2.LINE_AA)
                cv2.polylines(pitch, [pts_arr], False, pass_col, 4, cv2.LINE_AA)
            else:
                # Draw clean dashed trajectory for Incomplete / Intercepted pass
                seg_dist_acc = 0.0
                for idx_p in range(1, len(pts_list)):
                    p1, p2 = pts_list[idx_p - 1], pts_list[idx_p]
                    d_seg = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
                    if d_seg < 1.0:
                        continue
                    steps = max(1, int(d_seg / 6.0))
                    for s_i in range(steps):
                        t0 = s_i / steps
                        t1 = (s_i + 1) / steps
                        q1 = (int(round(p1[0] + t0 * (p2[0] - p1[0]))), int(round(p1[1] + t0 * (p2[1] - p1[1]))))
                        q2 = (int(round(p1[0] + t1 * (p2[0] - p1[0]))), int(round(p1[1] + t1 * (p2[1] - p1[1]))))
                        if int(seg_dist_acc // 14.0) % 2 == 0:
                            cv2.line(pitch, q1, q2, (15, 15, 15), 6, cv2.LINE_AA)
                            cv2.line(pitch, q1, q2, pass_col, 4, cv2.LINE_AA)
                        seg_dist_acc += d_seg / steps

            # Directional arrowhead pointing along final flight tangent
            end_pt = pts_list[-1]
            prev_pt = pts_list[-2]
            # Walk back a few points if last 2 points are nearly identical
            for back_i in range(len(pts_list) - 2, -1, -1):
                if np.hypot(end_pt[0] - pts_list[back_i][0], end_pt[1] - pts_list[back_i][1]) >= 12.0:
                    prev_pt = pts_list[back_i]
                    break
            ang = float(np.arctan2(end_pt[1] - prev_pt[1], end_pt[0] - prev_pt[0]))
            # Pull arrowhead tip slightly back from player center so it sits cleanly at circle edge
            tip_x = int(round(end_pt[0] - 18.0 * np.cos(ang)))
            tip_y = int(round(end_pt[1] - 18.0 * np.sin(ang)))
            ah = 15.0
            a1 = (int(round(tip_x - ah * np.cos(ang - 0.48))), int(round(tip_y - ah * np.sin(ang - 0.48))))
            a2 = (int(round(tip_x - ah * np.cos(ang + 0.48))), int(round(tip_y - ah * np.sin(ang + 0.48))))
            tri = np.array([[tip_x, tip_y], a1, a2], dtype=np.int32)
            cv2.drawContours(pitch, [tri], 0, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.drawContours(pitch, [tri], 0, pass_col, -1, cv2.LINE_AA)

            # White football dot at kick origin
            cv2.circle(pitch, pts_list[0], 6, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(pitch, pts_list[0], 4, (255, 255, 255), -1, cv2.LINE_AA)

            # Bold 'X' interception marker if incomplete
            if not is_ok:
                ix, iy = end_pt
                cv2.line(pitch, (ix - 11, iy - 11), (ix + 11, iy + 11), (15, 15, 15), 5, cv2.LINE_AA)
                cv2.line(pitch, (ix - 11, iy + 11), (ix + 11, iy - 11), (15, 15, 15), 5, cv2.LINE_AA)
                cv2.line(pitch, (ix - 10, iy - 10), (ix + 10, iy + 10), (70, 85, 255), 3, cv2.LINE_AA)
                cv2.line(pitch, (ix - 10, iy + 10), (ix + 10, iy - 10), (70, 85, 255), 3, cv2.LINE_AA)

        # ------------------------------------------------------- 2. DRAW RECEIVER/INTERCEPTOR RUN LINE
        if target_id in players_snap:
            t_start = (int(round(players_snap[target_id]["pos_2d"][0])), int(round(players_snap[target_id]["pos_2d"][1])))
            t_end = (int(round(recv_pos_final[0])), int(round(recv_pos_final[1])))
            if np.hypot(t_end[0] - t_start[0], t_end[1] - t_start[1]) >= 14.0:
                # Subtle ghost dot at kick frame + white dotted movement line to reception point
                cv2.circle(pitch, t_start, 8, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.arrowedLine(pitch, t_start, t_end, (235, 235, 235), 2, cv2.LINE_AA, tipLength=0.18)

        # ------------------------------------------------------- 3. DRAW ALL 22 PLAYERS (TEAMMATES & OPPONENTS)
        # Draw uninvolved players first, then Passer & Receiver/Interceptor on top!
        bg_layer = pitch.copy()
        occupied_boxes: List[Tuple[int, int, int, int]] = []
        for pid, pinfo in players_snap.items():
            if pid in (passer_id, target_id):
                continue
            pt = (int(round(pinfo["pos_2d"][0])), int(round(pinfo["pos_2d"][1])))
            occupied_boxes.append((pt[0] - 16, pt[1] - 16, pt[0] + 16, pt[1] + 16))
            p_col = team_colors.get(pinfo["team_id"], (180, 180, 180))
            is_gk = pinfo.get("is_gk", False)
            if is_gk:
                cv2.circle(bg_layer, pt, 15, (30, 215, 255), 2, cv2.LINE_AA)
            cv2.circle(bg_layer, pt, 13, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(bg_layer, pt, 11, p_col, -1, cv2.LINE_AA)
            lbl = "GK" if is_gk else str(pid)
            (tw, th), _ = cv2.getTextSize(lbl, font, 0.36, 1)
            cv2.putText(bg_layer, lbl, (pt[0] - tw // 2, pt[1] + th // 2), font, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(bg_layer, 0.80, pitch, 0.20, 0, pitch)

        # Highlight PASSER (at kick position) and RECEIVER/INTERCEPTOR (at reception position)
        featured = [
            (passer_id, team_id, pass_ev["start_2d"], (255, 255, 255), "PASSER"),
            (
                target_id,
                int(pass_ev.get("receiver_team_id", team_id)),
                recv_pos_final,
                (110, 245, 85) if is_ok else (70, 85, 255),
                "RECEIVER" if is_ok else "INTERCEPTOR"
            ),
        ]
        for pid, p_tid, pos_xy, halo_col, role_tag in featured:
            pt = (int(round(pos_xy[0])), int(round(pos_xy[1])))
            occupied_boxes.append((pt[0] - 26, pt[1] - 26, pt[0] + 26, pt[1] + 26))
            p_col = team_colors.get(p_tid, (200, 200, 200))
            cv2.circle(pitch, pt, 22, halo_col, 2, cv2.LINE_AA)
            cv2.circle(pitch, pt, 17, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(pitch, pt, 15, p_col, -1, cv2.LINE_AA)
            lbl = f"{pid}"
            (tw, th), _ = cv2.getTextSize(lbl, font, 0.46, 2)
            cv2.putText(pitch, lbl, (pt[0] - tw // 2, pt[1] + th // 2), font, 0.46, (255, 255, 255), 2, cv2.LINE_AA)

            # Role micro-badge below featured player (flip above if very close to bottom edge)
            (rw, rh), _ = cv2.getTextSize(role_tag, font, 0.34, 1)
            rx = int(np.clip(pt[0] - rw // 2, 12, self.width - rw - 12))
            if pt[1] + 40 < self.height - 8:
                ry = pt[1] + 34
            else:
                ry = max(56, pt[1] - 26)
            cv2.rectangle(pitch, (rx - 4, ry - rh - 3), (rx + rw + 4, ry + 3), (15, 18, 26), -1)
            cv2.rectangle(pitch, (rx - 4, ry - rh - 3), (rx + rw + 4, ry + 3), halo_col, 1)
            cv2.putText(pitch, role_tag, (rx, ry), font, 0.34, halo_col, 1, cv2.LINE_AA)
            occupied_boxes.append((rx - 6, ry - rh - 5, rx + rw + 6, ry + 5))

        # ------------------------------------------------------- 3b. FLOATING CALLOUT PILL (ON TOP, COLLISION-FREE)
        if len(pts_list) >= 2:
            mid_pt = pts_list[len(pts_list) // 2]
            pill_txt = f"{pass_ev['length_m']:.1f}m  |  {pass_ev['avg_speed_kmh']:.0f} km/h"
            (pw, ph), _ = cv2.getTextSize(pill_txt, font, 0.46, 1)
            path_ang = float(np.arctan2(pts_list[-1][1] - pts_list[0][1], pts_list[-1][0] - pts_list[0][0]))
            px_u, py_u = -np.sin(path_ang), np.cos(path_ang)
            al_u, al_v = np.cos(path_ang), np.sin(path_ang)

            # Candidate placements around midpoint (both sides, near & slightly wider)
            cand_offsets = [
                (px_u * 38.0, py_u * 38.0),
                (-px_u * 38.0, -py_u * 38.0),
                (px_u * 64.0, py_u * 64.0),
                (-px_u * 64.0, -py_u * 64.0),
                (px_u * 48.0 + al_u * 32.0, py_u * 48.0 + al_v * 32.0),
                (-px_u * 48.0 - al_u * 32.0, -py_u * 48.0 - al_v * 32.0),
                (92.0, 0.0),
                (-92.0, 0.0),
            ]

            best_box = None
            best_penalty = 1e12
            for ox, oy in cand_offsets:
                bx1 = int(np.clip(mid_pt[0] + ox - pw // 2, 50, self.width - pw - 56))
                by_base = int(np.clip(mid_pt[1] + oy + ph // 2, 62, self.height - 50))
                r_x1, r_y1, r_x2, r_y2 = bx1 - 8, by_base - ph - 7, bx1 + pw + 8, by_base + 7

                pen = 0.0
                # Penalize overlap with players & role micro-badges
                for ox1, oy1, ox2, oy2 in occupied_boxes:
                    iw = max(0, min(r_x2, ox2) - max(r_x1, ox1))
                    ih = max(0, min(r_y2, oy2) - max(r_y1, oy1))
                    pen += iw * ih * 5.0
                # Penalize covering the ball arrow itself
                for bp in pts_list[::max(1, len(pts_list) // 16)]:
                    if r_x1 - 4 <= bp[0] <= r_x2 + 4 and r_y1 - 4 <= bp[1] <= r_y2 + 4:
                        pen += 450.0
                # Mild distance tie-breaker so pill stays visually anchored to the pass
                pen += float(np.hypot((r_x1 + r_x2) * 0.5 - mid_pt[0], (r_y1 + r_y2) * 0.5 - mid_pt[1])) * 1.2
                if pen < best_penalty:
                    best_penalty = pen
                    best_box = (bx1, by_base)

            px_box, py_box = best_box
            cv2.rectangle(pitch, (px_box - 7, py_box - ph - 6), (px_box + pw + 7, py_box + 6), (16, 20, 28), -1)
            cv2.rectangle(pitch, (px_box - 7, py_box - ph - 6), (px_box + pw + 7, py_box + 6), pass_col, 2)
            cv2.putText(pitch, pill_txt, (px_box, py_box), font, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        # Place pitch into composite frame
        full_img[header_h:header_h + self.height, :] = pitch

        # ------------------------------------------------------- 4. DRAW BROADCAST TOP HUD BANNER
        cv2.line(full_img, (0, header_h - 1), (self.width, header_h - 1), (55, 65, 82), 2)

        # Left block: PASS #N + Team + From -> To
        t_col = team_colors.get(team_id, (200, 200, 200))
        cv2.rectangle(full_img, (18, 16), (26, 68), t_col, -1)

        title_l1 = f"PASS #{pass_ev['pass_number']}   |   Team {team_id} ({'Home' if team_id == 0 else 'Away'})   |   {pass_ev['time_sec']:.1f}s -> {pass_ev['end_time_sec']:.1f}s"
        cv2.putText(full_img, title_l1, (38, 36), font, 0.55, (245, 248, 255), 2, cv2.LINE_AA)

        if is_ok:
            title_l2 = f"Player #{passer_id}  --->  Player #{target_id}   |   {pass_ev['subtype']}"
        else:
            opp_tid = int(pass_ev.get("receiver_team_id", 1 - team_id))
            title_l2 = f"Player #{passer_id}  --X->  Intercepted by #{target_id} (Team {opp_tid})"

        # Guarantee title_l2 never collides with the right-hand outcome badge (x=635)
        l2_scale = 0.47
        while l2_scale > 0.36 and cv2.getTextSize(title_l2, font, l2_scale, 1)[0][0] > 580:
            l2_scale -= 0.02
        cv2.putText(full_img, title_l2, (38, 64), font, l2_scale, (185, 198, 218), 1, cv2.LINE_AA)

        # Right block: Outcome Badge + Length + Speed + Duration pills
        badge_txt = "COMPLETED" if is_ok else "INCOMPLETE"
        badge_bg = (35, 130, 50) if is_ok else (45, 45, 195)
        badge_bd = (110, 245, 85) if is_ok else (85, 100, 255)
        cv2.rectangle(full_img, (635, 18), (765, 66), badge_bg, -1)
        cv2.rectangle(full_img, (635, 18), (765, 66), badge_bd, 2)
        (bw_t, bh_t), _ = cv2.getTextSize(badge_txt, font, 0.50, 2)
        cv2.putText(full_img, badge_txt, (700 - bw_t // 2, 42 + bh_t // 2), font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        # Length & Speed metric boxes
        metrics_hud = [
            (778, 862, "LENGTH", f"{pass_ev['length_m']:.1f} m"),
            (872, 964, "SPEED", f"{pass_ev['avg_speed_kmh']:.0f} km/h"),
            (974, 1036, "TIME", f"{pass_ev['duration_sec']:.2f}s"),
        ]
        for mx1, mx2, m_lbl, m_val in metrics_hud:
            cv2.rectangle(full_img, (mx1, 18), (mx2, 66), (28, 35, 48), -1)
            cv2.rectangle(full_img, (mx1, 18), (mx2, 66), (62, 74, 94), 1)
            cx_m = (mx1 + mx2) // 2
            (lw, lh), _ = cv2.getTextSize(m_lbl, font, 0.34, 1)
            (vw, vh), _ = cv2.getTextSize(m_val, font, 0.48, 2)
            cv2.putText(full_img, m_lbl, (cx_m - lw // 2, 34), font, 0.34, (148, 163, 184), 1, cv2.LINE_AA)
            cv2.putText(full_img, m_val, (cx_m - vw // 2, 57), font, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

        # ------------------------------------------------------- 5. COMPACT FOOTER LEGEND (SAFE MARGINS)
        fy = header_h + self.height + 22
        cv2.circle(full_img, (26, fy - 4), 6, team_colors[0], -1, cv2.LINE_AA)
        cv2.putText(full_img, "Team 0 (Home)", (38, fy), font, 0.38, (210, 218, 230), 1, cv2.LINE_AA)

        cv2.circle(full_img, (156, fy - 4), 6, team_colors[1], -1, cv2.LINE_AA)
        cv2.putText(full_img, "Team 1 (Away)", (168, fy), font, 0.38, (210, 218, 230), 1, cv2.LINE_AA)

        cv2.line(full_img, (290, fy - 4), (318, fy - 4), (110, 245, 85), 3, cv2.LINE_AA)
        cv2.putText(full_img, "Completed Pass", (326, fy), font, 0.38, (210, 218, 230), 1, cv2.LINE_AA)

        cv2.line(full_img, (458, fy - 4), (486, fy - 4), (70, 85, 255), 3, cv2.LINE_AA)
        cv2.putText(full_img, "Incomplete / Intercepted", (494, fy), font, 0.38, (210, 218, 230), 1, cv2.LINE_AA)

        cv2.line(full_img, (682, fy - 4), (708, fy - 4), (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(full_img, "Receiver Run  |  FIFA 105m x 68m", (716, fy), font, 0.38, (165, 178, 196), 1, cv2.LINE_AA)

        return full_img
