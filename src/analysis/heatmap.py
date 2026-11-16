"""
Heatmap, trajectory, and average position spatial analytics module.
Supports both camera perspective and 2D top-down rectangular pitch projections,
metrics calculation (physical distance, speed, tactical zones), and serialization.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class SpatialAnalytics:
    """Computes player movement heatmaps, trajectories, and average pitch positions."""

    # Pitch dimensions in meters (FIFA Standard)
    PITCH_LENGTH_M = 105.0
    PITCH_WIDTH_M = 68.0

    # 2D pitch pixel area inside margins (field_x1=40, field_x2=1010 -> 970px; field_y1=40, field_y2=640 -> 600px)
    FIELD_X1 = 40.0
    FIELD_Y1 = 40.0
    FIELD_W = 970.0
    FIELD_H = 600.0

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        self.width = frame_width
        self.height = frame_height
        # Mapping from tracker_id to list of (x, y) bottom-center coordinates (camera space)
        self.player_tracks: Dict[int, List[Tuple[float, float]]] = {}
        # Mapping from tracker_id to list of (x, y) 2D pitch coordinates (2D space)
        self.player_tracks_2d: Dict[int, List[Tuple[float, float]]] = {}
        # Mapping from tracker_id to team_id (0: Team 0, 1: Team 1, -1: Ref)
        self.player_teams: Dict[int, int] = {}
        self.total_processed_frames: int = 0

    def add_position(
        self,
        tracker_id: int,
        bbox: List[float],
        team_id: int,
        pos_2d: Optional[Tuple[float, float]] = None
    ):
        """
        Adds player's bottom-center coordinate (feet location) and optional 2D pitch coordinate.
        """
        x1, y1, x2, y2 = bbox
        bottom_center_x = (x1 + x2) / 2.0
        bottom_center_y = y2
        
        if tracker_id not in self.player_tracks:
            self.player_tracks[tracker_id] = []
            self.player_tracks_2d[tracker_id] = []

        self.player_tracks[tracker_id].append((float(bottom_center_x), float(bottom_center_y)))
        if pos_2d is not None:
            self.player_tracks_2d[tracker_id].append((float(pos_2d[0]), float(pos_2d[1])))

        self.player_teams[tracker_id] = team_id

    def set_total_frames(self, frames: int):
        self.total_processed_frames = max(1, frames)

    def get_average_positions(self) -> Dict[int, Tuple[float, float]]:
        """Calculates average (mean) (x, y) camera position for each player."""
        avg_positions = {}
        for tracker_id, positions in self.player_tracks.items():
            if len(positions) > 0:
                coords = np.array(positions)
                avg_positions[tracker_id] = (float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1])))
        return avg_positions

    def get_average_positions_2d(self) -> Dict[int, Tuple[float, float]]:
        """Calculates average (mean) (x, y) 2D pitch position for each player."""
        avg_positions = {}
        for tracker_id, positions in self.player_tracks_2d.items():
            if len(positions) > 0:
                coords = np.array(positions)
                avg_positions[tracker_id] = (float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1])))
        return avg_positions

    def get_2d_positions(
        self,
        tracker_id: Optional[int] = None,
        team_id: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        """Gathers 2D pitch coordinates filtered by tracker_id or team_id."""
        coords = []
        for p_id, positions in self.player_tracks_2d.items():
            if tracker_id is not None and p_id != tracker_id:
                continue
            if team_id is not None and self.player_teams.get(p_id) != team_id:
                continue
            coords.extend(positions)
        return coords

    def get_player_metrics(self, tracker_id: int, fps: float = 25.0) -> Dict:
        """
        Calculates comprehensive physical and tactical metrics for a specific player.
        """
        positions_2d = self.player_tracks_2d.get(tracker_id, [])
        positions_cam = self.player_tracks.get(tracker_id, [])
        team_id = self.player_teams.get(tracker_id, 0)
        total_f = max(1, self.total_processed_frames)
        frames_active = len(positions_2d)

        # 1. Physical Distance Calculation (2D pixels to meters)
        scale_x = self.PITCH_LENGTH_M / max(1.0, self.FIELD_W)
        scale_y = self.PITCH_WIDTH_M / max(1.0, self.FIELD_H)

        distance_m = 0.0
        for i in range(1, len(positions_2d)):
            dx_m = (positions_2d[i][0] - positions_2d[i - 1][0]) * scale_x
            dy_m = (positions_2d[i][1] - positions_2d[i - 1][1]) * scale_y
            step_dist = math.hypot(dx_m, dy_m)
            # Filter teleporting glitches / ID jumps (> 4.5m in a single frame is physically impossible)
            if step_dist < 4.5:
                distance_m += step_dist

        active_seconds = frames_active / fps if fps > 0 else 1.0
        avg_speed_kmh = (distance_m / max(0.1, active_seconds)) * 3.6 if active_seconds > 0 else 0.0

        # 2. Average Position
        if len(positions_2d) > 0:
            coords = np.array(positions_2d)
            avg_x = float(np.mean(coords[:, 0]))
            avg_y = float(np.mean(coords[:, 1]))
        else:
            avg_x, avg_y = (self.FIELD_X1 + self.FIELD_W / 2.0, self.FIELD_Y1 + self.FIELD_H / 2.0)

        # Meter Coordinates
        xm = np.clip((avg_x - self.FIELD_X1) * scale_x, 0.0, self.PITCH_LENGTH_M)
        ym = np.clip((avg_y - self.FIELD_Y1) * scale_y, 0.0, self.PITCH_WIDTH_M)

        # 3. Tactical Zone Classification
        # Longitudinal (X): 1st third (Defensive), 2nd third (Middle), 3rd third (Attacking)
        if xm < 35.0:
            x_zone = "1. Bölge (Savunma)"
        elif xm < 70.0:
            x_zone = "2. Bölge (Orta Saha)"
        else:
            x_zone = "3. Bölge (Hücum)"

        # Lateral (Y): Left flank, Center channel, Right flank
        if ym < 22.6:
            y_zone = "Sol Koridor"
        elif ym < 45.3:
            y_zone = "Merkez Kanal"
        else:
            y_zone = "Sağ Koridor"

        tactical_role = f"{y_zone} • {x_zone}"

        # 4. Thirds Distribution (%)
        def_cnt, mid_cnt, att_cnt = 0, 0, 0
        for px, py in positions_2d:
            cur_xm = (px - self.FIELD_X1) * scale_x
            if cur_xm < 35.0:
                def_cnt += 1
            elif cur_xm < 70.0:
                mid_cnt += 1
            else:
                att_cnt += 1

        n_pts = max(1, len(positions_2d))
        def_pct = round((def_cnt / n_pts) * 100, 1)
        mid_pct = round((mid_cnt / n_pts) * 100, 1)
        att_pct = round((att_cnt / n_pts) * 100, 1)

        return {
            "tracker_id": tracker_id,
            "team_id": team_id,
            "team_label": f"Takım {team_id}" if team_id >= 0 else "Hakem",
            "frames_active": frames_active,
            "presence_pct": round((frames_active / total_f) * 100, 1),
            "distance_m": round(distance_m, 1),
            "avg_speed_kmh": round(avg_speed_kmh, 1),
            "avg_pos_2d": (round(avg_x, 1), round(avg_y, 1)),
            "pos_meters": (round(xm, 1), round(ym, 1)),
            "tactical_zone": tactical_role,
            "defensive_pct": def_pct,
            "middle_pct": mid_pct,
            "attacking_pct": att_pct,
            "positions_2d": positions_2d,
            "positions_cam": positions_cam
        }

    def get_team_metrics(self, team_id: int, fps: float = 25.0) -> Dict:
        """
        Aggregates individual player statistics into complete team tactical intelligence.
        """
        player_ids = [pid for pid, tid in self.player_teams.items() if tid == team_id]
        if not player_ids:
            return {
                "team_id": team_id,
                "team_label": f"Takım {team_id}",
                "player_count": 0,
                "player_ids": [],
                "centroid_2d": (0.0, 0.0),
                "centroid_m": (0.0, 0.0),
                "team_width_m": 0.0,
                "team_depth_m": 0.0,
                "total_distance_m": 0.0,
                "defensive_pct": 0.0,
                "middle_pct": 0.0,
                "attacking_pct": 0.0,
                "left_flank_pct": 0.0,
                "center_pct": 0.0,
                "right_flank_pct": 0.0
            }

        scale_x = self.PITCH_LENGTH_M / max(1.0, self.FIELD_W)
        scale_y = self.PITCH_WIDTH_M / max(1.0, self.FIELD_H)

        avg_positions_2d = self.get_average_positions_2d()
        team_avg_pts = [avg_positions_2d[pid] for pid in player_ids if pid in avg_positions_2d]

        # Centroid
        if team_avg_pts:
            pts_arr = np.array(team_avg_pts)
            cx = float(np.mean(pts_arr[:, 0]))
            cy = float(np.mean(pts_arr[:, 1]))
            # Compactness (spread)
            depth_m = float((np.max(pts_arr[:, 0]) - np.min(pts_arr[:, 0])) * scale_x)
            width_m = float((np.max(pts_arr[:, 1]) - np.min(pts_arr[:, 1])) * scale_y)
        else:
            cx, cy = (self.FIELD_X1 + self.FIELD_W / 2.0, self.FIELD_Y1 + self.FIELD_H / 2.0)
            depth_m, width_m = 0.0, 0.0

        cx_m = np.clip((cx - self.FIELD_X1) * scale_x, 0.0, self.PITCH_LENGTH_M)
        cy_m = np.clip((cy - self.FIELD_Y1) * scale_y, 0.0, self.PITCH_WIDTH_M)

        # Aggregate distance
        total_dist_m = 0.0
        for pid in player_ids:
            pm = self.get_player_metrics(pid, fps=fps)
            total_dist_m += pm["distance_m"]

        # Aggregate zone distribution across all team 2D positions
        all_team_positions = self.get_2d_positions(team_id=team_id)
        def_cnt, mid_cnt, att_cnt = 0, 0, 0
        left_cnt, center_cnt, right_cnt = 0, 0, 0

        for px, py in all_team_positions:
            xm = (px - self.FIELD_X1) * scale_x
            ym = (py - self.FIELD_Y1) * scale_y

            if xm < 35.0:
                def_cnt += 1
            elif xm < 70.0:
                mid_cnt += 1
            else:
                att_cnt += 1

            if ym < 22.6:
                left_cnt += 1
            elif ym < 45.3:
                center_cnt += 1
            else:
                right_cnt += 1

        n_all = max(1, len(all_team_positions))

        return {
            "team_id": team_id,
            "team_label": f"Takım {team_id}",
            "player_count": len(player_ids),
            "player_ids": sorted(player_ids),
            "centroid_2d": (round(cx, 1), round(cy, 1)),
            "centroid_m": (round(cx_m, 1), round(cy_m, 1)),
            "team_width_m": round(width_m, 1),
            "team_depth_m": round(depth_m, 1),
            "total_distance_m": round(total_dist_m, 1),
            "defensive_pct": round((def_cnt / n_all) * 100, 1),
            "middle_pct": round((mid_cnt / n_all) * 100, 1),
            "attacking_pct": round((att_cnt / n_all) * 100, 1),
            "left_flank_pct": round((left_cnt / n_all) * 100, 1),
            "center_pct": round((center_cnt / n_all) * 100, 1),
            "right_flank_pct": round((right_cnt / n_all) * 100, 1)
        }

    def get_player_summary(self) -> List[Dict]:
        """Returns structured summary table of all tracked players."""
        summary = []
        avg_pos = self.get_average_positions()
        avg_pos_2d = self.get_average_positions_2d()
        total_f = max(1, self.total_processed_frames)

        for p_id in sorted(self.player_tracks.keys()):
            positions = self.player_tracks[p_id]
            team_id = self.player_teams.get(p_id, 0)
            avg_x, avg_y = avg_pos.get(p_id, (0.0, 0.0))
            p2d = avg_pos_2d.get(p_id, (0.0, 0.0))
            pm = self.get_player_metrics(p_id)
            
            summary.append({
                "Player ID": f"#{p_id}",
                "Raw ID": p_id,
                "Team": f"Team {team_id}" if team_id >= 0 else "Referee",
                "Team ID": team_id,
                "Frames Active": len(positions),
                "Presence (%)": round((len(positions) / total_f) * 100, 1),
                "Est. Distance (m)": pm["distance_m"],
                "Avg Speed (km/h)": pm["avg_speed_kmh"],
                "Pitch (X, Y) [m]": f"({pm['pos_meters'][0]}m, {pm['pos_meters'][1]}m)",
                "Tactical Zone": pm["tactical_zone"]
            })
        return summary

    def generate_heatmap(
        self,
        tracker_id: Optional[int] = None,
        team_id: Optional[int] = None,
        sigma: float = 25.0
    ) -> np.ndarray:
        """Generates camera perspective 2D Gaussian density heatmap."""
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

        for x, y in coords_to_include:
            ix, iy = int(x), int(y)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                density[iy, ix] += 1.0

        density = gaussian_filter(density, sigma=sigma)
        max_val = np.max(density)
        if max_val > 0:
            density = (density / max_val * 255).astype(np.uint8)
        else:
            density = density.astype(np.uint8)

        heatmap_bgr = cv2.applyColorMap(density, cv2.COLORMAP_JET)
        mask = density < 8
        heatmap_bgr[mask] = 0
        return heatmap_bgr

    def generate_player_overlay(
        self,
        tracker_id: int,
        background_frame: np.ndarray,
        sigma: float = 20.0
    ) -> np.ndarray:
        """Generates a heatmap for a single player overlaid onto camera background."""
        heatmap_bgr = self.generate_heatmap(tracker_id=tracker_id, sigma=sigma)
        overlay = background_frame.copy()
        nonzero_mask = np.any(heatmap_bgr > 0, axis=-1)

        if np.any(nonzero_mask):
            overlay[nonzero_mask] = cv2.addWeighted(
                overlay[nonzero_mask], 0.35,
                heatmap_bgr[nonzero_mask], 0.65,
                0
            )
            
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
        """Draws the movement trajectory trail of a specific player on camera frame."""
        canvas = background_frame.copy()
        positions = self.player_tracks.get(tracker_id, [])
        if len(positions) < 2:
            return canvas

        team_id = self.player_teams.get(tracker_id, 0)
        trail_color = (35, 45, 235) if team_id == 0 else (235, 130, 25)

        pts = np.array(positions, np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts], isClosed=False, color=trail_color, thickness=3, lineType=cv2.LINE_AA)

        start_pt = (int(positions[0][0]), int(positions[0][1]))
        cv2.circle(canvas, start_pt, 8, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, start_pt, 9, (0, 0, 0), 2, cv2.LINE_AA)

        end_pt = (int(positions[-1][0]), int(positions[-1][1]))
        cv2.circle(canvas, end_pt, 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, end_pt, 9, (0, 0, 0), 2, cv2.LINE_AA)

        return canvas

    def save_to_json(self, file_path: str):
        """Serializes spatial tracking and 2D data to a JSON file."""
        data = {
            "width": self.width,
            "height": self.height,
            "total_processed_frames": self.total_processed_frames,
            "player_teams": {str(k): int(v) for k, v in self.player_teams.items()},
            "player_tracks": {str(k): [[float(p[0]), float(p[1])] for p in v] for k, v in self.player_tracks.items()},
            "player_tracks_2d": {str(k): [[float(p[0]), float(p[1])] for p in v] for k, v in self.player_tracks_2d.items()},
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load_from_json(cls, file_path: str) -> "SpatialAnalytics":
        """Reconstructs a SpatialAnalytics instance from a saved JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Spatial data file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(frame_width=data.get("width", 1920), frame_height=data.get("height", 1080))
        instance.total_processed_frames = data.get("total_processed_frames", 0)
        instance.player_teams = {int(k): int(v) for k, v in data.get("player_teams", {}).items()}
        instance.player_tracks = {
            int(k): [(float(p[0]), float(p[1])) for p in v]
            for k, v in data.get("player_tracks", {}).items()
        }
        instance.player_tracks_2d = {
            int(k): [(float(p[0]), float(p[1])) for p in v]
            for k, v in data.get("player_tracks_2d", {}).items()
        }
        return instance
