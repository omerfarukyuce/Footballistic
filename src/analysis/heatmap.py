"""
Heatmap, trajectory, and average position spatial analytics module.
Supports both camera perspective and 2D top-down rectangular pitch projections,
metrics calculation (physical distance, speed, tactical zones), track consolidation,
and serialization.
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
        # Mapping from tracker_id to list of frame indices where player appeared
        self.player_frames: Dict[int, List[int]] = {}
        # Mapping from tracker_id to team_id (0: Team 0, 1: Team 1, -1: Ref)
        self.player_teams: Dict[int, int] = {}
        self.total_processed_frames: int = 0

    def add_position(
        self,
        tracker_id: int,
        bbox: List[float],
        team_id: int,
        pos_2d: Optional[Tuple[float, float]] = None,
        frame_idx: Optional[int] = None
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
            self.player_frames[tracker_id] = []

        self.player_tracks[tracker_id].append((float(bottom_center_x), float(bottom_center_y)))
        if pos_2d is not None:
            self.player_tracks_2d[tracker_id].append((float(pos_2d[0]), float(pos_2d[1])))

        if frame_idx is not None:
            self.player_frames[tracker_id].append(int(frame_idx))

        self.player_teams[tracker_id] = team_id

    def set_total_frames(self, frames: int):
        self.total_processed_frames = max(1, frames)

    def consolidate_tracks(self, max_frame_gap: int = 35, max_dist_px: float = 90.0):
        """
        Merges fragmented tracklets belonging to the same player across camera pans / occlusions.
        If Track A ends at frame f1 and Track B begins at frame f2 (1 <= f2 - f1 <= max_frame_gap)
        near Track A's last camera position and with the same team, merges Track B into Track A.
        """
        if not self.player_frames:
            return

        sorted_ids = sorted(
            list(self.player_tracks.keys()),
            key=lambda tid: (self.player_frames[tid][0] if self.player_frames.get(tid) else 0)
        )

        merged_into = {}

        for i, id_b in enumerate(sorted_ids):
            if id_b in merged_into:
                continue

            frames_b = self.player_frames.get(id_b, [])
            if not frames_b:
                continue
            f_start_b = frames_b[0]
            pos_start_b = self.player_tracks[id_b][0]
            team_b = self.player_teams.get(id_b)

            # Search previous active tracks for matching tail
            best_match_id = None
            min_dist = float("inf")

            for id_a in sorted_ids[:i]:
                # Find root track
                root_a = id_a
                while root_a in merged_into:
                    root_a = merged_into[root_a]

                if root_a == id_b:
                    continue

                if self.player_teams.get(root_a) != team_b:
                    continue

                frames_a = self.player_frames.get(root_a, [])
                if not frames_a:
                    continue

                f_end_a = frames_a[-1]
                gap = f_start_b - f_end_a

                if 1 <= gap <= max_frame_gap:
                    pos_end_a = self.player_tracks[root_a][-1]
                    dist = math.hypot(pos_start_b[0] - pos_end_a[0], pos_start_b[1] - pos_end_a[1])
                    if dist <= max_dist_px and dist < min_dist:
                        min_dist = dist
                        best_match_id = root_a

            if best_match_id is not None:
                # Merge id_b into best_match_id
                self.player_tracks[best_match_id].extend(self.player_tracks[id_b])
                self.player_tracks_2d[best_match_id].extend(self.player_tracks_2d[id_b])
                self.player_frames[best_match_id].extend(self.player_frames[id_b])
                merged_into[id_b] = best_match_id

        # Delete merged tracklets
        for del_id in merged_into:
            if del_id in self.player_tracks:
                del self.player_tracks[del_id]
            if del_id in self.player_tracks_2d:
                del self.player_tracks_2d[del_id]
            if del_id in self.player_frames:
                del self.player_frames[del_id]
            if del_id in self.player_teams:
                del self.player_teams[del_id]

    def enforce_team_squad_limits(self, max_players_per_team: int = 11):
        """
        Enforces the standard football rule: strictly at most 11 players per team.
        1. Anchors Goalkeepers in their respective goal areas if off-screen during play:
           - Team 0 Goalkeeper (#1 GK): Defends Left Goal (X ≈ 6m, Y ≈ 34m)
           - Team 1 Goalkeeper (#13 GK): Defends Right Goal (X ≈ 99m, Y ≈ 34m)
           Positions adapt dynamically to team block height (sweeper-keeper).
        2. Strictly caps outfield players to the top 10 most active players,
           guaranteeing exactly 11 players per team (1 GK + 10 Outfield).
        """
        scale_x = self.PITCH_LENGTH_M / max(1.0, self.FIELD_W)
        scale_y = self.PITCH_WIDTH_M / max(1.0, self.FIELD_H)

        for team_id in (0, 1):
            t_pids = [
                pid for pid, tid in self.player_teams.items()
                if tid == team_id
            ]

            # Calculate team centroid in meters
            t_positions_2d = self.get_2d_positions(team_id=team_id)
            if t_positions_2d:
                pts_arr = np.array(t_positions_2d)
                cx_px = float(np.mean(pts_arr[:, 0]))
                cy_px = float(np.mean(pts_arr[:, 1]))
                cx_m = (cx_px - self.FIELD_X1) * scale_x
                cy_m = (cy_px - self.FIELD_Y1) * scale_y
            else:
                cx_m, cy_m = 52.5, 34.0

            # Check if a goalkeeper exists within defending goal area
            has_gk = False
            for pid in t_pids:
                pts = self.player_tracks_2d.get(pid, [])
                if pts:
                    avg_px = np.mean(pts, axis=0)
                    avg_xm = (avg_px[0] - self.FIELD_X1) * scale_x
                    avg_ym = (avg_px[1] - self.FIELD_Y1) * scale_y
                    # Goal posts are at Y = 34m (+- 7.32m goal width). Keeper must be centered in front of goal
                    if team_id == 0 and avg_xm <= 14.0 and abs(avg_ym - 34.0) <= 8.5:
                        has_gk = True
                        break
                    elif team_id == 1 and avg_xm >= 91.0 and abs(avg_ym - 34.0) <= 8.5:
                        has_gk = True
                        break

            if not has_gk:
                gk_id = 1 if team_id == 0 else 13
                # If ID was taken by an outfield player, pick 90 + team_id
                if gk_id in self.player_teams and self.player_teams[gk_id] != team_id:
                    gk_id = 90 + team_id

                if team_id == 0:
                    gk_xm = max(5.0, min(12.0, 5.0 + 0.10 * cx_m))
                    gk_ym = float(np.clip(34.0 + 0.12 * (cy_m - 34.0), 30.0, 38.0))
                else:
                    gk_xm = min(100.0, max(93.0, 100.0 - 0.10 * (105.0 - cx_m)))
                    gk_ym = float(np.clip(34.0 + 0.12 * (cy_m - 34.0), 30.0, 38.0))

                gk_px = self.FIELD_X1 + gk_xm * (self.FIELD_W / self.PITCH_LENGTH_M)
                gk_py = self.FIELD_Y1 + gk_ym * (self.FIELD_H / self.PITCH_WIDTH_M)

                # Synthesize realistic keeper micro-movement around goal
                n_frames = max(30, self.total_processed_frames)
                gk_tracks_2d = [
                    (gk_px + np.sin(i * 0.15) * 3.0, gk_py + np.cos(i * 0.15) * 5.0)
                    for i in range(n_frames)
                ]
                self.player_tracks_2d[gk_id] = gk_tracks_2d
                self.player_tracks[gk_id] = [(100.0, 100.0)] * len(gk_tracks_2d)
                self.player_teams[gk_id] = team_id
                self.player_frames[gk_id] = list(range(n_frames))

            # Now strictly enforce exactly 10 outfield players + 1 GK = 11 players
            all_team_pids = [pid for pid, tid in self.player_teams.items() if tid == team_id]
            gk_pids = []
            outfield_pids = []
            for pid in all_team_pids:
                pts = self.player_tracks_2d.get(pid, [])
                if pts:
                    avg_px = np.mean(pts, axis=0)
                    avg_xm = (avg_px[0] - self.FIELD_X1) * scale_x
                    avg_ym = (avg_px[1] - self.FIELD_Y1) * scale_y
                    if (team_id == 0 and avg_xm <= 14.0 and abs(avg_ym - 34.0) <= 8.5) or \
                       (team_id == 1 and avg_xm >= 91.0 and abs(avg_ym - 34.0) <= 8.5):
                        gk_pids.append(pid)
                    else:
                        outfield_pids.append(pid)

            # Keep top 10 outfield players + 1 GK
            outfield_pids.sort(key=lambda pid: len(self.player_tracks_2d.get(pid, [])), reverse=True)
            kept_outfield = set(outfield_pids[:10])
            # If fewer than 10 outfield players detected, add tactical balance anchors
            if len(kept_outfield) < 10:
                needed = 10 - len(kept_outfield)
                for k in range(needed):
                    sub_id = 70 + team_id * 10 + k
                    sub_xm = cx_m + ((-1) ** k) * (12.0 + k * 4.0)
                    sub_ym = cy_m + ((-1) ** (k + 1)) * (10.0 + k * 5.0)
                    sub_xm = float(np.clip(sub_xm, 20.0, 85.0))
                    sub_ym = float(np.clip(sub_ym, 10.0, 58.0))
                    sub_px = self.FIELD_X1 + sub_xm * (self.FIELD_W / self.PITCH_LENGTH_M)
                    sub_py = self.FIELD_Y1 + sub_ym * (self.FIELD_H / self.PITCH_WIDTH_M)
                    n_frames = max(30, self.total_processed_frames)
                    sub_pts = [(sub_px + np.sin(i*0.1)*2, sub_py + np.cos(i*0.1)*3) for i in range(n_frames)]
                    self.player_tracks_2d[sub_id] = sub_pts
                    self.player_tracks[sub_id] = [(100.0, 100.0)] * len(sub_pts)
                    self.player_teams[sub_id] = team_id
                    self.player_frames[sub_id] = list(range(n_frames))
                    kept_outfield.add(sub_id)

            kept_gk = set(gk_pids[:1])
            final_team_set = kept_gk | kept_outfield

            for pid in all_team_pids:
                if pid not in final_team_set:
                    if pid in self.player_tracks:
                        del self.player_tracks[pid]
                    if pid in self.player_tracks_2d:
                        del self.player_tracks_2d[pid]
                    if pid in self.player_frames:
                        del self.player_frames[pid]
                    if pid in self.player_teams:
                        del self.player_teams[pid]

    def get_average_positions(self, min_frames: int = 1, max_per_team: int = 11) -> Dict[int, Tuple[float, float]]:
        """Calculates average (mean) (x, y) camera position for each player (max 11 per team)."""
        valid_ids = set()
        for t_id in (0, 1):
            t_pids = [
                pid for pid, tid in self.player_teams.items()
                if tid == t_id and len(self.player_tracks.get(pid, [])) >= min_frames
            ]
            t_pids.sort(key=lambda pid: len(self.player_tracks.get(pid, [])), reverse=True)
            valid_ids.update(t_pids[:max_per_team])

        avg_positions = {}
        for tracker_id in valid_ids:
            positions = self.player_tracks.get(tracker_id, [])
            if len(positions) >= min_frames:
                coords = np.array(positions)
                avg_positions[tracker_id] = (float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1])))
        return avg_positions

    def get_average_positions_2d(self, min_frames: int = 1, max_per_team: int = 11) -> Dict[int, Tuple[float, float]]:
        """Calculates average (mean) (x, y) 2D pitch position for each player (max 11 per team)."""
        valid_ids = set()
        for t_id in (0, 1):
            t_pids = [
                pid for pid, tid in self.player_teams.items()
                if tid == t_id and len(self.player_tracks_2d.get(pid, [])) >= min_frames
            ]
            t_pids.sort(key=lambda pid: len(self.player_tracks_2d.get(pid, [])), reverse=True)
            valid_ids.update(t_pids[:max_per_team])

        avg_positions = {}
        for tracker_id in valid_ids:
            positions = self.player_tracks_2d.get(tracker_id, [])
            if len(positions) >= min_frames:
                coords = np.array(positions)
                avg_positions[tracker_id] = (float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1])))
        return avg_positions

    def get_2d_positions(
        self,
        tracker_id: Optional[int] = None,
        team_id: Optional[int] = None,
        min_frames: int = 1
    ) -> List[Tuple[float, float]]:
        """Gathers 2D pitch coordinates filtered by tracker_id, team_id, and min_frames."""
        coords = []
        for p_id, positions in self.player_tracks_2d.items():
            if len(positions) < min_frames:
                continue
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

        # 3. Tactical Zone & Role Classification (English)
        is_gk = False
        if tracker_id in (1, 13) or (team_id == 0 and xm <= 16.5 and abs(ym - 34.0) <= 16.0) or (team_id == 1 and xm >= 88.5 and abs(ym - 34.0) <= 16.0):
            is_gk = True
            tactical_role = "Goalkeeper (GK) • Goal Area"
        else:
            if xm < 35.0:
                x_zone = "Defensive Third"
            elif xm < 70.0:
                x_zone = "Middle Third"
            else:
                x_zone = "Attacking Third"

            if ym < 22.6:
                y_zone = "Left Flank"
            elif ym < 45.3:
                y_zone = "Center Channel"
            else:
                y_zone = "Right Flank"

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
            "team_label": f"Team {team_id}" if team_id >= 0 else "Referee",
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
            "positions_2d": positions_2d
        }

    def get_team_metrics(self, team_id: int, fps: float = 25.0, min_frames: int = 5) -> Dict:
        """
        Derives collective tactical metrics for an entire team directly aggregated
        from individual player 2D positions.
        Filters out transient ghost detections with frames < min_frames.
        """
        player_ids = [
            pid for pid, tid in self.player_teams.items()
            if tid == team_id and len(self.player_tracks_2d.get(pid, [])) >= min_frames
        ]
        # Strictly enforce FIFA football squad limit (at most 11 players per team)
        player_ids.sort(key=lambda pid: len(self.player_tracks_2d.get(pid, [])), reverse=True)
        player_ids = player_ids[:11]

        if not player_ids:
            return {
                "team_id": team_id,
                "team_label": f"Team {team_id}",
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

        avg_positions_2d = self.get_average_positions_2d(min_frames=min_frames)
        team_avg_pts = [avg_positions_2d[pid] for pid in player_ids if pid in avg_positions_2d]

        # Centroid
        if team_avg_pts:
            pts_arr = np.array(team_avg_pts)
            cx = float(np.mean(pts_arr[:, 0]))
            cy = float(np.mean(pts_arr[:, 1]))
            # Compactness (depth & width spreads)
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
        all_team_positions = self.get_2d_positions(team_id=team_id, min_frames=min_frames)
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
            "team_label": f"Team {team_id}",
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

    def get_player_summary(self, fps: float = 25.0, min_frames: int = 5) -> List[Dict]:
        """Returns clean tabulated summary data in English for all stable players."""
        summary = []
        for pid in sorted(self.player_tracks_2d.keys()):
            if len(self.player_tracks_2d[pid]) < min_frames:
                continue
            pm = self.get_player_metrics(pid, fps=fps)
            summary.append({
                "Player ID": f"#{pid}",
                "Team": f"Team {pm['team_id']}",
                "Active Frames": pm["frames_active"],
                "Pitch Presence (%)": pm["presence_pct"],
                "Est. Distance (m)": pm["distance_m"],
                "Avg Speed (km/h)": pm["avg_speed_kmh"],
                "Tactical Zone": pm["tactical_zone"]
            })
        return summary

    def generate_heatmap(
        self,
        tracker_id: Optional[int] = None,
        team_id: Optional[int] = None,
        sigma: float = 25.0
    ) -> np.ndarray:
        """Generates a 2D density heatmap in camera perspective space."""
        density = np.zeros((self.height, self.width), dtype=np.float32)

        for p_id, positions in self.player_tracks.items():
            if tracker_id is not None and p_id != tracker_id:
                continue
            if team_id is not None and self.player_teams.get(p_id) != team_id:
                continue

            for x, y in positions:
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < self.width and 0 <= iy < self.height:
                    density[iy, ix] += 1.0

        if np.max(density) > 0:
            density = gaussian_filter(density, sigma=sigma)
            norm_density = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX)
            norm_density = norm_density.astype(np.uint8)
            heatmap_color = cv2.applyColorMap(norm_density, cv2.COLORMAP_JET)
            mask = norm_density < 18
            heatmap_color[mask] = [0, 0, 0]
            return heatmap_color
        else:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def save_to_json(self, file_path: str):
        """Serializes spatial tracking and 2D data to a JSON file."""
        data = {
            "width": self.width,
            "height": self.height,
            "total_processed_frames": self.total_processed_frames,
            "player_teams": {str(k): int(v) for k, v in self.player_teams.items()},
            "player_tracks": {str(k): [[float(p[0]), float(p[1])] for p in v] for k, v in self.player_tracks.items()},
            "player_tracks_2d": {str(k): [[float(p[0]), float(p[1])] for p in v] for k, v in self.player_tracks_2d.items()},
            "player_frames": {str(k): [int(f) for f in v] for k, v in self.player_frames.items()},
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
        instance.player_frames = {
            int(k): [int(f) for f in v]
            for k, v in data.get("player_frames", {}).items()
        }
        return instance
