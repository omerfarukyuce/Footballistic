"""
Phase 2: Football Event Spotting, Dynamic Ball Possession & Passing Network Engine.

Extracts higher-order tactical match intelligence from 2D player trajectories
and GMC-compensated ball tracking:
  1. Dynamic Ball Possession:
     - Perspective-aware foot-to-ball proximity + 2D metric distance gating.
     - Kinematic aerial-flight filter (ignores players underneath a high ball
       flying overhead unless the ball decelerates, is held >= 3 frames, or
       deflects direction by > 22 degrees).
     - Continuous team possession percentage (%), possession duration (s),
       touch counts, and second-by-second possession timeline.
  2. Passing Network & Pass Classification:
     - Completed passes between teammates, pass distance (m), direction, and
       pass matrix / graph edges for 2D Passing Network rendering.
     - Tactical pass qualifiers: Progressive Pass, Final-Third Entry,
       Penalty-Box Entry, and Cross.
  3. Turnovers, Interceptions & Ball Recoveries:
     - Possession transitions between opposing teams with ball-winner attribution
       and pitch recovery zone (Defensive / Middle / Attacking Third).
  4. Progressive Carries & Shot / Goal-Mouth Vectors:
     - Single-player ball carries advancing >= 7m or entering the final third / box.
     - High-velocity ball vectors directed toward the opponent goal mouth.
"""

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.analysis.heatmap import SpatialAnalytics


@dataclass
class TouchSpell:
    player_id: int
    team_id: int
    start_frame: int
    end_frame: int
    start_cam: Tuple[float, float]
    end_cam: Tuple[float, float]
    start_2d: Tuple[float, float]
    end_2d: Tuple[float, float]
    start_m: Tuple[float, float]
    end_m: Tuple[float, float]
    min_norm_dist: float


class MatchEventEngine:
    """Computes ball possession, completed passes, passing networks, recoveries, and entries."""

    def __init__(self, spatial_analytics: SpatialAnalytics, fps: float = 25.0):
        self.sa = spatial_analytics
        self.fps = max(1.0, float(fps))
        self.scale_x = self.sa.PITCH_LENGTH_M / max(1.0, self.sa.FIELD_W)
        self.scale_y = self.sa.PITCH_WIDTH_M / max(1.0, self.sa.FIELD_H)

        # Attacking direction per team: +1 means attacking left->right (toward X=105m),
        # -1 means attacking right->left (toward X=0m).
        self.attack_dir: Dict[int, int] = self._infer_attack_directions()

        # Computed outputs
        self.touch_spells: List[TouchSpell] = []
        self.frame_team_owner: Dict[int, int] = {}
        self.frame_player_owner: Dict[int, int] = {}
        self.passes: List[Dict] = []
        self.interceptions: List[Dict] = []
        self.carries_and_entries: List[Dict] = []
        self.shots: List[Dict] = []
        self.all_events: List[Dict] = []
        self.player_touches: Dict[int, int] = defaultdict(int)
        self.player_pass_stats: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"touches": 0, "passes_completed": 0, "passes_received": 0, "progressive_passes": 0, "recoveries": 0}
        )

        self._run_analysis()

    # ------------------------------------------------------------------ helpers

    def _px2d_to_m(self, px: float, py: float) -> Tuple[float, float]:
        xm = float(np.clip((px - self.sa.FIELD_X1) * self.scale_x, 0.0, self.sa.PITCH_LENGTH_M))
        ym = float(np.clip((py - self.sa.FIELD_Y1) * self.scale_y, 0.0, self.sa.PITCH_WIDTH_M))
        return round(xm, 1), round(ym, 1)

    def _infer_attack_directions(self) -> Dict[int, int]:
        """Infers which goal each team defends from average outfield X coordinates."""
        mean_x = {}
        for tid in (0, 1):
            pts = self.sa.get_2d_positions(team_id=tid)
            if pts:
                mean_x[tid] = float(np.mean([(p[0] - self.sa.FIELD_X1) * self.scale_x for p in pts]))
            else:
                mean_x[tid] = 45.0 if tid == 0 else 60.0
        if mean_x[0] <= mean_x[1]:
            return {0: 1, 1: -1}  # Team 0 defends Left (attacks +X), Team 1 defends Right (attacks -X)
        return {0: -1, 1: 1}

    def _attack_x(self, xm: float, team_id: int) -> float:
        """Converts raw pitch X (0..105m) to team-relative attacking X (0=own goal, 105=opp goal)."""
        return xm if self.attack_dir.get(team_id, 1) == 1 else (105.0 - xm)

    def _zone_name(self, xm: float, team_id: int) -> str:
        ax = self._attack_x(xm, team_id)
        if ax < 35.0:
            return "Defensive Third"
        elif ax < 70.0:
            return "Middle Third"
        return "Attacking Third"

    # ------------------------------------------------------------------ core analysis

    def _build_frame_lookups(self):
        """Builds per-frame player and ball coordinate tables (with fallback for legacy JSONs)."""
        f_players: Dict[int, List[Tuple[int, int, float, float, float, float]]] = defaultdict(list)

        for pid, frames in self.sa.player_frames.items():
            tid = self.sa.player_teams.get(pid, -1)
            if tid not in (0, 1):
                continue
            tr_cam = self.sa.player_tracks.get(pid, [])
            tr_2d = self.sa.player_tracks_2d.get(pid, [])
            for f, p_cam, p_2d in zip(frames, tr_cam, tr_2d):
                # Skip off-screen synthesized anchor points at (100.0, 100.0)
                if abs(p_cam[0] - 100.0) < 1e-3 and abs(p_cam[1] - 100.0) < 1e-3:
                    continue
                f_players[int(f)].append((
                    int(pid), int(tid),
                    float(p_cam[0]), float(p_cam[1]),
                    float(p_2d[0]), float(p_2d[1])
                ))

        f_ball: Dict[int, Tuple[float, float, float, float]] = {}
        cam_map = {int(f): (float(x), float(y)) for f, x, y in self.sa.ball_track}
        p2d_map = {int(f): (float(x), float(y)) for f, x, y in self.sa.ball_track_2d}

        for f in sorted(set(cam_map.keys()) & set(p2d_map.keys())):
            bx, by = cam_map[f]
            bx2, by2 = p2d_map[f]
            xm, ym = self._px2d_to_m(bx2, by2)

            # Filter out unattended static penalty-spot detections in legacy saved tracks
            if (9.5 <= xm <= 13.5 or 91.5 <= xm <= 95.5) and (27.0 <= ym <= 41.0):
                pls = f_players.get(f, [])
                min_cam_d = min(
                    (math.hypot(bx - p[2], by - p[3]) for p in pls),
                    default=999.0
                )
                if min_cam_d > 38.0:
                    continue

            f_ball[f] = (bx, by, bx2, by2)

        # Fallback for legacy spatial_data.json files without saved ball_track:
        # Estimate active ball position from player-interaction convergence with temporal hold
        if not f_ball and f_players:
            prev_bx2, prev_by2 = 525.0, 340.0
            cur_carrier = None
            hold_until = -1
            for f in sorted(f_players.keys()):
                pls = f_players[f]
                if len(pls) < 2:
                    continue
                if cur_carrier is not None and f <= hold_until:
                    match_p = next((p for p in pls if p[0] == cur_carrier), None)
                    if match_p is not None:
                        f_ball[f] = (match_p[2], match_p[3], match_p[4], match_p[5])
                        prev_bx2, prev_by2 = match_p[4], match_p[5]
                        continue
                best_p = None
                best_score = 1e9
                for i, p1 in enumerate(pls):
                    d_near = min(
                        (math.hypot(p1[4] - p2[4], p1[5] - p2[5]) for j, p2 in enumerate(pls) if i != j),
                        default=200.0
                    )
                    d_cont = math.hypot(p1[4] - prev_bx2, p1[5] - prev_by2)
                    score = d_near + 0.40 * d_cont
                    if score < best_score:
                        best_score = score
                        best_p = p1
                if best_p is not None:
                    f_ball[f] = (best_p[2], best_p[3], best_p[4], best_p[5])
                    prev_bx2, prev_by2 = best_p[4], best_p[5]
                    cur_carrier = best_p[0]
                    hold_until = f + 14

        return f_players, f_ball

    def _run_analysis(self):
        f_players, f_ball = self._build_frame_lookups()
        if not f_ball:
            return

        sorted_b_frames = sorted(f_ball.keys())

        # Precompute ball speed & direction change in 2D pitch space to reject aerial fly-overs
        ball_speed_2d: Dict[int, float] = {}
        ball_turn_deg: Dict[int, float] = {}
        for idx, f in enumerate(sorted_b_frames):
            _, _, x2, y2 = f_ball[f]
            if idx > 0:
                fp = sorted_b_frames[idx - 1]
                _, _, xp, yp = f_ball[fp]
                dt = max(1, f - fp)
                ball_speed_2d[f] = math.hypot(x2 - xp, y2 - yp) / dt
            else:
                ball_speed_2d[f] = 0.0

            if 0 < idx < len(sorted_b_frames) - 1:
                fp = sorted_b_frames[idx - 1]
                fn = sorted_b_frames[idx + 1]
                v1 = (x2 - f_ball[fp][2], y2 - f_ball[fp][3])
                v2 = (f_ball[fn][2] - x2, f_ball[fn][3] - y2)
                n1, n2 = math.hypot(*v1), math.hypot(*v2)
                if n1 > 2.0 and n2 > 2.0:
                    cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
                    ball_turn_deg[f] = math.degrees(math.acos(cos_a))
                else:
                    ball_turn_deg[f] = 180.0
            else:
                ball_turn_deg[f] = 180.0

        # 1. Raw frame-level candidate owner (nearest player within perspective control gate)
        raw_hits: List[Tuple[int, int, int, float, Tuple[float, float], Tuple[float, float]]] = []
        for f in sorted_b_frames:
            bx, by, bx2, by2 = f_ball[f]
            pls = f_players.get(f, [])
            best_hit = None
            best_norm = 1.0  # must be < 1.0 to qualify

            for pid, tid, px, py, px2, py2 in pls:
                d_cam = math.hypot(bx - px, by - py)
                d_2d_m = math.hypot((bx2 - px2) * self.scale_x, (by2 - py2) * self.scale_y)

                # Perspective camera control radius: ~38px at far touchline, ~82px at near touchline
                rad_cam = 38.0 + 44.0 * float(np.clip((py - 280.0) / 700.0, 0.0, 1.0))
                norm_d = min(d_cam / rad_cam, d_2d_m / 4.5)
                if norm_d < best_norm:
                    best_norm = norm_d
                    best_hit = (f, pid, tid, norm_d, (px, py), (px2, py2))

            if best_hit is not None:
                raw_hits.append(best_hit)

        # 2. Group consecutive/nearby hits by the same player into TouchSpells
        spells: List[TouchSpell] = []
        for f, pid, tid, norm_d, p_cam, p_2d in raw_hits:
            p_m = self._px2d_to_m(p_2d[0], p_2d[1])
            if spells and spells[-1].player_id == pid and (f - spells[-1].end_frame) <= 8:
                sp = spells[-1]
                sp.end_frame = f
                sp.end_cam = p_cam
                sp.end_2d = p_2d
                sp.end_m = p_m
                sp.min_norm_dist = min(sp.min_norm_dist, norm_d)
            else:
                spells.append(TouchSpell(
                    player_id=pid,
                    team_id=tid,
                    start_frame=f,
                    end_frame=f,
                    start_cam=p_cam,
                    end_cam=p_cam,
                    start_2d=p_2d,
                    end_2d=p_2d,
                    start_m=p_m,
                    end_m=p_m,
                    min_norm_dist=norm_d
                ))

        # 3. Kinematic aerial-flight filter: discard 1-2 frame spells where a fast ball
        #    simply flies straight past/over a player without turning or slowing down.
        validated_spells: List[TouchSpell] = []
        for sp in spells:
            dur = sp.end_frame - sp.start_frame + 1
            if dur >= 3:
                validated_spells.append(sp)
                continue
            spd = max(ball_speed_2d.get(sp.start_frame, 0.0), ball_speed_2d.get(sp.end_frame, 0.0))
            turn = max(ball_turn_deg.get(sp.start_frame, 0.0), ball_turn_deg.get(sp.end_frame, 0.0))
            if spd < 12.0 or turn >= 20.0 or sp.min_norm_dist <= 0.55:
                validated_spells.append(sp)

        # Merge adjacent spells of the SAME player if an eliminated aerial spell sat between them
        merged_spells: List[TouchSpell] = []
        for sp in validated_spells:
            if merged_spells and merged_spells[-1].player_id == sp.player_id and (sp.start_frame - merged_spells[-1].end_frame) <= 14:
                prev = merged_spells[-1]
                prev.end_frame = sp.end_frame
                prev.end_cam = sp.end_cam
                prev.end_2d = sp.end_2d
                prev.end_m = sp.end_m
                prev.min_norm_dist = min(prev.min_norm_dist, sp.min_norm_dist)
            else:
                merged_spells.append(sp)

        self.touch_spells = merged_spells

        # 4. Record player touches & frame-level team possession
        for sp in self.touch_spells:
            self.player_touches[sp.player_id] += 1
            self.player_pass_stats[sp.player_id]["touches"] += 1
            for f in range(sp.start_frame, sp.end_frame + 1):
                self.frame_team_owner[f] = sp.team_id
                self.frame_player_owner[f] = sp.player_id

        # Fill team possession across ball flight between consecutive spells
        for i in range(len(self.touch_spells) - 1):
            sa_sp = self.touch_spells[i]
            sb_sp = self.touch_spells[i + 1]
            if sa_sp.team_id == sb_sp.team_id and (sb_sp.start_frame - sa_sp.end_frame) <= 130:
                for f in range(sa_sp.end_frame + 1, sb_sp.start_frame):
                    self.frame_team_owner[f] = sa_sp.team_id
            elif sa_sp.team_id != sb_sp.team_id and (sb_sp.start_frame - sa_sp.end_frame) <= 60:
                # Outgoing pass stays with passing team until intercepted at sb_sp.start_frame
                for f in range(sa_sp.end_frame + 1, sb_sp.start_frame):
                    self.frame_team_owner[f] = sa_sp.team_id

        # 5. Extract Passes, Interceptions/Recoveries, Carries & Shots
        for i, sp in enumerate(self.touch_spells):
            # Check single-player Progressive Carry / Final-Third Carry / Box Carry
            carry_dx_att = self._attack_x(sp.end_m[0], sp.team_id) - self._attack_x(sp.start_m[0], sp.team_id)
            carry_dist = math.hypot(sp.end_m[0] - sp.start_m[0], sp.end_m[1] - sp.start_m[1])
            start_ax = self._attack_x(sp.start_m[0], sp.team_id)
            end_ax = self._attack_x(sp.end_m[0], sp.team_id)

            is_box_carry = (start_ax < 88.5 or not (13.8 <= sp.start_m[1] <= 54.2)) and (
                end_ax >= 88.5 and 13.8 <= sp.end_m[1] <= 54.2
            )
            is_ft_carry = (start_ax < 70.0) and (end_ax >= 70.0)

            if is_box_carry or is_ft_carry or (carry_dist >= 7.5 and carry_dx_att >= 5.5):
                c_type = (
                    "Penalty-Box Carry" if is_box_carry
                    else ("Final-Third Carry" if is_ft_carry else "Progressive Carry")
                )
                ev = {
                    "event_type": c_type,
                    "team_id": sp.team_id,
                    "player_id": sp.player_id,
                    "target_id": None,
                    "frame": sp.start_frame,
                    "end_frame": sp.end_frame,
                    "time_sec": round(sp.start_frame / self.fps, 1),
                    "start_2d": sp.start_2d,
                    "end_2d": sp.end_2d,
                    "start_m": sp.start_m,
                    "end_m": sp.end_m,
                    "distance_m": round(carry_dist, 1),
                    "zone": self._zone_name(sp.end_m[0], sp.team_id),
                }
                self.carries_and_entries.append(ev)
                self.all_events.append(ev)

            if i >= len(self.touch_spells) - 1:
                continue

            nxt = self.touch_spells[i + 1]
            gap = nxt.start_frame - sp.end_frame
            if gap < 0 or gap > 130:
                continue

            dist_m = math.hypot(nxt.start_m[0] - sp.end_m[0], nxt.start_m[1] - sp.end_m[1])

            # CASE A: Same team, different player -> Completed Pass
            if sp.team_id == nxt.team_id and sp.player_id != nxt.player_id and dist_m >= 2.0:
                sx_m, sy_m = sp.end_m
                ex_m, ey_m = nxt.start_m
                s_ax = self._attack_x(sx_m, sp.team_id)
                e_ax = self._attack_x(ex_m, sp.team_id)
                gain_x = max(e_ax - s_ax, abs(ex_m - sx_m) if (ex_m >= 72.0 or ex_m <= 33.0) else (e_ax - s_ax))

                start_in_box = (sx_m <= 16.5 or sx_m >= 88.5) and (13.8 <= sy_m <= 54.2)
                end_in_box = (ex_m <= 16.5 or ex_m >= 88.5) and (13.8 <= ey_m <= 54.2)
                is_box_entry = (not start_in_box) and end_in_box

                start_in_outer_third = (sx_m >= 70.0) or (sx_m <= 35.0)
                end_in_outer_third = (ex_m >= 70.0) or (ex_m <= 35.0)
                is_ft_entry = ((s_ax < 70.0) and (e_ax >= 70.0)) or ((not start_in_outer_third) and end_in_outer_third)

                is_cross = (
                    (sy_m <= 16.0 or sy_m >= 52.0)
                    and dist_m >= 11.0
                    and (ex_m >= 75.0 or ex_m <= 30.0)
                    and 16.0 <= ey_m <= 52.0
                )
                is_progressive = (
                    is_box_entry
                    or is_ft_entry
                    or (gain_x >= 8.0)
                    or (dist_m >= 14.0 and abs(ex_m - sx_m) >= 8.0)
                )

                subtype = "Standard Pass"
                if is_box_entry:
                    subtype = "Box Entry Pass"
                elif is_cross:
                    subtype = "Cross"
                elif is_ft_entry:
                    subtype = "Final-Third Entry Pass"
                elif is_progressive:
                    subtype = "Progressive Pass"

                pass_ev = {
                    "event_type": "Pass",
                    "subtype": subtype,
                    "team_id": sp.team_id,
                    "player_id": sp.player_id,
                    "target_id": nxt.player_id,
                    "frame": sp.end_frame,
                    "end_frame": nxt.start_frame,
                    "time_sec": round(sp.end_frame / self.fps, 1),
                    "start_2d": sp.end_2d,
                    "end_2d": nxt.start_2d,
                    "start_m": sp.end_m,
                    "end_m": nxt.start_m,
                    "distance_m": round(dist_m, 1),
                    "forward_gain_m": round(gain_x, 1),
                    "is_progressive": is_progressive,
                    "is_final_third_entry": is_ft_entry,
                    "is_box_entry": is_box_entry,
                    "is_cross": is_cross,
                    "zone": self._zone_name(nxt.start_m[0], sp.team_id),
                }
                self.passes.append(pass_ev)
                self.all_events.append(pass_ev)

                self.player_pass_stats[sp.player_id]["passes_completed"] += 1
                self.player_pass_stats[nxt.player_id]["passes_received"] += 1
                if is_progressive:
                    self.player_pass_stats[sp.player_id]["progressive_passes"] += 1

            # CASE B: Different team -> Interception / Ball Recovery (Turnover)
            elif sp.team_id != nxt.team_id and gap <= 115:
                rec_type = "Interception" if (gap <= 35 and dist_m >= 3.5) else "Ball Recovery"
                rec_ev = {
                    "event_type": rec_type,
                    "team_id": nxt.team_id,
                    "player_id": nxt.player_id,
                    "target_id": sp.player_id,  # opponent who lost possession
                    "frame": nxt.start_frame,
                    "end_frame": nxt.end_frame,
                    "time_sec": round(nxt.start_frame / self.fps, 1),
                    "start_2d": sp.end_2d,
                    "end_2d": nxt.start_2d,
                    "start_m": sp.end_m,
                    "end_m": nxt.start_m,
                    "distance_m": round(dist_m, 1),
                    "zone": self._zone_name(nxt.start_m[0], nxt.team_id),
                }
                self.interceptions.append(rec_ev)
                self.all_events.append(rec_ev)
                self.player_pass_stats[nxt.player_id]["recoveries"] += 1

        # 6. Check for Shot / Goal-Mouth Vectors from attacking third spells
        for sp in self.touch_spells:
            e_ax = self._attack_x(sp.end_m[0], sp.team_id)
            if e_ax < 72.0:
                continue
            # Look 3..12 frames ahead of sp.end_frame for a fast ball moving toward opponent goal mouth
            f0 = sp.end_frame
            if f0 not in f_ball:
                continue
            _, _, bx0, by0 = f_ball[f0]
            xm0, ym0 = self._px2d_to_m(bx0, by0)
            goal_x = 105.0 if self.attack_dir.get(sp.team_id, 1) == 1 else 0.0

            for f1 in range(f0 + 3, f0 + 14):
                if f1 not in f_ball:
                    continue
                _, _, bx1, by1 = f_ball[f1]
                xm1, ym1 = self._px2d_to_m(bx1, by1)
                dt_s = (f1 - f0) / self.fps
                vx = (xm1 - xm0) / dt_s
                vy = (ym1 - ym0) / dt_s
                spd_ms = math.hypot(vx, vy)
                # Moving fast toward opponent goal line
                toward_goal = (vx > 8.5) if goal_x == 105.0 else (vx < -8.5)
                if toward_goal and spd_ms >= 13.5:
                    t_goal = (goal_x - xm0) / vx if abs(vx) > 1e-3 else 999.0
                    if 0.0 < t_goal <= 3.0:
                        y_at_goal = ym0 + vy * t_goal
                        if 22.0 <= y_at_goal <= 46.0:
                            shot_ev = {
                                "event_type": "Shot / Goal Attack",
                                "team_id": sp.team_id,
                                "player_id": sp.player_id,
                                "target_id": None,
                                "frame": f0,
                                "end_frame": f1,
                                "time_sec": round(f0 / self.fps, 1),
                                "start_2d": (bx0, by0),
                                "end_2d": (bx1, by1),
                                "start_m": (xm0, ym0),
                                "end_m": (xm1, ym1),
                                "distance_m": round(math.hypot(xm1 - xm0, ym1 - ym0), 1),
                                "speed_kmh": round(spd_ms * 3.6, 1),
                                "zone": "Attacking Third",
                            }
                            self.shots.append(shot_ev)
                            self.all_events.append(shot_ev)
                            break

        self.all_events.sort(key=lambda e: e["frame"])

    # ------------------------------------------------------------------ public API

    def get_possession_summary(self) -> Dict:
        """Returns ball possession percentages, durations, touches, and timeline."""
        c0 = sum(1 for t in self.frame_team_owner.values() if t == 0)
        c1 = sum(1 for t in self.frame_team_owner.values() if t == 1)
        total_owned = c0 + c1

        if total_owned == 0:
            t0_touches = sum(v for pid, v in self.player_touches.items() if self.sa.player_teams.get(pid) == 0)
            t1_touches = sum(v for pid, v in self.player_touches.items() if self.sa.player_teams.get(pid) == 1)
            tot_t = max(1, t0_touches + t1_touches)
            pct0 = round((t0_touches / tot_t) * 100.0, 1) if (t0_touches + t1_touches) > 0 else 50.0
            pct1 = round(100.0 - pct0, 1)
        else:
            pct0 = round((c0 / total_owned) * 100.0, 1)
            pct1 = round(100.0 - pct0, 1)

        t0_passes = [p for p in self.passes if p["team_id"] == 0]
        t1_passes = [p for p in self.passes if p["team_id"] == 1]
        t0_prog = [p for p in t0_passes if p["is_progressive"]]
        t1_prog = [p for p in t1_passes if p["is_progressive"]]
        t0_rec = [r for r in self.interceptions if r["team_id"] == 0]
        t1_rec = [r for r in self.interceptions if r["team_id"] == 1]

        t0_entries = sum(1 for p in t0_passes if p["is_final_third_entry"] or p["is_box_entry"]) + sum(
            1 for c in self.carries_and_entries if c["team_id"] == 0
        )
        t1_entries = sum(1 for p in t1_passes if p["is_final_third_entry"] or p["is_box_entry"]) + sum(
            1 for c in self.carries_and_entries if c["team_id"] == 1
        )

        # Build per-second possession timeline for visual bar
        total_sec = max(1, int(math.ceil(self.sa.total_processed_frames / self.fps)))
        timeline = []
        for s in range(total_sec):
            f_start = int(s * self.fps)
            f_end = int((s + 1) * self.fps)
            s0 = sum(1 for f in range(f_start, f_end) if self.frame_team_owner.get(f) == 0)
            s1 = sum(1 for f in range(f_start, f_end) if self.frame_team_owner.get(f) == 1)
            owner = 0 if s0 > s1 else (1 if s1 > s0 else -1)
            timeline.append({"second": s + 1, "team_id": owner})

        return {
            "team0_possession_pct": pct0,
            "team1_possession_pct": pct1,
            "team0_possession_sec": round(c0 / self.fps, 1),
            "team1_possession_sec": round(c1 / self.fps, 1),
            "team0_touches": sum(sp.team_id == 0 for sp in self.touch_spells),
            "team1_touches": sum(sp.team_id == 1 for sp in self.touch_spells),
            "team0_passes": len(t0_passes),
            "team1_passes": len(t1_passes),
            "team0_progressive_passes": len(t0_prog),
            "team1_progressive_passes": len(t1_prog),
            "team0_recoveries": len(t0_rec),
            "team1_recoveries": len(t1_rec),
            "team0_attacking_entries": t0_entries,
            "team1_attacking_entries": t1_entries,
            "total_passes": len(self.passes),
            "total_recoveries": len(self.interceptions),
            "total_shots": len(self.shots),
            "timeline": timeline,
        }

    def get_passing_network(self, team_id: int) -> Dict:
        """
        Builds nodes and weighted edges for a team's 2D Passing Network.
        Returns:
            {
                "nodes": {pid: {"pos_2d": (x, y), "touches": int, "passes_made": int, "passes_received": int}},
                "edges": [
                    {"pair": (pid_a, pid_b), "count": int, "a_to_b": int, "b_to_a": int, "progressive": int}
                ]
            }
        """
        avg_pos_2d = self.sa.get_average_positions_2d(min_frames=1, max_per_team=11)
        nodes = {}
        for pid, pos in avg_pos_2d.items():
            if self.sa.player_teams.get(pid) != team_id:
                continue
            st = self.player_pass_stats[pid]
            nodes[pid] = {
                "pos_2d": pos,
                "touches": max(1, st["touches"]),
                "passes_made": st["passes_completed"],
                "passes_received": st["passes_received"],
                "progressive_passes": st["progressive_passes"],
                "recoveries": st["recoveries"],
            }

        pair_counts: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(
            lambda: {"count": 0, "a_to_b": 0, "b_to_a": 0, "progressive": 0}
        )
        for p in self.passes:
            if p["team_id"] != team_id:
                continue
            u, v = int(p["player_id"]), int(p["target_id"])
            if u not in nodes or v not in nodes or u == v:
                continue
            pair = (min(u, v), max(u, v))
            pair_counts[pair]["count"] += 1
            if u == pair[0]:
                pair_counts[pair]["a_to_b"] += 1
            else:
                pair_counts[pair]["b_to_a"] += 1
            if p["is_progressive"]:
                pair_counts[pair]["progressive"] += 1

        edges = [
            {
                "pair": pair,
                "count": info["count"],
                "a_to_b": info["a_to_b"],
                "b_to_a": info["b_to_a"],
                "progressive": info["progressive"],
            }
            for pair, info in sorted(pair_counts.items(), key=lambda kv: -kv[1]["count"])
        ]

        return {"nodes": nodes, "edges": edges}

    def get_event_log_table(self) -> List[Dict]:
        """Returns a formatted list of match events suitable for a Streamlit dataframe."""
        rows = []
        for ev in self.all_events:
            tid = ev["team_id"]
            t_label = f"Team {tid} ({'Home' if tid == 0 else 'Away'})"
            etype = ev["event_type"]
            pid = ev["player_id"]
            tgt = ev.get("target_id")

            if etype == "Pass":
                detail = f"#{pid} → #{tgt} ({ev['subtype']} • {ev['distance_m']}m)"
            elif etype in ("Interception", "Ball Recovery"):
                detail = f"#{pid} won ball from #{tgt}" if tgt else f"#{pid} recovered possession"
            elif etype == "Shot / Goal Attack":
                detail = f"#{pid} goal-bound strike ({ev.get('speed_kmh', 0)} km/h)"
            else:
                detail = f"#{pid} carried {ev['distance_m']}m"

            rows.append({
                "Time (s)": f"{ev['time_sec']:.1f}s",
                "Frame": ev["frame"],
                "Team": t_label,
                "Event": ev.get("subtype") if etype == "Pass" else etype,
                "Player": f"#{pid}",
                "Target / Opponent": f"#{tgt}" if tgt else "—",
                "Distance (m)": ev["distance_m"],
                "Pitch Zone": ev["zone"],
                "Description": detail,
            })
        return rows
