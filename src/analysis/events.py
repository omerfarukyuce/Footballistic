"""
Phase 2: Football Pass Analysis, Individual Pass Freeze-Frames, Ball Possession & Match Events.

Extracts accurate, physics-verified pass telemetry and tactical events from 2D player
trajectories and GMC-compensated ball tracking:
  1. Kinematic Aerial Fly-Over Rejection:
     Filters out players standing underneath a fast straight aerial pass so long
     switches (e.g. 45m cross-field balls) are NOT falsely broken into mid-flight touches.
  2. Exact Kick & Reception Frame Timing:
     Pinpoints the exact frame the ball leaves the passer's foot (`kick_frame`) and
     the exact frame it reaches the receiver/interceptor (`receive_frame`), giving
     physically accurate pass duration (s), pass length (m on the 105m x 68m FIFA
     perspective-calibrated pitch), average ball speed (km/h), and peak speed (km/h).
  3. Complete Pass Catalogue (Completed + Incomplete / Intercepted):
     Every pass stores:
       - `pass_number`, `team_id`, `passer_id`, `target_id` (receiver or interceptor)
       - `is_completed` (True / False), `outcome` ("Completed" / "Incomplete")
       - `kick_frame`, `receive_frame`, `time_sec`, `duration_sec`
       - `length_m`, `arc_length_m`, `avg_speed_kmh`, `avg_speed_ms`, `peak_speed_kmh`
       - `ball_path_2d` (actual frame-by-frame 2D trajectory showing HOW the ball moved)
       - `players_at_kick` (exact snapshot of ALL teammates & opponents at `kick_frame`)
       - `receiver_pos_at_receive` (where the receiver/interceptor met the ball)
       - `subtype` ("Long Switch", "Box Entry", "Final-Third Entry", "Progressive Pass",
         "Cross", "Short Pass", "Intercepted Pass")
  4. Dynamic Ball Possession (%) & Timeline + Passing Network Graph Data.
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
    """Computes individual pass freeze-frames, completed & incomplete passes, speeds, lengths, and possession."""

    def __init__(self, spatial_analytics: SpatialAnalytics, fps: float = 25.0):
        self.sa = spatial_analytics
        self.fps = max(1.0, float(fps))
        self.scale_x = self.sa.PITCH_LENGTH_M / max(1.0, self.sa.FIELD_W)
        self.scale_y = self.sa.PITCH_WIDTH_M / max(1.0, self.sa.FIELD_H)

        # Attacking direction per team: +1 means attacking left->right (X=105m), -1 means right->left (X=0m)
        self.attack_dir: Dict[int, int] = self._infer_attack_directions()

        # Internal lookups retained for freeze-frame snapshots
        self._f_players: Dict[int, List[Tuple[int, int, float, float, float, float]]] = {}
        self._f_ball: Dict[int, Tuple[float, float, float, float]] = {}

        # Computed outputs
        self.touch_spells: List[TouchSpell] = []
        self.frame_team_owner: Dict[int, int] = {}
        self.frame_player_owner: Dict[int, int] = {}
        self.all_passes: List[Dict] = []       # BOTH completed AND incomplete/intercepted passes
        self.passes: List[Dict] = []           # Completed passes only (for Passing Network & compatibility)
        self.incomplete_passes: List[Dict] = []
        self.interceptions: List[Dict] = []
        self.carries_and_entries: List[Dict] = []
        self.shots: List[Dict] = []
        self.all_events: List[Dict] = []
        self.player_touches: Dict[int, int] = defaultdict(int)
        self.player_pass_stats: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {
                "touches": 0,
                "passes_completed": 0,
                "passes_incomplete": 0,
                "passes_received": 0,
                "progressive_passes": 0,
                "recoveries": 0,
            }
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
            return {0: 1, 1: -1}
        return {0: -1, 1: 1}

    def _attack_x(self, xm: float, team_id: int) -> float:
        return xm if self.attack_dir.get(team_id, 1) == 1 else (105.0 - xm)

    def _zone_name(self, xm: float, team_id: int) -> str:
        ax = self._attack_x(xm, team_id)
        if ax < 35.0:
            return "Defensive Third"
        elif ax < 70.0:
            return "Middle Third"
        return "Attacking Third"

    def _snapshot_players_at_frame(self, frame_idx: int) -> Dict[int, Dict]:
        """
        Captures the exact 2D pitch coordinates of ALL active players (teammates & opponents)
        at `frame_idx`. If a player is missing on that exact frame (e.g. off-screen GK),
        falls back to their nearest recorded frame within +/- 25 frames or average position.
        """
        snap: Dict[int, Dict] = {}
        gk_set = set(self.sa.goalkeeper_ids) | {1, 13}

        for pid, tid in self.sa.player_teams.items():
            if tid not in (0, 1):
                continue
            frames = self.sa.player_frames.get(pid, [])
            tr_2d = self.sa.player_tracks_2d.get(pid, [])
            tr_cam = self.sa.player_tracks.get(pid, [])
            if not frames or not tr_2d:
                continue

            # Binary search / closest frame lookup
            arr_f = np.asarray(frames, dtype=np.int32)
            idx = int(np.argmin(np.abs(arr_f - frame_idx)))
            f_near = int(arr_f[idx])
            is_synth = (
                idx < len(tr_cam)
                and abs(tr_cam[idx][0] - 100.0) < 1e-3
                and abs(tr_cam[idx][1] - 100.0) < 1e-3
            )

            if abs(f_near - frame_idx) <= 30 or is_synth:
                pos = (float(tr_2d[idx][0]), float(tr_2d[idx][1]))
                snap[int(pid)] = {
                    "team_id": int(tid),
                    "pos_2d": pos,
                    "is_gk": (int(pid) in gk_set),
                    "on_screen": not is_synth and abs(f_near - frame_idx) <= 8,
                }
        return snap

    def _get_player_pos_near(self, pid: int, frame_idx: int) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Returns ((cam_x, cam_y), (x2d, y2d)) for `pid` at or closest to `frame_idx`."""
        frames = self.sa.player_frames.get(pid, [])
        tr_cam = self.sa.player_tracks.get(pid, [])
        tr_2d = self.sa.player_tracks_2d.get(pid, [])
        if not frames or not tr_2d:
            return None
        arr_f = np.asarray(frames, dtype=np.int32)
        idx = int(np.argmin(np.abs(arr_f - frame_idx)))
        if abs(int(arr_f[idx]) - frame_idx) > 12:
            return None
        return (float(tr_cam[idx][0]), float(tr_cam[idx][1])), (float(tr_2d[idx][0]), float(tr_2d[idx][1]))

    def _refine_kick_and_receive_frames(
        self,
        sp_passer: TouchSpell,
        sp_receiver: TouchSpell
    ) -> Tuple[int, int]:
        """
        Pinpoints the exact `kick_frame` (last closest approach of ball to passer's foot
        right before departure) and `receive_frame` (first closest approach to receiver's
        foot upon arrival). Eliminates halo-overlap distortion so ball speed (km/h) is
        physically accurate!
        """
        # 1. Refine kick_frame inside [sp_passer.start_frame .. sp_passer.end_frame]
        kick_f = sp_passer.end_frame
        p_frames = [
            f for f in range(max(sp_passer.start_frame, sp_passer.end_frame - 25), sp_passer.end_frame + 1)
            if f in self._f_ball
        ]
        if len(p_frames) >= 2:
            # Walk backward while ball was moving away from passer's foot
            best_f = p_frames[-1]
            best_d = 1e9
            for f in reversed(p_frames):
                p_info = self._get_player_pos_near(sp_passer.player_id, f)
                if p_info is None:
                    continue
                bx, by, _, _ = self._f_ball[f]
                d = math.hypot(bx - p_info[0][0], by - p_info[0][1])
                if d <= best_d + 2.5:
                    best_d = min(best_d, d)
                    best_f = f
                else:
                    # Distance started increasing again going backward -> we passed the kick local min!
                    if best_f < p_frames[-1]:
                        break
            kick_f = best_f

        # 1b. If there is a long gap after sp_passer.end_frame where the ball sits nearly
        # stationary near the passer before actually launching (e.g. set-piece / slow hold),
        # advance kick_f to the true launch moment so flight duration & speed stay accurate!
        if (sp_receiver.start_frame - sp_passer.end_frame) > 40:
            scan_f = [
                f for f in range(sp_passer.end_frame, sp_receiver.start_frame)
                if f in self._f_ball
            ]
            last_slow_f = None
            for idx_s in range(1, len(scan_f)):
                f_prev, f_cur = scan_f[idx_s - 1], scan_f[idx_s]
                bx0, by0, _, _ = self._f_ball[f_prev]
                bx1, by1, _, _ = self._f_ball[f_cur]
                spd = math.hypot(bx1 - bx0, by1 - by0) / max(1, f_cur - f_prev)
                d_from_passer = math.hypot(bx1 - sp_passer.end_cam[0], by1 - sp_passer.end_cam[1])
                if spd < 6.5 and d_from_passer < 140.0:
                    last_slow_f = f_cur
                elif spd >= 11.0 and last_slow_f is not None:
                    kick_f = last_slow_f
                    break

        # 2. Refine receive_frame inside [sp_receiver.start_frame .. sp_receiver.end_frame]
        recv_f = sp_receiver.start_frame
        r_frames = [
            f for f in range(sp_receiver.start_frame, min(sp_receiver.end_frame, sp_receiver.start_frame + 25) + 1)
            if f in self._f_ball
        ]
        if len(r_frames) >= 2:
            best_f = r_frames[0]
            best_d = 1e9
            for f in r_frames:
                p_info = self._get_player_pos_near(sp_receiver.player_id, f)
                if p_info is None:
                    continue
                bx, by, _, _ = self._f_ball[f]
                d = math.hypot(bx - p_info[0][0], by - p_info[0][1])
                if d <= best_d + 2.5:
                    best_d = min(best_d, d)
                    best_f = f
                else:
                    if best_f > r_frames[0]:
                        break
            recv_f = best_f

        if recv_f <= kick_f:
            recv_f = min(sp_receiver.end_frame, kick_f + max(2, sp_receiver.start_frame - sp_passer.end_frame))
        return kick_f, recv_f

    # ------------------------------------------------------------------ core analysis

    def _build_frame_lookups(self):
        f_players: Dict[int, List[Tuple[int, int, float, float, float, float]]] = defaultdict(list)

        for pid, frames in self.sa.player_frames.items():
            tid = self.sa.player_teams.get(pid, -1)
            if tid not in (0, 1):
                continue
            tr_cam = self.sa.player_tracks.get(pid, [])
            tr_2d = self.sa.player_tracks_2d.get(pid, [])
            for f, p_cam, p_2d in zip(frames, tr_cam, tr_2d):
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
                min_cam_d = min((math.hypot(bx - p[2], by - p[3]) for p in pls), default=999.0)
                if min_cam_d > 38.0:
                    continue

            f_ball[f] = (bx, by, bx2, by2)

        # Fallback for legacy spatial_data.json files without saved ball_track
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

    def _is_aerial_flyover_spell(
        self,
        sp: TouchSpell,
        spells: List[TouchSpell],
        sp_idx: int
    ) -> bool:
        """
        Detects when a fast aerial/ground ball simply passes over or past a bystander
        player's 2D projection without genuine touch/control:
          - Never drops the clip-opening (idx=0) or clip-closing (idx=last) spell.
          - Short intermediate spells (<= 12 frames) where the ball NEVER slows down
            under player control (`min_step_spd >= 10.5 px/frame`) and either:
              (A) continues straight before & after (`turn_deg < 28°`),
              (B) launches straight past a nearby defender right off the passer's foot
                  (`spd_mid >= 11`, `spd_out >= 11`, `turn_deg(v_mid, v_out) < 26°`), or
              (C) glances past a defender (`dur <= 4`, `min_step_spd >= 12`) within 3
                  frames of the real receiver trapping it cleanly (`next_dur >= 8`).
        """
        if sp_idx <= 0 or sp_idx >= len(spells) - 1:
            return False

        dur = sp.end_frame - sp.start_frame + 1
        if dur > 12:
            return False

        # Frame-to-frame camera speed INSIDE this spell
        in_frames = [f for f in range(max(sp.start_frame - 1, 0), sp.end_frame + 1) if f in self._f_ball]
        step_spds: List[float] = []
        for k in range(1, len(in_frames)):
            f0, f1 = in_frames[k - 1], in_frames[k]
            if f1 < sp.start_frame:
                continue
            p0, p1 = self._f_ball[f0], self._f_ball[f1]
            step_spds.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / max(1, f1 - f0))

        min_step_spd = min(step_spds) if step_spds else 0.0

        f_before = [f for f in range(sp.start_frame - 6, sp.start_frame) if f in self._f_ball]
        f_after = [f for f in range(sp.end_frame + 1, sp.end_frame + 7) if f in self._f_ball]

        def _vec_angle_deg(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return 0.0
            c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
            return math.degrees(math.acos(c))

        if sp.start_frame not in self._f_ball or sp.end_frame not in self._f_ball:
            return False

        p_s = self._f_ball[sp.start_frame]
        p_e = self._f_ball[sp.end_frame]
        v_mid = (
            (p_e[0] - p_s[0]) / max(1, sp.end_frame - sp.start_frame),
            (p_e[1] - p_s[1]) / max(1, sp.end_frame - sp.start_frame),
        )
        spd_mid = math.hypot(*v_mid) if dur > 1 else min_step_spd

        v_in = None
        spd_in = 0.0
        if f_before:
            fb = f_before[0]
            p_b = self._f_ball[fb]
            v_in = ((p_s[0] - p_b[0]) / max(1, sp.start_frame - fb), (p_s[1] - p_b[1]) / max(1, sp.start_frame - fb))
            spd_in = math.hypot(*v_in)

        v_out = None
        spd_out = 0.0
        if f_after:
            fa = f_after[-1]
            p_a = self._f_ball[fa]
            v_out = ((p_a[0] - p_e[0]) / max(1, fa - sp.end_frame), (p_a[1] - p_e[1]) / max(1, fa - sp.end_frame))
            spd_out = math.hypot(*v_out)

        # Case A: Classic straight continuous high-speed flight across a bystander
        if v_in is not None and v_out is not None and spd_in >= 10.0 and spd_out >= 10.0 and spd_mid >= 9.0:
            turn_deg = _vec_angle_deg(v_in, v_out)
            if turn_deg < 28.0 and (spd_out / max(1.0, spd_in)) >= 0.55:
                return True

        # Case B: Bystander right off the passer's foot — ball never slows down & keeps flying straight
        if min_step_spd >= 10.5 and v_out is not None and spd_mid >= 11.0 and spd_out >= 11.0:
            if dur == 1 or _vec_angle_deg(v_mid, v_out) < 26.0:
                return True

        # Case C: Glancing fly-past right beside the actual receiver who traps it immediately after
        nxt_sp = spells[sp_idx + 1]
        nxt_dur = nxt_sp.end_frame - nxt_sp.start_frame + 1
        if (
            dur <= 4
            and min_step_spd >= 12.0
            and spd_in >= 10.0
            and (nxt_sp.start_frame - sp.end_frame) <= 3
            and nxt_dur >= 8
        ):
            return True

        return False

    def _run_analysis(self):
        self._f_players, self._f_ball = self._build_frame_lookups()
        if not self._f_ball:
            return

        sorted_b_frames = sorted(self._f_ball.keys())

        # 1. Raw frame-level candidate owner
        raw_hits: List[Tuple[int, int, int, float, Tuple[float, float], Tuple[float, float]]] = []
        for f in sorted_b_frames:
            bx, by, bx2, by2 = self._f_ball[f]
            pls = self._f_players.get(f, [])
            best_hit = None
            best_norm = 1.0

            for pid, tid, px, py, px2, py2 in pls:
                d_cam = math.hypot(bx - px, by - py)
                d_2d_m = math.hypot((bx2 - px2) * self.scale_x, (by2 - py2) * self.scale_y)

                rad_cam = 36.0 + 44.0 * float(np.clip((py - 280.0) / 700.0, 0.0, 1.0))
                norm_d = min(d_cam / rad_cam, d_2d_m / 4.2)
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

        # 3a. Genuine Foot-Contact Gate: discard distant proximity halos (> 0.62 norm radius)
        # where the ball never came within realistic foot-control reach (~2.5m).
        contact_spells: List[TouchSpell] = [
            sp for sp in spells
            if sp.min_norm_dist <= 0.62
        ]

        # 3b. Kinematic aerial-flight filter: discard bystanders underneath fast flights!
        validated_spells: List[TouchSpell] = [
            sp for idx_sp, sp in enumerate(contact_spells)
            if not self._is_aerial_flyover_spell(sp, contact_spells, idx_sp)
        ]

        # Merge adjacent spells of the SAME player if a fly-over noise spell sat between them
        merged_spells: List[TouchSpell] = []
        for sp in validated_spells:
            if (
                merged_spells
                and merged_spells[-1].player_id == sp.player_id
                and (sp.start_frame - merged_spells[-1].end_frame) <= 45
            ):
                prev = merged_spells[-1]
                prev.end_frame = sp.end_frame
                prev.end_cam = sp.end_cam
                prev.end_2d = sp.end_2d
                prev.end_m = sp.end_m
                prev.min_norm_dist = min(prev.min_norm_dist, sp.min_norm_dist)
            else:
                merged_spells.append(sp)

        self.touch_spells = merged_spells

        # 4. Record touches & continuous frame-level team possession
        for sp in self.touch_spells:
            self.player_touches[sp.player_id] += 1
            self.player_pass_stats[sp.player_id]["touches"] += 1
            for f in range(sp.start_frame, sp.end_frame + 1):
                self.frame_team_owner[f] = sp.team_id
                self.frame_player_owner[f] = sp.player_id

        for i in range(len(self.touch_spells) - 1):
            sa_sp = self.touch_spells[i]
            sb_sp = self.touch_spells[i + 1]
            gap = sb_sp.start_frame - sa_sp.end_frame
            if sa_sp.team_id == sb_sp.team_id and gap <= 135:
                for f in range(sa_sp.end_frame + 1, sb_sp.start_frame):
                    self.frame_team_owner[f] = sa_sp.team_id
            elif sa_sp.team_id != sb_sp.team_id and gap <= 75:
                for f in range(sa_sp.end_frame + 1, sb_sp.start_frame):
                    self.frame_team_owner[f] = sa_sp.team_id

        # 5. Extract Completed Passes, Incomplete/Intercepted Passes, Recoveries & Carries
        pass_counter = 0

        for i, sp in enumerate(self.touch_spells):
            # Single-player Progressive / Box Carry check
            carry_dx_att = self._attack_x(sp.end_m[0], sp.team_id) - self._attack_x(sp.start_m[0], sp.team_id)
            carry_dist = math.hypot(sp.end_m[0] - sp.start_m[0], sp.end_m[1] - sp.start_m[1])
            start_ax = self._attack_x(sp.start_m[0], sp.team_id)
            end_ax = self._attack_x(sp.end_m[0], sp.team_id)

            is_box_carry = (start_ax < 88.5 or not (13.8 <= sp.start_m[1] <= 54.2)) and (
                end_ax >= 88.5 and 13.8 <= sp.end_m[1] <= 54.2
            )
            is_ft_carry = (start_ax < 70.0) and (end_ax >= 70.0)

            if is_box_carry or is_ft_carry or (carry_dist >= 8.0 and carry_dx_att >= 6.0):
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
            raw_gap = nxt.start_frame - sp.end_frame
            if raw_gap < 0 or raw_gap > 135 or sp.player_id == nxt.player_id:
                continue

            # Refine exact kick & reception frames for physically accurate speed & trajectory!
            kick_f, recv_f = self._refine_kick_and_receive_frames(sp, nxt)
            flight_frames = max(2, recv_f - kick_f)
            duration_sec = round(flight_frames / self.fps, 2)

            # Extract actual frame-by-frame 2D ball flight trajectory from kick_f to recv_f
            ball_path_2d: List[Tuple[float, float]] = [
                (self._f_ball[f][2], self._f_ball[f][3])
                for f in range(kick_f, recv_f + 1)
                if f in self._f_ball
            ]
            # Anchor start & end to exact player positions at kick_f / recv_f if ball missing on endpoint
            p_kick_info = self._get_player_pos_near(sp.player_id, kick_f)
            p_recv_info = self._get_player_pos_near(nxt.player_id, recv_f)
            start_2d = p_kick_info[1] if p_kick_info else sp.end_2d
            end_2d = p_recv_info[1] if p_recv_info else nxt.start_2d

            if len(ball_path_2d) < 2:
                ball_path_2d = [start_2d, end_2d]
            else:
                # Gently connect start_2d -> ball_path -> end_2d if endpoints are close
                if math.hypot(ball_path_2d[0][0] - start_2d[0], ball_path_2d[0][1] - start_2d[1]) > 4.0:
                    ball_path_2d = [start_2d] + ball_path_2d
                if math.hypot(ball_path_2d[-1][0] - end_2d[0], ball_path_2d[-1][1] - end_2d[1]) > 4.0:
                    ball_path_2d = ball_path_2d + [end_2d]

            start_m = self._px2d_to_m(start_2d[0], start_2d[1])
            end_m = self._px2d_to_m(end_2d[0], end_2d[1])
            length_m = round(math.hypot(end_m[0] - start_m[0], end_m[1] - start_m[1]), 1)

            if length_m < 2.5:
                # Too short to be a pass (close tackle / shoulder duel); still log recovery if opp team
                if sp.team_id != nxt.team_id:
                    rec_ev = {
                        "event_type": "Ball Recovery",
                        "team_id": nxt.team_id,
                        "player_id": nxt.player_id,
                        "target_id": sp.player_id,
                        "frame": recv_f,
                        "end_frame": nxt.end_frame,
                        "time_sec": round(recv_f / self.fps, 1),
                        "start_2d": start_2d,
                        "end_2d": end_2d,
                        "start_m": start_m,
                        "end_m": end_m,
                        "distance_m": length_m,
                        "zone": self._zone_name(end_m[0], nxt.team_id),
                    }
                    self.interceptions.append(rec_ev)
                    self.all_events.append(rec_ev)
                    self.player_pass_stats[nxt.player_id]["recoveries"] += 1
                continue

            # Compute true arc length & instantaneous peak speed along ball_path_2d
            arc_m = 0.0
            step_speeds_kmh: List[float] = []
            b_frames_in_pass = [f for f in range(kick_f, recv_f + 1) if f in self._f_ball]
            for k in range(1, len(b_frames_in_pass)):
                fa, fb = b_frames_in_pass[k - 1], b_frames_in_pass[k]
                ma = self._px2d_to_m(self._f_ball[fa][2], self._f_ball[fa][3])
                mb = self._px2d_to_m(self._f_ball[fb][2], self._f_ball[fb][3])
                seg_m = math.hypot(mb[0] - ma[0], mb[1] - ma[1])
                dt_seg = (fb - fa) / self.fps
                arc_m += seg_m
                if dt_seg > 0:
                    sp_k = (seg_m / dt_seg) * 3.6
                    if sp_k <= 135.0:  # reject single-frame tracking jitter spikes
                        step_speeds_kmh.append(sp_k)

            arc_length_m = round(max(length_m, arc_m), 1)
            avg_speed_ms = round(length_m / max(0.12, duration_sec), 1)
            avg_speed_kmh = round(float(np.clip(avg_speed_ms * 3.6, 8.0, 118.0)), 1)
            peak_speed_kmh = round(
                float(np.clip(np.percentile(step_speeds_kmh, 85), avg_speed_kmh, 128.0))
                if step_speeds_kmh else avg_speed_kmh,
                1
            )

            # Classify tactical pass attributes
            sx_m, sy_m = start_m
            ex_m, ey_m = end_m
            s_ax = self._attack_x(sx_m, sp.team_id)
            e_ax = self._attack_x(ex_m, sp.team_id)
            gain_x = round(e_ax - s_ax, 1)

            start_in_box = (sx_m <= 16.5 or sx_m >= 88.5) and (13.8 <= sy_m <= 54.2)
            end_in_box = (ex_m <= 16.5 or ex_m >= 88.5) and (13.8 <= ey_m <= 54.2)
            is_box_entry = (not start_in_box) and end_in_box
            is_ft_entry = (s_ax < 70.0) and (e_ax >= 70.0)
            is_long_switch = (length_m >= 30.0) and (abs(ey_m - sy_m) >= 18.0)
            is_cross = (
                (sy_m <= 16.0 or sy_m >= 52.0)
                and length_m >= 12.0
                and (ex_m >= 75.0 or ex_m <= 30.0)
                and 16.0 <= ey_m <= 52.0
            )
            is_progressive = is_box_entry or is_ft_entry or (gain_x >= 8.0) or (length_m >= 22.0 and gain_x >= 5.0)

            is_completed = (sp.team_id == nxt.team_id)
            pass_counter += 1

            if is_completed:
                if is_box_entry:
                    subtype = "Box Entry Pass"
                elif is_long_switch:
                    subtype = "Long Switch Pass"
                elif is_cross:
                    subtype = "Cross"
                elif is_ft_entry:
                    subtype = "Final-Third Entry"
                elif is_progressive:
                    subtype = "Progressive Pass"
                elif length_m >= 22.0:
                    subtype = "Long Pass"
                else:
                    subtype = "Short Pass"
            else:
                subtype = "Intercepted Pass"

            # Exact snapshot of ALL 22 players at the exact kick moment!
            players_at_kick = self._snapshot_players_at_frame(kick_f)

            pass_ev = {
                "pass_number": pass_counter,
                "event_type": "Pass",
                "outcome": "Completed" if is_completed else "Incomplete",
                "is_completed": is_completed,
                "subtype": subtype,
                "team_id": sp.team_id,
                "receiver_team_id": nxt.team_id,
                "player_id": sp.player_id,
                "target_id": nxt.player_id,
                "frame": kick_f,
                "end_frame": recv_f,
                "mid_frame": (kick_f + recv_f) // 2,
                "time_sec": round(kick_f / self.fps, 1),
                "end_time_sec": round(recv_f / self.fps, 1),
                "duration_sec": duration_sec,
                "start_2d": start_2d,
                "end_2d": end_2d,
                "start_m": start_m,
                "end_m": end_m,
                "distance_m": length_m,
                "length_m": length_m,
                "arc_length_m": arc_length_m,
                "avg_speed_ms": avg_speed_ms,
                "avg_speed_kmh": avg_speed_kmh,
                "peak_speed_kmh": peak_speed_kmh,
                "forward_gain_m": gain_x,
                "is_progressive": is_progressive if is_completed else False,
                "is_final_third_entry": is_ft_entry if is_completed else False,
                "is_box_entry": is_box_entry if is_completed else False,
                "is_cross": is_cross if is_completed else False,
                "ball_path_2d": ball_path_2d,
                "players_at_kick": players_at_kick,
                "receiver_pos_at_receive": end_2d,
                "start_zone": self._zone_name(sx_m, sp.team_id),
                "zone": self._zone_name(ex_m, sp.team_id),
            }

            self.all_passes.append(pass_ev)
            self.all_events.append(pass_ev)

            if is_completed:
                self.passes.append(pass_ev)
                self.player_pass_stats[sp.player_id]["passes_completed"] += 1
                self.player_pass_stats[nxt.player_id]["passes_received"] += 1
                if is_progressive:
                    self.player_pass_stats[sp.player_id]["progressive_passes"] += 1
            else:
                self.incomplete_passes.append(pass_ev)
                self.player_pass_stats[sp.player_id]["passes_incomplete"] += 1
                # Also credit an Interception / Ball Recovery to the opponent!
                rec_ev = {
                    "event_type": "Interception",
                    "team_id": nxt.team_id,
                    "player_id": nxt.player_id,
                    "target_id": sp.player_id,
                    "frame": recv_f,
                    "end_frame": nxt.end_frame,
                    "time_sec": round(recv_f / self.fps, 1),
                    "start_2d": start_2d,
                    "end_2d": end_2d,
                    "start_m": start_m,
                    "end_m": end_m,
                    "distance_m": length_m,
                    "zone": self._zone_name(ex_m, nxt.team_id),
                }
                self.interceptions.append(rec_ev)
                self.all_events.append(rec_ev)
                self.player_pass_stats[nxt.player_id]["recoveries"] += 1

        self.all_events.sort(key=lambda e: e["frame"])

    # ------------------------------------------------------------------ public API

    def get_possession_summary(self) -> Dict:
        """Returns possession %, pass accuracy, avg pass length & speed, and timeline."""
        c0 = sum(1 for t in self.frame_team_owner.values() if t == 0)
        c1 = sum(1 for t in self.frame_team_owner.values() if t == 1)
        total_owned = c0 + c1

        if total_owned == 0:
            pct0, pct1 = 50.0, 50.0
        else:
            pct0 = round((c0 / total_owned) * 100.0, 1)
            pct1 = round(100.0 - pct0, 1)

        t0_all = [p for p in self.all_passes if p["team_id"] == 0]
        t1_all = [p for p in self.all_passes if p["team_id"] == 1]
        t0_ok = [p for p in t0_all if p["is_completed"]]
        t1_ok = [p for p in t1_all if p["is_completed"]]

        t0_acc = round((len(t0_ok) / len(t0_all)) * 100.0, 1) if t0_all else 0.0
        t1_acc = round((len(t1_ok) / len(t1_all)) * 100.0, 1) if t1_all else 0.0
        tot_acc = round((len(self.passes) / len(self.all_passes)) * 100.0, 1) if self.all_passes else 0.0

        avg_len = round(float(np.mean([p["length_m"] for p in self.all_passes])), 1) if self.all_passes else 0.0
        avg_spd = round(float(np.mean([p["avg_speed_kmh"] for p in self.all_passes])), 1) if self.all_passes else 0.0

        t0_prog = sum(1 for p in t0_ok if p["is_progressive"])
        t1_prog = sum(1 for p in t1_ok if p["is_progressive"])
        t0_rec = sum(1 for r in self.interceptions if r["team_id"] == 0)
        t1_rec = sum(1 for r in self.interceptions if r["team_id"] == 1)

        t0_entries = sum(1 for p in t0_ok if p["is_final_third_entry"] or p["is_box_entry"]) + sum(
            1 for c in self.carries_and_entries if c["team_id"] == 0
        )
        t1_entries = sum(1 for p in t1_ok if p["is_final_third_entry"] or p["is_box_entry"]) + sum(
            1 for c in self.carries_and_entries if c["team_id"] == 1
        )

        return {
            "team0_possession_pct": pct0,
            "team1_possession_pct": pct1,
            "team0_possession_sec": round(c0 / self.fps, 1),
            "team1_possession_sec": round(c1 / self.fps, 1),
            "team0_touches": sum(sp.team_id == 0 for sp in self.touch_spells),
            "team1_touches": sum(sp.team_id == 1 for sp in self.touch_spells),
            "team0_passes": len(t0_ok),
            "team0_passes_attempted": len(t0_all),
            "team0_pass_accuracy": t0_acc,
            "team1_passes": len(t1_ok),
            "team1_passes_attempted": len(t1_all),
            "team1_pass_accuracy": t1_acc,
            "total_passes": len(self.passes),
            "total_passes_attempted": len(self.all_passes),
            "total_incomplete_passes": len(self.incomplete_passes),
            "total_pass_accuracy": tot_acc,
            "avg_pass_length_m": avg_len,
            "avg_pass_speed_kmh": avg_spd,
            "team0_progressive_passes": t0_prog,
            "team1_progressive_passes": t1_prog,
            "team0_recoveries": t0_rec,
            "team1_recoveries": t1_rec,
            "team0_attacking_entries": t0_entries,
            "team1_attacking_entries": t1_entries,
            "total_recoveries": len(self.interceptions),
            "total_shots": len(self.shots),
        }

    def get_passes_table(self) -> List[Dict]:
        """Returns a clean, comprehensive table of ALL passes (completed & incomplete)."""
        rows = []
        for p in self.all_passes:
            tid = p["team_id"]
            t_str = f"Team {tid} ({'Home' if tid == 0 else 'Away'})"
            to_str = (
                f"#{p['target_id']}"
                if p["is_completed"]
                else f"#{p['target_id']} (Opp. Interception)"
            )
            status_str = "✅ Completed" if p["is_completed"] else "❌ Incomplete"
            rows.append({
                "Pass #": p["pass_number"],
                "Time": f"{p['time_sec']:.1f}s → {p['end_time_sec']:.1f}s",
                "Team": t_str,
                "From": f"#{p['player_id']}",
                "To / Interceptor": to_str,
                "Outcome": status_str,
                "Pass Type": p["subtype"],
                "Length (m)": p["length_m"],
                "Avg Speed (km/h)": p["avg_speed_kmh"],
                "Peak Speed (km/h)": p["peak_speed_kmh"],
                "Duration (s)": p["duration_sec"],
                "Target Zone": p["zone"],
            })
        return rows

    def get_passing_network(self, team_id: int) -> Dict:
        """Builds nodes and weighted edges for a team's 2D Passing Network."""
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
        """Backwards-compatible event log table."""
        return self.get_passes_table()
