"""
Phase 3: Advanced Physical & Tactical Metrics Engine.

Provides two broadcast-grade analytics engines operating directly on pan-compensated
2D pitch trajectories (`SpatialAnalytics`) and possession states (`MatchEventEngine`):

1. PhysicalPerformanceEngine:
   - 5 FIFA / Catapult standard speed zones:
       Z1 Standing/Walking (< 7 km/h)
       Z2 Jogging (7–14 km/h)
       Z3 Running (14–18 km/h)
       Z4 High-Speed Running / HSR (18–21 km/h)
       Z5 Sprinting (>= 21 km/h)
   - Distinct sprint effort counting (>= 21 km/h sustained >= 0.40s with 0.8s cooldown)
   - High-Speed Running distance (> 18 km/h), Sprint distance (> 21 km/h),
     Peak Speed (km/h), Peak Acceleration (m/s^2), and full velocity timeline.

2. TacticalDynamicsEngine:
   - Time-series Defensive Line Height, Midfield Line, Attacking Line,
     Outfield Block Compactness (Depth & Width), and Defense-to-Midfield Gap (m).
   - Ball-Carrier Pressing Intensity: nearest-defender pressure distance (m),
     pressers within 5m/10m, and High-Press Share (% of opponent possession within 4.5m).

3. Obsidian-Native Retina Chart Renderers (Matplotlib Agg -> RGB numpy arrays):
   - Harmonized to Footballistic's #0F1218 matte obsidian studio palette.
"""

from collections import defaultdict
import io
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.analysis.heatmap import SpatialAnalytics
from src.analysis.events import MatchEventEngine


# Obsidian Pitch Studio Chart Palette
BG_FIG = "#0F1218"
BG_AX = "#0D1016"
GRID_COL = "#1E2430"
SPINE_COL = "#222938"
TXT_PRI = "#F3F5F8"
TXT_SEC = "#8A93A5"
TXT_MUTED = "#5A6273"
COL_HOME = "#E5484D"
COL_AWAY = "#3B82F6"
COL_EMERALD = "#10B981"
COL_AMBER = "#F59E0B"
COL_PURPLE = "#8B5CF6"


# FIFA / Catapult Speed Zones (km/h)
SPEED_ZONES = [
    ("Z1 Walking", 0.0, 7.0, "#475569"),
    ("Z2 Jogging", 7.0, 14.0, "#38BDF8"),
    ("Z3 Running", 14.0, 18.0, "#10B981"),
    ("Z4 High-Speed", 18.0, 21.0, "#F59E0B"),
    ("Z5 Sprinting", 21.0, 99.0, "#E5484D"),
]


class PhysicalPerformanceEngine:
    """Computes FIFA speed zones, HSR, sprint efforts, peak speed & physical leaderboards."""

    HSR_THRESH_KMH = 18.0
    SPRINT_THRESH_KMH = 21.0
    MIN_SPRINT_DUR_SEC = 0.40
    SPRINT_COOLDOWN_SEC = 0.80

    def __init__(self, spatial_analytics: SpatialAnalytics, fps: float = 25.0):
        self.sa = spatial_analytics
        self.fps = float(fps) if fps > 0 else 25.0
        self.player_profiles: Dict[int, Dict] = {}
        self._compute_all()

    def _compute_all(self):
        for pid in sorted(self.sa.player_tracks_2d.keys()):
            tid = self.sa.player_teams.get(pid, -1)
            if tid not in (0, 1):
                continue
            self.player_profiles[pid] = self._compute_player(pid, tid)

    def _compute_player(self, pid: int, tid: int) -> Dict:
        kin = self.sa.get_player_kinematics(pid, fps=self.fps)
        spds = np.asarray(kin["speeds_kmh"], dtype=np.float64)
        dts = np.asarray(kin["dt_sec"], dtype=np.float64)
        frs = np.asarray(kin["valid_frames"], dtype=np.int32)

        zone_dist = {name: 0.0 for name, _, _, _ in SPEED_ZONES}
        zone_time = {name: 0.0 for name, _, _, _ in SPEED_ZONES}
        hsr_dist = 0.0
        sprint_dist = 0.0
        sprint_count = 0
        sprint_events: List[Tuple[float, float, float]] = []  # (start_s, end_s, peak_kmh)
        peak_accel = 0.0

        if len(spds) >= 2:
            step_dists = (spds / 3.6) * dts * (spds >= 1.2)
            for name, lo, hi, _ in SPEED_ZONES:
                mask = (spds >= lo) & (spds < hi)
                zone_dist[name] = float(np.sum(step_dists[mask]))
                zone_time[name] = float(np.sum(dts[mask]))

            hsr_dist = float(np.sum(step_dists[spds >= self.HSR_THRESH_KMH]))
            sprint_dist = float(np.sum(step_dists[spds >= self.SPRINT_THRESH_KMH]))

            # Distinct sprint effort detection (>= 21 km/h sustained >= 0.40s)
            in_sprint = False
            s_start_t = 0.0
            s_peak = 0.0
            s_dur = 0.0
            last_sprint_end = -99.0

            for k in range(len(spds)):
                t_cur = float(frs[k]) / self.fps
                v = float(spds[k])
                dt = float(dts[k])
                if v >= self.SPRINT_THRESH_KMH:
                    if not in_sprint:
                        in_sprint = True
                        s_start_t = t_cur
                        s_peak = v
                        s_dur = dt
                    else:
                        s_dur += dt
                        if v > s_peak:
                            s_peak = v
                else:
                    if in_sprint:
                        if s_dur >= self.MIN_SPRINT_DUR_SEC and (s_start_t - last_sprint_end) >= self.SPRINT_COOLDOWN_SEC:
                            sprint_count += 1
                            sprint_events.append((round(s_start_t, 1), round(t_cur, 1), round(s_peak, 1)))
                            last_sprint_end = t_cur
                        in_sprint = False
            if in_sprint and s_dur >= self.MIN_SPRINT_DUR_SEC:
                t_end = float(frs[-1]) / self.fps
                if (s_start_t - last_sprint_end) >= self.SPRINT_COOLDOWN_SEC:
                    sprint_count += 1
                    sprint_events.append((round(s_start_t, 1), round(t_end, 1), round(s_peak, 1)))

            # Smoothed acceleration (m/s^2) over ~0.4s window
            v_ms = spds / 3.6
            lag = min(10, max(1, len(v_ms) // 6))
            if len(v_ms) > lag:
                acc = (v_ms[lag:] - v_ms[:-lag]) / np.maximum(0.15, (frs[lag:] - frs[:-lag]) / self.fps)
                # Clip to physically valid human football acceleration (<= 6.5 m/s^2)
                acc_valid = acc[np.abs(acc) <= 6.5]
                if len(acc_valid) > 0:
                    peak_accel = float(np.percentile(acc_valid, 98))

        tot_dist = max(0.0, kin["distance_m"])
        zone_pct = {
            name: round((zone_dist[name] / tot_dist) * 100.0, 1) if tot_dist > 0.5 else (100.0 if idx == 0 else 0.0)
            for idx, (name, _, _, _) in enumerate(SPEED_ZONES)
        }

        # Downsample timeline cleanly to ~5 Hz (every 5 frames) for silky chart rendering
        t_axis = (frs / self.fps).tolist()
        v_axis = spds.tolist()
        if len(t_axis) > 150:
            step = max(1, len(t_axis) // 150)
            t_axis = t_axis[::step]
            v_axis = v_axis[::step]

        return {
            "player_id": pid,
            "team_id": tid,
            "is_gk": (pid in self.sa.goalkeeper_ids) or (pid in (1, 13)),
            "distance_m": kin["distance_m"],
            "avg_speed_kmh": kin["avg_speed_kmh"],
            "top_speed_kmh": kin["top_speed_kmh"],
            "peak_accel_ms2": round(max(0.0, peak_accel), 2),
            "hsr_distance_m": round(hsr_dist, 1),
            "sprint_distance_m": round(sprint_dist, 1),
            "sprint_count": sprint_count,
            "sprint_events": sprint_events,
            "zone_distance_m": {k: round(v, 1) for k, v in zone_dist.items()},
            "zone_time_sec": {k: round(v, 1) for k, v in zone_time.items()},
            "zone_pct": zone_pct,
            "timeline_t": t_axis,
            "timeline_kmh": v_axis,
        }

    def get_player_profile(self, pid: int) -> Dict:
        return self.player_profiles.get(pid, self._compute_player(pid, self.sa.player_teams.get(pid, 0)))

    def get_team_physical_summary(self, team_id: int) -> Dict:
        ps = [p for p in self.player_profiles.values() if p["team_id"] == team_id]
        if not ps:
            return {
                "total_distance_m": 0.0,
                "hsr_distance_m": 0.0,
                "sprint_distance_m": 0.0,
                "total_sprints": 0,
                "fastest_player_id": None,
                "fastest_speed_kmh": 0.0,
            }
        fastest = max(ps, key=lambda x: x["top_speed_kmh"])
        return {
            "total_distance_m": round(sum(p["distance_m"] for p in ps), 1),
            "hsr_distance_m": round(sum(p["hsr_distance_m"] for p in ps), 1),
            "sprint_distance_m": round(sum(p["sprint_distance_m"] for p in ps), 1),
            "total_sprints": sum(p["sprint_count"] for p in ps),
            "fastest_player_id": fastest["player_id"],
            "fastest_speed_kmh": fastest["top_speed_kmh"],
        }

    def get_physical_leaderboard(self, team_filter: Optional[int] = None) -> List[Dict]:
        rows = []
        for p in self.player_profiles.values():
            if team_filter is not None and p["team_id"] != team_filter:
                continue
            rows.append({
                "Player": f"#{p['player_id']} (GK)" if p["is_gk"] else f"#{p['player_id']}",
                "Team": f"Team {p['team_id']} ({'Home' if p['team_id'] == 0 else 'Away'})",
                "Distance (m)": p["distance_m"],
                "Avg Speed (km/h)": p["avg_speed_kmh"],
                "Top Speed (km/h)": p["top_speed_kmh"],
                "HSR >18 (m)": p["hsr_distance_m"],
                "Sprint >21 (m)": p["sprint_distance_m"],
                "Sprints": p["sprint_count"],
                "Peak Accel (m/s²)": p["peak_accel_ms2"],
            })
        rows.sort(key=lambda r: (-r["Top Speed (km/h)"], -r["Distance (m)"]))
        return rows


class TacticalDynamicsEngine:
    """
    Tracks frame-by-frame team structural shape & pressing intensity:
      - Defensive Line Height (m from own goal)
      - Midfield Line & Defense-to-Midfield Gap (m)
      - Outfield Block Compactness (Length & Width in m)
      - Pressing Intensity (nearest defender distance to ball & High-Press %)
    """

    HIGH_PRESS_DIST_M = 4.5

    def __init__(
        self,
        spatial_analytics: SpatialAnalytics,
        event_engine: Optional[MatchEventEngine] = None,
        fps: float = 25.0,
        sample_step: int = 8,
    ):
        self.sa = spatial_analytics
        self.ev = event_engine
        self.fps = float(fps) if fps > 0 else 25.0
        self.sample_step = max(2, int(sample_step))

        self.scale_x = self.sa.PITCH_LENGTH_M / max(1.0, self.sa.FIELD_W)
        self.scale_y = self.sa.PITCH_WIDTH_M / max(1.0, self.sa.FIELD_H)

        self.defends_left: Dict[int, bool] = {0: True, 1: False}
        self.team_series: Dict[int, Dict[str, List[float]]] = {}
        self.pressing_series: Dict[str, List] = {"t": [], "dist_m": [], "pressing_team": [], "within_5m": []}
        self.summary: Dict[int, Dict] = {}

        self._compute()

    def _px_to_m(self, px: float, py: float) -> Tuple[float, float]:
        xm = float(np.clip((px - self.sa.FIELD_X1) * self.scale_x, 0.0, self.sa.PITCH_LENGTH_M))
        ym = float(np.clip((py - self.sa.FIELD_Y1) * self.scale_y, 0.0, self.sa.PITCH_WIDTH_M))
        return xm, ym

    def _detect_defending_sides(self):
        avg_x = {0: [], 1: []}
        for pid, pts in self.sa.player_tracks_2d.items():
            tid = self.sa.player_teams.get(pid, -1)
            if tid in (0, 1) and pts:
                xm, _ = self._px_to_m(float(np.mean([p[0] for p in pts])), 340.0)
                avg_x[tid].append(xm)
        m0 = float(np.mean(avg_x[0])) if avg_x[0] else 35.0
        m1 = float(np.mean(avg_x[1])) if avg_x[1] else 70.0
        if m0 <= m1:
            self.defends_left = {0: True, 1: False}
        else:
            self.defends_left = {0: False, 1: True}

    def _from_own_goal_x(self, xm: float, team_id: int) -> float:
        return xm if self.defends_left.get(team_id, True) else (self.sa.PITCH_LENGTH_M - xm)

    @staticmethod
    def _smooth(vals: List[float], win: int = 5) -> List[float]:
        if len(vals) < 3:
            return vals
        w = min(win, len(vals) if len(vals) % 2 == 1 else len(vals) - 1)
        if w < 3:
            return vals
        pad = w // 2
        arr = np.asarray(vals, dtype=np.float64)
        return np.convolve(np.pad(arr, pad, mode="edge"), np.ones(w) / w, mode="valid").tolist()

    def _compute(self):
        self._detect_defending_sides()

        # Build per-frame lookup of REAL (non-synthesized) outfield players in meters
        frame_players: Dict[int, Dict[int, List[Tuple[int, float, float]]]] = defaultdict(lambda: {0: [], 1: []})
        gk_set = set(self.sa.goalkeeper_ids) | {1, 13}

        for pid, frs in self.sa.player_frames.items():
            tid = self.sa.player_teams.get(pid, -1)
            if tid not in (0, 1) or pid in gk_set:
                continue
            pts = self.sa.player_tracks_2d.get(pid, [])
            cams = self.sa.player_tracks.get(pid, [])
            for idx, (f, pt) in enumerate(zip(frs, pts)):
                if idx < len(cams) and abs(cams[idx][0] - 100.0) < 1e-3 and abs(cams[idx][1] - 100.0) < 1e-3:
                    continue
                xm, ym = self._px_to_m(pt[0], pt[1])
                frame_players[int(f)][tid].append((pid, xm, ym))

        all_frames = sorted(frame_players.keys())
        if not all_frames:
            for tid in (0, 1):
                self.team_series[tid] = {k: [] for k in ("t", "def_line", "mid_line", "att_line", "block_len", "block_w", "def_mid_gap")}
                self.summary[tid] = {
                    "avg_def_line_m": 0.0, "avg_att_line_m": 0.0,
                    "avg_block_length_m": 0.0, "avg_block_width_m": 0.0,
                    "avg_def_mid_gap_m": 0.0, "avg_press_dist_m": 0.0, "high_press_pct": 0.0,
                }
            return

        sampled = all_frames[:: self.sample_step]
        for tid in (0, 1):
            ts, defs, mids, atts, blens, bws, gaps = [], [], [], [], [], [], []
            for f in sampled:
                pls = frame_players[f][tid]
                if len(pls) < 4:
                    continue
                xs = sorted([self._from_own_goal_x(p[1], tid) for p in pls])
                ys = [p[2] for p in pls]

                # 10th percentile / 2nd deepest outfield defender = true defensive line
                def_l = float(np.percentile(xs, 12))
                att_l = float(np.percentile(xs, 90))
                # Partition sorted outfield players into Defense / Midfield / Attack tertiles
                n = len(xs)
                d_cut = max(2, n // 3)
                m_cut = max(d_cut + 1, (2 * n) // 3)
                def_grp = xs[:d_cut]
                mid_grp = xs[d_cut:m_cut] if m_cut > d_cut else xs[d_cut:]
                mid_l = float(np.mean(mid_grp)) if mid_grp else float(np.median(xs))
                gap = max(0.0, mid_l - float(np.mean(def_grp)))

                ts.append(round(f / self.fps, 2))
                defs.append(def_l)
                mids.append(mid_l)
                atts.append(att_l)
                blens.append(max(0.0, att_l - def_l))
                bws.append(max(0.0, max(ys) - min(ys)))
                gaps.append(gap)

            self.team_series[tid] = {
                "t": ts,
                "def_line": self._smooth(defs, 5),
                "mid_line": self._smooth(mids, 5),
                "att_line": self._smooth(atts, 5),
                "block_len": self._smooth(blens, 5),
                "block_w": self._smooth(bws, 5),
                "def_mid_gap": self._smooth(gaps, 5),
            }

        # Pressing Intensity (nearest opposing defender distance to tracked ball)
        ball_map = {int(f): self._px_to_m(bx, by) for f, bx, by in self.sa.ball_track_2d}
        frame_owner = self.ev.frame_team_owner if self.ev is not None else {}

        press_by_team: Dict[int, List[float]] = {0: [], 1: []}
        pt_t, pt_d, pt_team, pt_w5 = [], [], [], []

        for f in all_frames[:: max(2, self.sample_step // 2)]:
            if f not in ball_map:
                continue
            bx_m, by_m = ball_map[f]
            # Identify possessing team from event engine or nearest player to ball
            owner = frame_owner.get(f)
            if owner not in (0, 1):
                best_d, best_t = 999.0, None
                for tid in (0, 1):
                    for _, px_m, py_m in frame_players[f][tid]:
                        d = math.hypot(px_m - bx_m, py_m - by_m)
                        if d < best_d:
                            best_d, best_t = d, tid
                owner = best_t if best_d <= 8.0 else None

            if owner not in (0, 1):
                continue

            pressing_team = 1 - owner
            opps = frame_players[f][pressing_team]
            if not opps:
                continue
            dists = [math.hypot(px_m - bx_m, py_m - by_m) for _, px_m, py_m in opps]
            d_min = min(dists)
            w5 = sum(1 for d in dists if d <= 5.0)

            press_by_team[pressing_team].append(d_min)
            pt_t.append(round(f / self.fps, 2))
            pt_d.append(d_min)
            pt_team.append(pressing_team)
            pt_w5.append(w5)

        self.pressing_series = {
            "t": pt_t,
            "dist_m": self._smooth(pt_d, 5),
            "pressing_team": pt_team,
            "within_5m": pt_w5,
        }

        for tid in (0, 1):
            s = self.team_series[tid]
            p_list = press_by_team[tid]
            avg_pr = float(np.mean(p_list)) if p_list else 0.0
            hp_pct = (sum(1 for d in p_list if d <= self.HIGH_PRESS_DIST_M) / len(p_list) * 100.0) if p_list else 0.0
            self.summary[tid] = {
                "avg_def_line_m": round(float(np.mean(s["def_line"])), 1) if s["def_line"] else 0.0,
                "avg_att_line_m": round(float(np.mean(s["att_line"])), 1) if s["att_line"] else 0.0,
                "avg_block_length_m": round(float(np.mean(s["block_len"])), 1) if s["block_len"] else 0.0,
                "avg_block_width_m": round(float(np.mean(s["block_w"])), 1) if s["block_w"] else 0.0,
                "avg_def_mid_gap_m": round(float(np.mean(s["def_mid_gap"])), 1) if s["def_mid_gap"] else 0.0,
                "avg_press_dist_m": round(avg_pr, 1),
                "high_press_pct": round(hp_pct, 1),
                "press_samples": len(p_list),
            }


# =============================================================================
# OBSIDIAN-NATIVE DARK CHART RENDERERS (Matplotlib Agg -> RGB numpy array)
# =============================================================================
def _fig_to_rgb(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, facecolor=BG_FIG, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    buf.seek(0)
    import cv2
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _style_ax(ax, xlabel: str, ylabel: str):
    ax.set_facecolor(BG_AX)
    ax.grid(True, color=GRID_COL, linewidth=0.7, linestyle="-", alpha=0.85)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(SPINE_COL)
        ax.spines[sp].set_linewidth(0.9)
    ax.tick_params(colors=TXT_SEC, labelsize=8.5, length=0, pad=6)
    ax.set_xlabel(xlabel, color=TXT_MUTED, fontsize=8.5, fontweight="semibold", labelpad=6)
    ax.set_ylabel(ylabel, color=TXT_MUTED, fontsize=8.5, fontweight="semibold", labelpad=6)


def render_player_speed_chart(profile: Dict, team_id: int) -> np.ndarray:
    """Renders a compact Obsidian-themed instantaneous speed timeline with HSR & Sprint zones."""
    fig, ax = plt.subplots(figsize=(6.8, 3.05), facecolor=BG_FIG)
    _style_ax(ax, "Match Time (s)", "Speed (km/h)")

    t = np.asarray(profile["timeline_t"], dtype=np.float64)
    v = np.asarray(profile["timeline_kmh"], dtype=np.float64)
    line_col = COL_HOME if team_id == 0 else COL_AWAY

    ymax = max(26.0, (float(np.max(v)) + 4.5) if len(v) else 26.0)
    ax.set_ylim(0, ymax)

    # Shaded HSR (18-21) & Sprint (>21) bands
    ax.axhspan(18.0, 21.0, color=COL_AMBER, alpha=0.07, zorder=1)
    ax.axhspan(21.0, ymax, color=COL_HOME, alpha=0.07, zorder=1)
    ax.axhline(18.0, color=COL_AMBER, linewidth=0.9, linestyle="--", alpha=0.65, zorder=2)
    ax.axhline(21.0, color=COL_HOME, linewidth=0.9, linestyle="--", alpha=0.75, zorder=2)

    if len(t) >= 2:
        ax.set_xlim(float(t[0]), max(float(t[-1]), float(t[0]) + 1.0))
        ax.fill_between(t, 0, v, color=line_col, alpha=0.14, zorder=3)
        ax.plot(t, v, color=line_col, linewidth=2.0, zorder=4)

        # Highlight peak speed point
        idx_pk = int(np.argmax(v))
        pk_t, pk_v = float(t[idx_pk]), float(v[idx_pk])
        if pk_v >= 5.0:
            ax.scatter([pk_t], [pk_v], color=TXT_PRI, edgecolor=line_col, s=36, linewidth=1.8, zorder=5)
            x_off = -34 if pk_t > (float(t[-1]) * 0.78) else 8
            ax.annotate(
                f"Peak {pk_v:.1f} km/h",
                xy=(pk_t, pk_v),
                xytext=(x_off, 7),
                textcoords="offset points",
                color=TXT_PRI,
                fontsize=8.0,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#151922", edgecolor=line_col, lw=0.9),
                zorder=6,
            )

        # Subtle right-edge threshold tags
        x_r = float(t[-1])
        ax.text(x_r, 18.4, " HSR 18", color=COL_AMBER, fontsize=7.5, ha="right", va="bottom", fontweight="semibold", alpha=0.85)
        ax.text(x_r, 21.4, " SPRINT 21", color=COL_HOME, fontsize=7.5, ha="right", va="bottom", fontweight="semibold", alpha=0.9)

    ax.set_title(
        f"Player #{profile['player_id']} — Instantaneous Speed Profile  (Avg {profile['avg_speed_kmh']} km/h · Sprints: {profile['sprint_count']})",
        color=TXT_PRI, fontsize=9.5, fontweight="bold", loc="left", pad=10,
    )
    return _fig_to_rgb(fig)


def render_tactical_shape_chart(dynamics: TacticalDynamicsEngine, target_team: Optional[int] = None) -> np.ndarray:
    """
    Renders team structural shape over time:
      - Single team: Defensive Line, Midfield Line, Attacking Line & shaded outfield block envelope.
      - Comparative (None): Team 0 vs Team 1 Defensive Line Height & Outfield Block Compactness.
    """
    fig, ax = plt.subplots(figsize=(6.8, 3.2), facecolor=BG_FIG)

    if target_team in (0, 1):
        s = dynamics.team_series[target_team]
        col = COL_HOME if target_team == 0 else COL_AWAY
        _style_ax(ax, "Match Time (s)", "Distance from Own Goal (m)")
        ax.set_ylim(0, 105)
        t = s["t"]
        if len(t) >= 2:
            ax.set_xlim(t[0], t[-1])
            ax.fill_between(t, s["def_line"], s["att_line"], color=col, alpha=0.14, label="Outfield Block")
            ax.plot(t, s["def_line"], color=col, linewidth=2.1, label="Defensive Line")
            ax.plot(t, s["mid_line"], color=COL_EMERALD, linewidth=1.6, linestyle="--", label="Midfield Line")
            ax.plot(t, s["att_line"], color=TXT_PRI, linewidth=1.4, alpha=0.75, label="Attacking Line")
            ax.axhline(52.5, color=TXT_MUTED, linewidth=0.8, linestyle=":", alpha=0.6)
            ax.text(t[-1], 53.6, "Halfway (52.5m)", color=TXT_MUTED, fontsize=7.2, ha="right")
        t_lbl = "Team 0 (Home)" if target_team == 0 else "Team 1 (Away)"
        ax.set_title(f"{t_lbl} — Line Heights & Outfield Block Envelope", color=TXT_PRI, fontsize=9.3, fontweight="bold", loc="left", pad=10)
    else:
        _style_ax(ax, "Match Time (s)", "Outfield Compactness / Def. Line (m)")
        s0, s1 = dynamics.team_series[0], dynamics.team_series[1]
        if len(s0["t"]) >= 2:
            ax.plot(s0["t"], s0["block_len"], color=COL_HOME, linewidth=2.1, label="Team 0 Block Depth")
            ax.plot(s0["t"], s0["def_line"], color=COL_HOME, linewidth=1.4, linestyle="--", alpha=0.72, label="Team 0 Def. Line")
        if len(s1["t"]) >= 2:
            ax.plot(s1["t"], s1["block_len"], color=COL_AWAY, linewidth=2.1, label="Team 1 Block Depth")
            ax.plot(s1["t"], s1["def_line"], color=COL_AWAY, linewidth=1.4, linestyle="--", alpha=0.72, label="Team 1 Def. Line")
        all_vals = s0["block_len"] + s0["def_line"] + s1["block_len"] + s1["def_line"]
        if all_vals:
            ax.set_ylim(max(0.0, min(all_vals) - 6.0), max(all_vals) + 18.0)
        ax.set_title("Comparative Structural Compactness & Defensive Line", color=TXT_PRI, fontsize=9.3, fontweight="bold", loc="left", pad=10)

    leg = ax.legend(loc="upper right", frameon=True, facecolor="#131720", edgecolor=SPINE_COL, fontsize=7.5, labelcolor=TXT_SEC, ncol=2)
    leg.get_frame().set_linewidth(0.8)
    return _fig_to_rgb(fig)


def render_pressing_chart(dynamics: TacticalDynamicsEngine, target_team: Optional[int] = None) -> np.ndarray:
    """Renders nearest-defender pressure distance to the ball over time."""
    fig, ax = plt.subplots(figsize=(6.8, 3.2), facecolor=BG_FIG)
    _style_ax(ax, "Match Time (s)", "Nearest Presser to Ball (m)")

    ps = dynamics.pressing_series
    t_all = np.asarray(ps["t"], dtype=np.float64)
    d_all = np.asarray(ps["dist_m"], dtype=np.float64)
    tm_all = np.asarray(ps["pressing_team"], dtype=np.int32)

    ymax = max(14.0, (float(np.max(d_all)) + 4.8) if len(d_all) else 14.0)
    ax.set_ylim(0, ymax)
    ax.axhspan(0, dynamics.HIGH_PRESS_DIST_M, color=COL_EMERALD, alpha=0.09, zorder=1)
    ax.axhline(dynamics.HIGH_PRESS_DIST_M, color=COL_EMERALD, linewidth=1.0, linestyle="--", alpha=0.8, zorder=2)

    if len(t_all) >= 2:
        ax.set_xlim(float(t_all[0]), float(t_all[-1]))
        ax.plot(t_all, d_all, color="#475569", linewidth=1.1, alpha=0.45, zorder=2)

        for tid, col, lbl in [(0, COL_HOME, "Team 0 Pressing"), (1, COL_AWAY, "Team 1 Pressing")]:
            if target_team in (0, 1) and tid != target_team:
                continue
            m = tm_all == tid
            if np.any(m):
                ax.scatter(t_all[m], d_all[m], color=col, s=16, alpha=0.92, label=lbl, zorder=4)

        ax.text(
            float(t_all[0]) + 0.5, 0.85,
            f"High Press Zone  <= {dynamics.HIGH_PRESS_DIST_M}m",
            color=COL_EMERALD, fontsize=7.5, ha="left", va="bottom", fontweight="semibold", alpha=0.92,
        )

    title_sfx = (
        f"Team {target_team} ({'Home' if target_team == 0 else 'Away'})"
        if target_team in (0, 1)
        else "Both Teams"
    )
    ax.set_title(f"Pressing Intensity — Nearest Opponent to Ball ({title_sfx})", color=TXT_PRI, fontsize=9.3, fontweight="bold", loc="left", pad=10)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(loc="upper right", frameon=True, facecolor="#131720", edgecolor=SPINE_COL, fontsize=7.6, labelcolor=TXT_SEC)
        leg.get_frame().set_linewidth(0.8)

    return _fig_to_rgb(fig)
