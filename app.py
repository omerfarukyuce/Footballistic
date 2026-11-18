"""
Footballistic: 2D Tactical Football Match Analytics Platform.
Minimalist, high-quality, explanatory sports intelligence.
No sidebar - clean full-width tactical dashboard with top-right controls.
"""

import os
import tempfile
from typing import Optional, List, Dict
import cv2
import pandas as pd
import streamlit as st
import numpy as np

from main import run_pipeline
from src.analysis.heatmap import SpatialAnalytics
from src.homography.pitch_view import PitchView2D

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Footballistic • Tactical Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- CLEAN MINIMAL REFINED CSS -----------------
st.markdown("""
<style>
    /* Completely hide Streamlit sidebar and toggle button */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Clean layout & typography */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }
    
    /* Header */
    .header-container {
        display: flex;
        align-items: baseline;
        gap: 12px;
        margin-bottom: 4px;
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        color: #F8FAFC;
    }
    .header-sub {
        font-size: 0.90rem;
        color: #94A3B8;
        margin: 0;
    }
    
    /* Subtle Metric Cards */
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 10px 16px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #F3F4F6 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #9CA3AF !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid #1F2937;
        margin-top: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.98rem;
        font-weight: 600;
        padding: 10px 16px;
        color: #9CA3AF;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }

    /* Popover button styling */
    div[data-testid="stPopover"] > button {
        background: #1F2937 !important;
        border: 1px solid #374151 !important;
        color: #F3F4F6 !important;
        font-weight: 600 !important;
        padding: 6px 14px !important;
        border-radius: 6px !important;
    }
    div[data-testid="stPopover"] > button:hover {
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- SESSION STATE & INSTANT PRE-COMPUTED LOAD -----------------
def load_match_data(match_key: str):
    """
    Instantly loads pre-computed match data from disk without any delay.
    """
    if match_key == "sample2":
        spatial_json = "output_sample2/spatial_data.json"
        video_file = "output_sample2/analyzed_match.mp4"
    else:
        spatial_json = "output/spatial_data.json"
        video_file = "output/analyzed_match.mp4"

    if os.path.exists(spatial_json):
        try:
            sa = SpatialAnalytics.load_from_json(spatial_json)
            sa.enforce_team_squad_limits(max_players_per_team=11)
            pv = PitchView2D(pitch_width=1050, pitch_height=680)
            return {
                "spatial_analytics": sa,
                "pitch_view": pv,
                "output_video_path": video_file if os.path.exists(video_file) else None,
                "total_players": len(sa.player_tracks),
                "video_info": {"width": sa.width, "height": sa.height, "total_frames": sa.total_processed_frames}
            }
        except Exception:
            pass
    return None


def init_session():
    if "current_match" not in st.session_state:
        st.session_state["current_match"] = "sample1"
    if "results" not in st.session_state or st.session_state["results"] is None:
        st.session_state["results"] = load_match_data(st.session_state["current_match"])


init_session()

# ----------------- TOP HEADER -----------------
st.markdown("""
<div class="header-container">
    <h1 class="header-title">⚽ Footballistic</h1>
    <span class="header-sub">2D Tactical Formations, Trajectories & Spatial Heatmaps</span>
</div>
""", unsafe_allow_html=True)

# ----------------- CONTROLS ROW (DOWN ABOVE TABS) -----------------
header_space, header_match, header_model, header_upload = st.columns([3.4, 2.6, 2.0, 2.0], vertical_alignment="bottom")

with header_match:
    match_labels = {
        "sample1": "Match 1: Broadcast (sample.mp4)",
        "sample2": "Match 2: Tactical Wide (sample2.mp4)"
    }
    match_keys = list(match_labels.keys())
    cur_match = st.session_state.get("current_match", "sample1")
    selected_match_idx = match_keys.index(cur_match) if cur_match in match_keys else 0

    chosen_match = st.selectbox(
        "Match Footage",
        options=match_keys,
        index=selected_match_idx,
        format_func=lambda k: match_labels[k],
        label_visibility="collapsed"
    )
    if chosen_match != st.session_state.get("current_match"):
        st.session_state["current_match"] = chosen_match
        st.session_state["results"] = load_match_data(chosen_match)
        st.rerun()

with header_model:
    model_choice = st.selectbox(
        "Detection Model",
        ["YOLO11 (yolo11m.pt)", "YOLOv8 (yolov8m.pt)"],
        index=0,
        help="Select detection model architecture",
        label_visibility="collapsed"
    )
    selected_model_path = "weights/yolo11m.pt" if "11" in model_choice else "weights/yolov8m.pt"

with header_upload:
    with st.popover("📁 Upload / Run Video", use_container_width=True):
        st.markdown("##### 📹 Match Video Analysis")
        st.caption("Upload a custom video or re-run analysis across 100% of frames.")
        uploaded_custom = st.file_uploader("Select Match Video (MP4 / AVI)", type=["mp4", "avi", "mov"])
        
        target_video = "data/sample_videos/sample2.mp4" if chosen_match == "sample2" else "data/sample_videos/sample.mp4"
        output_dest = "output_sample2" if chosen_match == "sample2" else "output"
        if uploaded_custom is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_custom.read())
            target_video = tfile.name
            output_dest = "output_custom"

        run_btn = st.button("🚀 Run Full Video Analysis", type="primary", use_container_width=True)

# ----------------- RE-RUN ANALYSIS IF TRIGGERED -----------------
if 'run_btn' in locals() and run_btn:
    prog_bar = st.progress(0)
    status_box = st.empty()

    def progress_callback(c, t, msg):
        pct = min(1.0, max(0.0, c / t))
        prog_bar.progress(pct)
        status_box.caption(f"{msg} ({int(pct * 100)}%)")

    try:
        with st.spinner("Processing every video frame with YOLO and 2D spatial tracking..."):
            pipeline_res = run_pipeline(
                video_path=target_video,
                output_dir=output_dest,
                max_frames=0,  # 0 = ALL FRAMES
                model_name=selected_model_path,
                confidence=0.35,
                progress_callback=progress_callback
            )
            st.session_state["results"] = pipeline_res
            prog_bar.empty()
            status_box.empty()
            st.success("All frames analyzed successfully!")
            st.rerun()
    except Exception as e:
        st.error(f"Error during video processing: {e}")

# ----------------- MAIN VIEW -----------------
results = st.session_state.get("results")

if results is None or results.get("spatial_analytics") is None:
    st.info("No match analysis data found. Please upload a video in the top right to begin.")
    st.stop()

spatial_analytics: SpatialAnalytics = results["spatial_analytics"]
pitch_view: PitchView2D = results.get("pitch_view") or PitchView2D(1050, 680)

# Minimal Tabs - TEAM VIEW FIRST AS DEFAULT
tab_team, tab_player, tab_media = st.tabs([
    "🛡️ Team Analytics",
    "🎯 Player Deep-Dive",
    "🎬 Match Video & Data"
])

# -------------------------------------------------------------
# TAB 1: TEAM TACTICAL ANALYTICS (DEFAULT OPEN FIRST TAB)
# -------------------------------------------------------------
with tab_team:
    team_view_mode = st.radio(
        "Select Team View:",
        ["🔴 Team 0 (Home)", "🔵 Team 1 (Away)", "🆚 Comparative Full Pitch"],
        horizontal=True
    )
    
    # Guaranteed <= 11 players per team with Goalkeepers anchored
    avg_pos_2d = spatial_analytics.get_average_positions_2d(min_frames=1, max_per_team=11)

    if "Comparative" in team_view_mode:
        t0 = spatial_analytics.get_team_metrics(0, min_frames=1)
        t1 = spatial_analytics.get_team_metrics(1, min_frames=1)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Team 0 Squad", f"{t0['player_count']} Players", help="1 Goalkeeper + 10 Outfield players (FIFA Standard)")
        with c2:
            st.metric("Team 0 Compactness", f"{t0['team_depth_m']} m")
        with c3:
            st.metric("Team 1 Squad", f"{t1['player_count']} Players", help="1 Goalkeeper + 10 Outfield players (FIFA Standard)")
        with c4:
            st.metric("Team 1 Compactness", f"{t1['team_depth_m']} m")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        t_left, t_right = st.columns(2)
        with t_left:
            st.markdown("##### 📍 2D Bird's-Eye Formations (Both Teams • 11 vs 11)")
            all_form = pitch_view.draw_2d_formations(avg_pos_2d, spatial_analytics.player_teams)
            st.image(cv2.cvtColor(all_form, cv2.COLOR_BGR2RGB), use_column_width=True)

        with t_right:
            st.markdown("##### 🔥 2D Bird's-Eye Collective Heatmap")
            all_heat = pitch_view.generate_2d_heatmap(
                spatial_analytics.get_2d_positions(min_frames=1),
                sigma=28.0
            )
            st.image(cv2.cvtColor(all_heat, cv2.COLOR_BGR2RGB), use_column_width=True)

    else:
        target_team = 0 if "Team 0" in team_view_mode else 1
        tm = spatial_analytics.get_team_metrics(target_team, min_frames=1)
        t_title = f"Team {target_team}"

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Active Squad", f"{tm['player_count']} Players", help="1 Goalkeeper (GK) + 10 Outfield players (FIFA Standard)")
        with k2:
            st.metric("Team Centroid (X, Y)", f"{tm['centroid_m'][0]}m, {tm['centroid_m'][1]}m")
        with k3:
            st.metric("Compactness (Depth)", f"{tm['team_depth_m']} m")
        with k4:
            st.metric("Team Width", f"{tm['team_width_m']} m")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown(f"##### 📍 {t_title} 2D Tactical Shape & Centroid")
            team_form_img = pitch_view.draw_2d_team_formation(
                target_team,
                avg_pos_2d,
                spatial_analytics.player_teams
            )
            st.image(cv2.cvtColor(team_form_img, cv2.COLOR_BGR2RGB), use_column_width=True)

        with t_col2:
            st.markdown(f"##### 🔥 {t_title} 2D Collective Heatmap")
            team_heat_img = pitch_view.generate_2d_heatmap(
                spatial_analytics.get_2d_positions(team_id=target_team, min_frames=1),
                sigma=28.0
            )
            st.image(cv2.cvtColor(team_heat_img, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.caption(
            f"**Channel Preference:** Left Flank: **{tm['left_flank_pct']}%** | "
            f"Center: **{tm['center_pct']}%** | "
            f"Right Flank: **{tm['right_flank_pct']}%** &nbsp;•&nbsp; "
            f"**Pitch Depth:** Defensive: **{tm['defensive_pct']}%** | "
            f"Middle: **{tm['middle_pct']}%** | "
            f"Attacking: **{tm['attacking_pct']}%**"
        )


# -------------------------------------------------------------
# TAB 2: PLAYER DEEP-DIVE ANALYTICS
# -------------------------------------------------------------
with tab_player:
    # Filter players respecting squad limits and presence
    all_pids = sorted(list(avg_pos_2d.keys()))

    if not all_pids:
        st.warning("No players found in active dataset.")
    else:
        # Filter Bar in 2 Columns
        col_filter, col_select = st.columns([1, 2])
        with col_filter:
            team_filter = st.selectbox(
                "Filter by Team:",
                ["All Teams", "🔴 Team 0", "🔵 Team 1"],
                index=0
            )

        if "Team 0" in team_filter:
            filtered_ids = [p for p in all_pids if spatial_analytics.player_teams.get(p) == 0]
        elif "Team 1" in team_filter:
            filtered_ids = [p for p in all_pids if spatial_analytics.player_teams.get(p) == 1]
        else:
            filtered_ids = all_pids

        with col_select:
            if not filtered_ids:
                st.warning("No players found in this team category.")
            else:
                def format_player_label(pid):
                    t = spatial_analytics.player_teams.get(pid, 0)
                    t_str = "Team 0" if t == 0 else "Team 1"
                    if pid in (1, 13) or str(pid) in ("1", "13"):
                        return f"Player #{pid} (GK)  •  {t_str} [Goalkeeper]"
                    return f"Player #{pid}  •  {t_str}"

                selected_pid = st.selectbox(
                    "Select Player ID:",
                    filtered_ids,
                    format_func=format_player_label
                )

                # Metrics for ONLY this selected player
                pm = spatial_analytics.get_player_metrics(selected_pid)
                team_id = pm["team_id"]
                is_keeper = selected_pid in (1, 13) or str(selected_pid) in ("1", "13")

                # Clean Metrics Strip
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    p_label = f"#{selected_pid} (GK)" if is_keeper else f"#{selected_pid}"
                    st.metric("Player Profile", f"{p_label} • Team {team_id}")
                with m2:
                    st.metric("Est. Distance", f"{pm['distance_m']} m")
                with m3:
                    st.metric("Avg Speed", f"{pm['avg_speed_kmh']} km/h")
                with m4:
                    st.metric("Tactical Role", pm["tactical_zone"])

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

                # 2 Prominent Top-Down Pitches (Side-by-Side)
                p_col1, p_col2 = st.columns(2)

                with p_col1:
                    st.markdown("##### 📍 Bird's-Eye Average Position & Trajectory")
                    focus_pitch = pitch_view.draw_2d_player_focus(
                        tracker_id=selected_pid,
                        avg_pos_2d=pm["avg_pos_2d"],
                        team_id=team_id,
                        trail_2d=pm["positions_2d"],
                        teammate_positions=None
                    )
                    st.image(
                        cv2.cvtColor(focus_pitch, cv2.COLOR_BGR2RGB),
                        use_column_width=True
                    )

                with p_col2:
                    st.markdown("##### 🔥 Bird's-Eye Individual Heatmap")
                    p_heat = pitch_view.generate_2d_heatmap(pm["positions_2d"], sigma=22.0)
                    st.image(
                        cv2.cvtColor(p_heat, cv2.COLOR_BGR2RGB),
                        use_column_width=True
                    )

                # Field Thirds Breakdown
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                st.caption(
                    f"**Field Distribution:** Defensive: **{pm['defensive_pct']}%** | "
                    f"Middle Third: **{pm['middle_pct']}%** | "
                    f"Attacking: **{pm['attacking_pct']}%** &nbsp;•&nbsp; "
                    f"Pitch Presence: **{pm['presence_pct']}%** ({pm['frames_active']} frames)"
                )


# -------------------------------------------------------------
# TAB 3: VIDEO REPLAY & DATASET
# -------------------------------------------------------------
with tab_media:
    col_v, col_d = st.columns([1, 1])

    with col_v:
        st.markdown("##### 🎥 Annotated Match Video Replay (Full 30s)")
        v_path = results.get("output_video_path")
        if v_path and os.path.exists(v_path):
            with open(v_path, "rb") as f:
                st.video(f.read())
        else:
            st.info("Annotated video has not been rendered yet.")

    with col_d:
        st.markdown("##### 📋 Spatial Player Analytics Dataset (11 vs 11)")
        summary_rows = spatial_analytics.get_player_summary(min_frames=1)
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            st.dataframe(df, use_container_width=True, height=280)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV Dataset",
                data=csv,
                file_name="footballistic_match_analytics.csv",
                mime="text/csv",
                use_container_width=True
            )
