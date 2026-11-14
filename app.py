"""
Footballistic: Streamlit Interactive Football Match Analytics Dashboard.
Features 2D Top-Down Rectangular Pitch Analytics, Live Motion Trails,
and Individual Player Deep-Dives.
"""

import os
import tempfile
from typing import Optional
import cv2
import pandas as pd
import streamlit as st
import numpy as np

from main import run_pipeline
from src.analysis.heatmap import SpatialAnalytics
from src.homography.pitch_view import PitchView2D

# Configure Streamlit Page
st.set_page_config(
    page_title="Footballistic - AI Match Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern tactical sports UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9E9E9E;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    if "results" not in st.session_state:
        st.session_state["results"] = None


init_session()

# Header
st.markdown('<div class="main-header">⚽ Footballistic Match Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Computer Vision & Deep Learning Engine for Automated Football Tracking & Spatial Intelligence</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Match Configuration")
    
    video_source = st.radio(
        "Select Video Input:",
        ["Sample Match Clip (DFL Bundesliga)", "Upload Custom Video"],
        index=0
    )
    
    input_video_path = None
    sample_clip_path = os.path.join("data", "sample_videos", "sample.mp4")

    if video_source == "Sample Match Clip (DFL Bundesliga)":
        if os.path.exists(sample_clip_path):
            input_video_path = sample_clip_path
            st.success("Loaded demo clip: `sample.mp4` (1080p, 25 FPS)")
        else:
            st.warning("Sample clip not found locally. Please upload a video.")
    else:
        uploaded_file = st.file_uploader(
            "Upload Match Video (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"]
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            input_video_path = tfile.name
            st.success("Custom video uploaded successfully!")

    st.markdown("---")
    st.subheader("Analysis Parameters")

    model_choice = st.selectbox(
        "Detection Model:",
        ["YOLO11 (Latest Flagship - Recommended)", "YOLOv8 (Classic)"],
        index=0,
        help="YOLO11 offers higher accuracy and faster inference with fewer parameters."
    )
    selected_model_path = "weights/yolo11m.pt" if "YOLO11" in model_choice else "weights/yolov8m.pt"

    analyze_full = st.checkbox("Analyze Full Video (All Frames / 30 seconds)", value=True)
    
    if not analyze_full:
        max_frames = st.slider(
            "Frames to Analyze (1 sec ≈ 25 frames):",
            min_value=25,
            max_value=750,
            value=150,
            step=25,
            help="Higher frame counts yield longer tracking but take more processing time."
        )
    else:
        max_frames = 0  # 0 indicates full video in main.py

    conf_thresh = st.slider(
        "Detection Confidence:",
        min_value=0.20,
        max_value=0.70,
        value=0.35,
        step=0.05
    )

    analyze_btn = st.button("🚀 Run Match Analysis", type="primary", use_container_width=True)


# ----------------- RUN ANALYSIS -----------------
if analyze_btn:
    if input_video_path is None or not os.path.exists(input_video_path):
        st.error("Please provide a valid video input before starting analysis.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(curr, total, msg):
            pct = min(1.0, max(0.0, curr / total))
            progress_bar.progress(pct)
            status_text.info(f"⏳ **{msg}** ({int(pct * 100)}%)")

        try:
            with st.spinner("Processing match video with YOLO11, pitch segmentation & motion tracking..."):
                results = run_pipeline(
                    video_path=input_video_path,
                    output_dir="output",
                    max_frames=max_frames,
                    model_name=selected_model_path,
                    confidence=conf_thresh,
                    progress_callback=update_progress
                )
                st.session_state["results"] = results
                status_text.success("🎉 Analysis complete! View the interactive reports below.")
                progress_bar.empty()
        except Exception as e:
            st.error(f"Error during video processing: {e}")


# ----------------- MAIN DISPLAY -----------------
results = st.session_state["results"]

if results is None:
    # If existing output files exist on disk from previous runs, load them
    existing_video = "output/analyzed_match.mp4"
    if os.path.exists(existing_video):
        st.info("💡 Previous analysis found on disk! Run a new analysis anytime from the sidebar or explore the session below.")
        results = {
            "output_video_path": existing_video,
            "team0_heatmap_path": "output/heatmap_team_0.png",
            "team1_heatmap_path": "output/heatmap_team_1.png",
            "overlay_path": "output/heatmap_pitch_overlay.png",
            "formation_path": "output/average_positions.png",
            "pitch_2d_formation_path": "output/pitch_2d_formation.png",
            "pitch_2d_team0_path": "output/pitch_2d_heatmap_team0.png",
            "pitch_2d_team1_path": "output/pitch_2d_heatmap_team1.png",
            "pitch_2d_all_path": "output/pitch_2d_heatmap_all.png",
            "first_frame_path": "output/first_frame.png",
            "total_players": 46,
            "video_info": {"width": 1920, "height": 1080, "fps": 25.0, "total_frames": 750}
        }
        st.session_state["results"] = results

if results is not None:
    # KPI Metrics
    v_info = results.get("video_info", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players Tracked", f"{results.get('total_players', 0)} players")
    with col2:
        st.metric("Resolution", f"{v_info.get('width', 1920)} x {v_info.get('height', 1080)}")
    with col3:
        st.metric("Frame Rate", f"{v_info.get('fps', 25.0)} FPS")
    with col4:
        st.metric("Pitch Tracking", "Active (Green Masked)")

    # Tabs
    tab_video, tab_heatmaps, tab_player, tab_data = st.tabs([
        "🎥 Video Replay & Tracking",
        "🗺️ Team Heatmaps & Formations",
        "👤 Player Deep-Dive",
        "📋 Roster & Spatial Data"
    ])

    # TAB 1: VIDEO REPLAY
    with tab_video:
        st.subheader("Annotated Match Video with Dynamic Trails")
        st.markdown(
            "🔴 **Team 0:** Red Trails | 🔵 **Team 1:** Royal Blue Trails | ⚪ **Ball:** White Flight Trajectory | 🟨 **Referee:** Yellow"
        )
        
        video_path = results.get("output_video_path")
        if video_path and os.path.exists(video_path):
            with open(video_path, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes)
        else:
            st.warning("Analyzed video not found.")

    # TAB 2: TEAM HEATMAPS & FORMATIONS
    with tab_heatmaps:
        st.subheader("Tactical Pitch Spatial Density & Average Formations")

        view_type = st.radio(
            "Choose Display Perspective:",
            ["📐 2D Top-Down Rectangular Pitch (Kuşbakışı - Önerilen)", "🏟️ Camera Perspective Overlay"],
            horizontal=True
        )

        if "2D Top-Down" in view_type:
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("#### 📍 2D Top-Down Average Formations")
                if os.path.exists(results.get("pitch_2d_formation_path", "")):
                    st.image(results["pitch_2d_formation_path"], caption="2D Tactical Pitch: Red = Team 0, Blue = Team 1", use_column_width=True)
                else:
                    st.info("Run analysis to generate 2D pitch formations.")

            with c_right:
                st.markdown("#### 🔥 2D Cumulative Match Heatmap")
                if os.path.exists(results.get("pitch_2d_all_path", "")):
                    st.image(results["pitch_2d_all_path"], caption="2D Top-Down Spatial Heatmap on Standard Pitch", use_column_width=True)
                else:
                    st.info("Run analysis to generate 2D pitch heatmap.")

            st.markdown("---")
            st.markdown("#### 🆚 2D Team-by-Team Heatmap Comparison")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("##### 🔴 Team 0 (2D Pitch Density)")
                if os.path.exists(results.get("pitch_2d_team0_path", "")):
                    st.image(results["pitch_2d_team0_path"], caption="Team 0 2D Heatmap", use_column_width=True)
            with t_col2:
                st.markdown("##### 🔵 Team 1 (2D Pitch Density)")
                if os.path.exists(results.get("pitch_2d_team1_path", "")):
                    st.image(results["pitch_2d_team1_path"], caption="Team 1 2D Heatmap", use_column_width=True)
        else:
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("#### 🏟️ Pitch Heatmap Overlay (Camera Perspective)")
                if os.path.exists(results.get("overlay_path", "")):
                    st.image(results["overlay_path"], caption="Cumulative Team Heatmaps Blended on Pitch", use_column_width=True)
                
            with c_right:
                st.markdown("#### 📍 Average Formations & Tactical Shape")
                if os.path.exists(results.get("formation_path", "")):
                    st.image(results["formation_path"], caption="Player Average Positions on Camera Frame", use_column_width=True)

            st.markdown("---")
            st.markdown("#### 🆚 Team-by-Team Camera Heatmaps")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("##### 🔴 Team 0 Density")
                if os.path.exists(results.get("team0_heatmap_path", "")):
                    st.image(results["team0_heatmap_path"], caption="Team 0 Camera Heatmap", use_column_width=True)
            with t_col2:
                st.markdown("##### 🔵 Team 1 Density")
                if os.path.exists(results.get("team1_heatmap_path", "")):
                    st.image(results["team1_heatmap_path"], caption="Team 1 Camera Heatmap", use_column_width=True)

    # TAB 3: PLAYER DEEP-DIVE
    with tab_player:
        st.subheader("Individual Player Spatial Intelligence")
        spatial_analytics: Optional[SpatialAnalytics] = results.get("spatial_analytics")
        pitch_view: Optional[PitchView2D] = results.get("pitch_view")
        first_frame = results.get("first_frame")

        if first_frame is None and os.path.exists("output/first_frame.png"):
            first_frame = cv2.imread("output/first_frame.png")

        if pitch_view is None:
            pitch_view = PitchView2D(1050, 680)

        if spatial_analytics is not None:
            player_list = sorted(list(spatial_analytics.player_tracks.keys()))
            if player_list:
                selected_pid = st.selectbox(
                    "Select Tracked Player ID:",
                    player_list,
                    format_func=lambda pid: f"Player #{pid} (Team {spatial_analytics.player_teams.get(pid, 0)})"
                )

                # Player Metrics
                team_id = spatial_analytics.player_teams.get(selected_pid, 0)
                positions = spatial_analytics.player_tracks.get(selected_pid, [])
                positions_2d = spatial_analytics.player_tracks_2d.get(selected_pid, [])
                avg_pos = spatial_analytics.get_average_positions().get(selected_pid, (0, 0))

                pm_col1, pm_col2, pm_col3, pm_col4 = st.columns(4)
                with pm_col1:
                    team_label = "🔴 Team 0" if team_id == 0 else "🔵 Team 1"
                    st.metric("Assigned Team", team_label)
                with pm_col2:
                    st.metric("Frames Tracked", f"{len(positions)} frames")
                with pm_col3:
                    presence = round((len(positions) / max(1, spatial_analytics.total_processed_frames)) * 100, 1)
                    st.metric("Presence Rate", f"{presence}%")
                with pm_col4:
                    st.metric("Avg Position (Camera X, Y)", f"{int(avg_pos[0])}, {int(avg_pos[1])}")

                st.markdown("---")
                p_view_choice = st.radio(
                    "Player Display View:",
                    ["📐 2D Top-Down Pitch View (Kuşbakışı)", "🏟️ Camera Perspective View"],
                    horizontal=True,
                    key="p_view_choice"
                )

                p_col1, p_col2 = st.columns(2)
                
                if "2D Top-Down" in p_view_choice:
                    with p_col1:
                        st.markdown(f"#### 🔥 Player #{selected_pid} 2D Heatmap")
                        p_heatmap_2d = pitch_view.generate_2d_heatmap(positions_2d, sigma=22.0)
                        st.image(
                            cv2.cvtColor(p_heatmap_2d, cv2.COLOR_BGR2RGB),
                            caption=f"2D Top-Down Heatmap for Player #{selected_pid}",
                            use_column_width=True
                        )

                    with p_col2:
                        st.markdown(f"#### 🏃 Player #{selected_pid} 2D Movement Trail")
                        p_trail_2d = pitch_view.draw_2d_player_trajectory(positions_2d, team_id=team_id)
                        st.image(
                            cv2.cvtColor(p_trail_2d, cv2.COLOR_BGR2RGB),
                            caption=f"2D Movement Path: Green = Start, Red = End",
                            use_column_width=True
                        )
                else:
                    if first_frame is not None:
                        with p_col1:
                            st.markdown(f"#### 🔥 Player #{selected_pid} Camera Heatmap")
                            player_heatmap_img = spatial_analytics.generate_player_overlay(
                                tracker_id=selected_pid,
                                background_frame=first_frame
                            )
                            st.image(
                                cv2.cvtColor(player_heatmap_img, cv2.COLOR_BGR2RGB),
                                caption=f"Heatmap & Avg Position for Player #{selected_pid}",
                                use_column_width=True
                            )

                        with p_col2:
                            st.markdown(f"#### 🏃 Player #{selected_pid} Movement Trajectory")
                            player_trail_img = spatial_analytics.generate_player_trajectory(
                                tracker_id=selected_pid,
                                background_frame=first_frame
                            )
                            st.image(
                                cv2.cvtColor(player_trail_img, cv2.COLOR_BGR2RGB),
                                caption=f"Movement Path on Camera Frame",
                                use_column_width=True
                            )
            else:
                st.info("No tracked players found in current run.")
        else:
            st.info("Run an analysis from the sidebar to inspect individual player heatmaps and movement paths.")

    # TAB 4: ROSTER & DATA
    with tab_data:
        st.subheader("Match Spatial Data Report")
        spatial_analytics = results.get("spatial_analytics")
        if spatial_analytics is not None:
            summary_data = spatial_analytics.get_player_summary()
            if summary_data:
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)
                
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Roster & Spatial Data (CSV)",
                    data=csv_data,
                    file_name="footballistic_match_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("Run an analysis from the sidebar to generate the spatial player dataset.")
else:
    st.info("👈 Select a video from the sidebar and click **'Run Match Analysis'** to start!")
