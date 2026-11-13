"""
Footballistic: Streamlit Interactive Football Match Analytics Dashboard.
"""

import os
import tempfile
import cv2
import pandas as pd
import streamlit as st
import numpy as np

from main import run_pipeline
from src.analysis.heatmap import SpatialAnalytics

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
    .metric-card {
        background-color: #1E222A;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 15px;
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
    if "is_analyzing" not in st.session_state:
        st.session_state["is_analyzing"] = False


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
            st.success(f"Loaded demo clip: `sample.mp4` (1080p, 25 FPS)")
        else:
            st.warning("Sample clip not found locally. Please upload a video.")
    else:
        uploaded_file = st.file_uploader(
            "Upload Match Video (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"]
        )
        if uploaded_file is not None:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            input_video_path = tfile.name
            st.success("Custom video uploaded successfully!")

    st.markdown("---")
    st.subheader("Analysis Parameters")
    
    max_frames = st.slider(
        "Frames to Analyze (1 sec ≈ 25 frames):",
        min_value=25,
        max_value=500,
        value=75,
        step=25,
        help="Higher frame counts yield longer tracking but take more processing time."
    )

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
            with st.spinner("Processing match video with YOLOv8 & ByteTrack..."):
                results = run_pipeline(
                    video_path=input_video_path,
                    output_dir="output",
                    max_frames=max_frames,
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
    # If existing output files exist on disk from previous runs, load them as initial preview!
    existing_video = "output/analyzed_match.mp4"
    if os.path.exists(existing_video) and os.path.exists("output/heatmap_pitch_overlay.png"):
        st.info("💡 Previous analysis found on disk! Run a new analysis from the sidebar anytime or explore the saved session below.")
        # Minimal mock results for initial view
        results = {
            "output_video_path": existing_video,
            "team0_heatmap_path": "output/heatmap_team_0.png",
            "team1_heatmap_path": "output/heatmap_team_1.png",
            "overlay_path": "output/heatmap_pitch_overlay.png",
            "formation_path": "output/average_positions.png",
            "first_frame_path": "output/first_frame.png",
            "total_players": 43,
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
        st.metric("Active Teams", "2 Teams + Ref")

    # Tabs
    tab_video, tab_heatmaps, tab_player, tab_data = st.tabs([
        "🎥 Video Replay & Tracking",
        "🗺️ Team Heatmaps & Formations",
        "👤 Player Deep-Dive",
        "📋 Roster & Spatial Data"
    ])

    # TAB 1: VIDEO REPLAY
    with tab_video:
        st.subheader("Annotated Match Video")
        st.markdown(
            "Every detected player is tracked with a continuous ID, tactical foot ellipse, and team color. The ball is marked with an inverted tracker pointer."
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
        st.subheader("Pitch Spatial Density & Tactical Formations")
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### 🏟️ Pitch Heatmap Overlay")
            st.image(results["overlay_path"], caption="Cumulative Team Heatmaps Blended on Pitch", use_column_width=True)
            
        with c_right:
            st.markdown("#### 📍 Average Formations & Tactical Shape")
            st.image(results["formation_path"], caption="Player Average Position Coordinates & Team Clusters", use_column_width=True)

        st.markdown("---")
        st.markdown("#### 🆚 Team-by-Team Heatmap Comparison")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown("##### 🔴 Team 0 Density")
            if os.path.exists(results["team0_heatmap_path"]):
                st.image(results["team0_heatmap_path"], caption="Team 0 Heatmap", use_column_width=True)
        with t_col2:
            st.markdown("##### 🟢 Team 1 Density")
            if os.path.exists(results["team1_heatmap_path"]):
                st.image(results["team1_heatmap_path"], caption="Team 1 Heatmap", use_column_width=True)

    # TAB 3: PLAYER DEEP-DIVE
    with tab_player:
        st.subheader("Individual Player Spatial Intelligence")
        spatial_analytics: Optional[SpatialAnalytics] = results.get("spatial_analytics")
        first_frame = results.get("first_frame")

        if first_frame is None and os.path.exists("output/first_frame.png"):
            first_frame = cv2.imread("output/first_frame.png")

        if spatial_analytics is not None and first_frame is not None:
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
                avg_pos = spatial_analytics.get_average_positions().get(selected_pid, (0, 0))

                pm_col1, pm_col2, pm_col3, pm_col4 = st.columns(4)
                with pm_col1:
                    st.metric("Assigned Team", f"Team {team_id}")
                with pm_col2:
                    st.metric("Frames Tracked", f"{len(positions)} frames")
                with pm_col3:
                    presence = round((len(positions) / max(1, spatial_analytics.total_processed_frames)) * 100, 1)
                    st.metric("Presence Rate", f"{presence}%")
                with pm_col4:
                    st.metric("Avg Position (X, Y)", f"{int(avg_pos[0])}, {int(avg_pos[1])}")

                st.markdown("---")
                p_col1, p_col2 = st.columns(2)
                
                with p_col1:
                    st.markdown(f"#### 🔥 Player #{selected_pid} Heatmap")
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
                        caption=f"Movement Path: Green = Start, Red = Current",
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
