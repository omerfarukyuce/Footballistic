"""
Footballistic: Main Pipeline Entrypoint for Phase 1.
Processes match footage, tracks players and the ball, classifies teams,
and generates spatial heatmaps & average formations.
"""

import os
import argparse
from typing import Callable, Dict, Optional
import cv2
import numpy as np
from tqdm import tqdm

from src.detection.detector import FootballDetector
from src.team_assignment.color_assigner import TeamColorAssigner
from src.analysis.heatmap import SpatialAnalytics
from src.utils.video_utils import VideoHandler, create_video_writer
from src.utils.drawing import TacticalVisualizer

# Ensure Ultralytics config is local
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath(".ultralytics")


def run_pipeline(
    video_path: str = "data/sample_videos/sample.mp4",
    output_dir: str = "output",
    max_frames: int = 150,  # Process first 150 frames (~6 sec) for fast MVP verification
    model_name: str = "weights/yolov8m.pt",
    confidence: float = 0.35,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict:
    """
    Executes the full computer vision analysis pipeline.
    """
    print("=" * 60)
    print("⚽ FOOTBALLISTIC: MATCH ANALYTICS ENGINE (PHASE 1)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    video = VideoHandler(video_path)
    video_info = video.get_info()
    print(f"📹 Input Video: {video_path}")
    print(f"📊 Info: {video_info['width']}x{video_info['height']} @ {video_info['fps']} FPS | Total: {video_info['total_frames']} frames")
    
    frames_to_process = min(max_frames, video_info["total_frames"]) if max_frames > 0 else video_info["total_frames"]
    print(f"⏱️  Frames to analyze: {frames_to_process}")

    if progress_callback:
        progress_callback(0, frames_to_process, "Initializing Vision & Tracking Models...")

    # 1. Initialize modules
    print("\n🧠 Initializing Vision & Tracking Modules...")
    detector = FootballDetector(model_path=model_name, confidence=confidence)
    team_assigner = TeamColorAssigner()
    spatial_analytics = SpatialAnalytics(video_info["width"], video_info["height"])
    spatial_analytics.set_total_frames(frames_to_process)

    # 2. Step 1: Pre-fit Team Colors from initial frames
    print("👕 Calibrating Team Jersey Colors from sample frames...")
    if progress_callback:
        progress_callback(0, frames_to_process, "Calibrating Team Colors...")
        
    sampled_colors = []
    sample_stride = 5
    for f_idx, frame in video.frame_generator():
        if f_idx >= 30:  # Sample within the first 30 frames
            break
        if f_idx % sample_stride != 0:
            continue
        
        detection = detector.detect_and_track(frame, persist=False)
        for player in detection.players:
            color = team_assigner.extract_jersey_color(frame, player.bbox)
            if color is not None:
                sampled_colors.append(color)

    if len(sampled_colors) >= 2:
        team_assigner.fit_teams(sampled_colors)
        print(f"✅ Successfully calibrated 2 teams. Cluster centers found.")
    else:
        print("⚠️ Not enough player samples found for team clustering. Using default colors.")

    # 3. Setup Video Output Writer
    output_video_path = os.path.join(output_dir, "analyzed_match.mp4")
    writer = create_video_writer(
        output_video_path,
        video_info["fps"],
        video_info["width"],
        video_info["height"]
    )

    # 4. Step 2: Main Processing Loop
    print("\n🚀 Processing video frames (Tracking, Classification, Spatial Accumulation)...")
    first_frame_saved = None

    pbar = tqdm(total=frames_to_process, desc="Analyzing Match")
    for f_idx, frame in video.frame_generator():
        if f_idx >= frames_to_process:
            break

        if first_frame_saved is None:
            first_frame_saved = frame.copy()

        annotated_frame = frame.copy()
        
        # Detection & Persistent Multi-Object Tracking
        detection = detector.detect_and_track(frame, persist=True)

        # Process and draw players
        for player in detection.players:
            bbox = player.bbox
            t_id = player.tracker_id

            # Determine Team
            team_id = 0
            if team_assigner.team_kmeans is not None:
                color = team_assigner.extract_jersey_color(frame, bbox)
                if color is not None:
                    team_id = team_assigner.predict_team(color)

            # Record spatial coordinates
            if t_id is not None:
                spatial_analytics.add_position(t_id, bbox, team_id)

            # Draw tactical foot ellipse & ID
            tactical_color = TacticalVisualizer.TEAM_COLORS.get(team_id, (200, 200, 200))
            TacticalVisualizer.draw_player_ellipse(
                annotated_frame,
                bbox,
                tactical_color,
                tracker_id=t_id
            )

        # Draw ball pointer
        for ball in detection.balls:
            TacticalVisualizer.draw_ball_pointer(annotated_frame, ball.bbox)

        # Write frame
        writer.write(annotated_frame)
        pbar.update(1)

        if progress_callback and f_idx % 5 == 0:
            progress_callback(f_idx + 1, frames_to_process, f"Tracking players... Frame {f_idx + 1}/{frames_to_process}")

    pbar.close()
    writer.release()
    print(f"\n🎥 Analyzed video saved to: {output_video_path}")

    # 5. Step 3: Spatial Post-Processing (Heatmaps & Formations)
    print("\n🗺️ Generating Heatmaps & Average Formations...")
    if progress_callback:
        progress_callback(frames_to_process, frames_to_process, "Generating Heatmaps and Tactical Formations...")
    
    # Team 0 Heatmap
    team0_heatmap = spatial_analytics.generate_heatmap(team_id=0)
    team0_path = os.path.join(output_dir, "heatmap_team_0.png")
    cv2.imwrite(team0_path, team0_heatmap)

    # Team 1 Heatmap
    team1_heatmap = spatial_analytics.generate_heatmap(team_id=1)
    team1_path = os.path.join(output_dir, "heatmap_team_1.png")
    cv2.imwrite(team1_path, team1_heatmap)

    # Overlay Heatmap onto first frame for tactical context
    overlay_path = os.path.join(output_dir, "heatmap_pitch_overlay.png")
    if first_frame_saved is not None:
        combined_overlay = first_frame_saved.copy()
        nonzero_mask = np.any(team0_heatmap > 0, axis=-1) | np.any(team1_heatmap > 0, axis=-1)
        blended = cv2.addWeighted(team0_heatmap, 0.5, team1_heatmap, 0.5, 0)
        combined_overlay[nonzero_mask] = cv2.addWeighted(
            combined_overlay[nonzero_mask], 0.4,
            blended[nonzero_mask], 0.6,
            0
        )
        cv2.imwrite(overlay_path, combined_overlay)

    # Average Formations
    formation_path = os.path.join(output_dir, "average_positions.png")
    avg_positions = spatial_analytics.get_average_positions()
    if first_frame_saved is not None:
        formation_img = TacticalVisualizer.draw_average_positions(
            first_frame_saved,
            avg_positions,
            spatial_analytics.player_teams
        )
        cv2.imwrite(formation_path, formation_img)

    first_frame_path = os.path.join(output_dir, "first_frame.png")
    if first_frame_saved is not None:
        cv2.imwrite(first_frame_path, first_frame_saved)

    print(f"📊 Heatmaps saved to: {team0_path} & {team1_path}")
    print(f"📍 Average positions saved to: {formation_path}")
    print(f"\n✅ Total unique players tracked: {len(avg_positions)}")
    print("=" * 60)
    print("🎉 Phase 1 Pipeline Completed Successfully!")
    print("=" * 60)

    return {
        "output_video_path": output_video_path,
        "spatial_analytics": spatial_analytics,
        "first_frame": first_frame_saved,
        "first_frame_path": first_frame_path,
        "team0_heatmap_path": team0_path,
        "team1_heatmap_path": team1_path,
        "overlay_path": overlay_path,
        "formation_path": formation_path,
        "video_info": video_info,
        "total_players": len(avg_positions)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Footballistic Match Analytics")
    parser.add_argument("--video", type=str, default="data/sample_videos/sample.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--frames", type=int, default=150, help="Max frames to process (0 for all)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence")
    args = parser.parse_args()

    run_pipeline(video_path=args.video, output_dir=args.output, max_frames=args.frames, confidence=args.conf)
