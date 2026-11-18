"""
Footballistic: Main Pipeline Entrypoint.
Processes match footage with YOLO11, pitch boundary segmentation,
smooth player & ball tracking trails, and 2D top-down pitch analytics.
"""

import os
# Ensure Ultralytics config is local before importing detector or ultralytics
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath(".ultralytics")

from collections import defaultdict, deque
import math
import argparse
from typing import Callable, Dict, Optional
import cv2
import numpy as np
from tqdm import tqdm

from src.detection.detector import FootballDetector
from src.detection.pitch_detector import PitchDetector
from src.team_assignment.color_assigner import TeamColorAssigner
from src.analysis.heatmap import SpatialAnalytics
from src.homography.pitch_view import PitchView2D
from src.utils.video_utils import VideoHandler, create_video_writer
from src.utils.drawing import TacticalVisualizer


def run_pipeline(
    video_path: str = "data/sample_videos/sample.mp4",
    output_dir: str = "output",
    max_frames: int = 0,  # 0 for full video
    model_name: str = "weights/yolo11m.pt",
    confidence: float = 0.35,
    player_trail_len: int = 15,
    ball_trail_len: int = 35,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict:
    """
    Executes the full computer vision analysis pipeline.
    """
    print("=" * 60)
    print("⚽ FOOTBALLISTIC: MATCH ANALYTICS ENGINE")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    video = VideoHandler(video_path)
    video_info = video.get_info()
    print(f"📹 Input Video: {video_path}")
    print(f"📊 Info: {video_info['width']}x{video_info['height']} @ {video_info['fps']} FPS | Total: {video_info['total_frames']} frames ({video_info['duration_sec']:.1f}s)")
    
    frames_to_process = min(max_frames, video_info["total_frames"]) if max_frames > 0 else video_info["total_frames"]
    print(f"⏱️  Frames to analyze: {frames_to_process} ({frames_to_process / video_info['fps']:.1f}s)")

    if progress_callback:
        progress_callback(0, frames_to_process, "Initializing YOLO11 & Pitch Modules...")

    # 1. Initialize modules
    print(f"\n🧠 Initializing Detector ({model_name})...")
    detector = FootballDetector(model_path=model_name, confidence=confidence)
    pitch_detector = PitchDetector()
    team_assigner = TeamColorAssigner()
    pitch_view = PitchView2D(pitch_width=1050, pitch_height=680)
    spatial_analytics = SpatialAnalytics(video_info["width"], video_info["height"])
    spatial_analytics.set_total_frames(frames_to_process)

    # 2. Step 1: Pre-fit Team Colors from initial frames
    # Use is_strictly_inside_pitch so coaches / staff at the touchline
    # are excluded from the jersey-colour calibration pass.
    print("👕 Calibrating Team Jersey Colors from sample frames...")
    if progress_callback:
        progress_callback(0, frames_to_process, "Calibrating Team Colors...")

    sampled_colors = []
    sample_stride = 4
    for f_idx, frame in video.frame_generator():
        if f_idx >= 90:   # Scan first 90 frames for a richer colour sample
            break
        if f_idx % sample_stride != 0:
            continue

        pitch_detector.get_pitch_mask(frame)
        detection = detector.detect_and_track(frame, persist=False)
        for player in detection.players:
            # Use STRICT boundary so touchline coaches are excluded
            foot_pos = ((player.bbox[0] + player.bbox[2]) / 2.0, player.bbox[3])
            if not pitch_detector.is_strictly_inside_pitch(foot_pos):
                continue

            # Skip very small bboxes (distant players sampled at calibration
            # time are noisy; include them only if large enough for a stable crop)
            bw = player.bbox[2] - player.bbox[0]
            bh = player.bbox[3] - player.bbox[1]
            if bw < 15 or bh < 25:
                continue

            color = team_assigner.extract_jersey_color(frame, player.bbox)
            if color is not None:
                sampled_colors.append(color)

    if len(sampled_colors) >= 2:
        team_assigner.fit_teams(sampled_colors)
        print(f"✅ Successfully calibrated 2 teams. Cluster centers found.")
    else:
        print("⚠️ Not enough player samples found for team clustering. Using default colors.")

    # 3. Setup Video Output Writer (PyAV H.264)
    output_video_path = os.path.join(output_dir, "analyzed_match.mp4")
    writer = create_video_writer(
        output_video_path,
        video_info["fps"],
        video_info["width"],
        video_info["height"]
    )

    # Motion Trails Buffers & Stable Team Voting
    player_trails = defaultdict(lambda: deque(maxlen=player_trail_len))
    player_team_votes = defaultdict(lambda: defaultdict(int))
    ball_trail = deque(maxlen=ball_trail_len)
    last_valid_ball_pos: Optional[Tuple[float, float]] = None

    # 4. Step 2: Main Processing Loop
    print("\n🚀 Processing video frames (Tracking, Classification, Motion Trails)...")
    first_frame_saved = None

    pbar = tqdm(total=frames_to_process, desc="Analyzing Match")
    for f_idx, frame in video.frame_generator():
        if f_idx >= frames_to_process:
            break

        if first_frame_saved is None:
            first_frame_saved = frame.copy()

        annotated_frame = frame.copy()

        # Update pitch boundary mask
        pitch_detector.get_pitch_mask(frame)
        
        # Detection & Persistent Multi-Object Tracking
        detection = detector.detect_and_track(frame, persist=True)

        # ----------------- PLAYERS -----------------
        current_frame_players = []
        active_ids = set()

        for player in detection.players:
            bbox = player.bbox
            t_id = player.tracker_id
            bottom_center = ((bbox[0] + bbox[2]) / 2.0, bbox[3])

            # 1. Filter out people outside the pitch boundary (stands, bench, coaches)
            #    tolerance_px=0.0 ensures nobody outside the touchline is admitted.
            if not pitch_detector.is_inside_pitch(bottom_center, tolerance_px=0.0):
                continue

            # 2. Skip bboxes too small to classify reliably (very distant / noise)
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            if bw < 8 or bh < 16:
                continue

            # Map to 2D top-down pitch early for positional intelligence (GK zone detection)
            pos_2d = pitch_view.map_camera_to_2d(
                bottom_center[0], bottom_center[1],
                cam_w=video_info["width"], cam_h=video_info["height"]
            )

            # 3. Team colour classification with outlier rejection
            #    predict_team_with_confidence returns team_id = -1 for non-standard jersey colors
            team_id = -1
            if team_assigner.team_kmeans is not None:
                color = team_assigner.extract_jersey_color(frame, bbox)
                if color is not None:
                    pred_team, conf = team_assigner.predict_team_with_confidence(color)
                    if pred_team == -1:
                        # Colour outlier: Check if located in a goalkeeper defending zone
                        # Pitch X: field_x1=40, field_x2=1010; Field Y: field_y1=40, field_y2=640 (Center=340).
                        # Defending left goal (<220px in 2D and within penalty box Y range [200, 480]) -> Team 0 GK
                        # Defending right goal (>830px in 2D and within penalty box Y range [200, 480]) -> Team 1 GK
                        in_left_gk_zone = (pos_2d[0] < 220) and (200.0 <= pos_2d[1] <= 480.0)
                        in_right_gk_zone = (pos_2d[0] > 830) and (200.0 <= pos_2d[1] <= 480.0)
                        if in_left_gk_zone:
                            pred_team = 0
                            conf = 0.90
                        elif in_right_gk_zone:
                            pred_team = 1
                            conf = 0.90
                        else:
                            # In midfield, touchline, or sideline -> Outlier is a Referee or Coach/Staff. Skip entirely!
                            continue

                    if t_id is not None:
                        # Confidence-weighted temporal majority voting
                        weight = 3 if conf > 0.70 else (2 if conf > 0.55 else 1)
                        player_team_votes[t_id][pred_team] += weight
                        team_id = max(player_team_votes[t_id], key=player_team_votes[t_id].get)
                    else:
                        team_id = pred_team
                else:
                    # No colour extracted: use previous vote if available
                    if t_id is not None and player_team_votes[t_id]:
                        team_id = max(player_team_votes[t_id], key=player_team_votes[t_id].get)
                    else:
                        continue  # Cannot classify -> skip
            elif t_id is not None and player_team_votes[t_id]:
                team_id = max(player_team_votes[t_id], key=player_team_votes[t_id].get)
            else:
                team_id = 0

            if team_id < 0:
                continue  # Safety: drop any remaining unclassified detections

            # 4. Record spatial coordinates and trail
            if t_id is not None:
                spatial_analytics.add_position(t_id, bbox, team_id, pos_2d=pos_2d, frame_idx=f_idx)
                active_ids.add(t_id)

                # Prevent motion-trail glitches from large ID-swap jumps
                if len(player_trails[t_id]) > 0:
                    prev_p = player_trails[t_id][-1]
                    dist = math.hypot(bottom_center[0] - prev_p[0], bottom_center[1] - prev_p[1])
                    if dist > 80.0:
                        player_trails[t_id].clear()

                player_trails[t_id].append(bottom_center)

            current_frame_players.append((player, team_id))

        # ----------------- BALL -----------------
        # Filter ball candidates: must be inside pitch, reasonable size, aspect ratio
        valid_balls = pitch_detector.filter_ball_candidates(detection.balls)
        best_ball = None

        if valid_balls:
            if last_valid_ball_pos is not None:
                # Find candidate closest to previous known ball position
                def ball_dist(b):
                    cx = (b.bbox[0] + b.bbox[2]) / 2.0
                    cy = (b.bbox[1] + b.bbox[3]) / 2.0
                    return math.hypot(cx - last_valid_ball_pos[0], cy - last_valid_ball_pos[1])

                sorted_by_dist = sorted(valid_balls, key=ball_dist)
                closest = sorted_by_dist[0]
                d = ball_dist(closest)
                # If within plausible physical jump limit (150px)
                if d < 160.0:
                    best_ball = closest
                else:
                    # Best confidence candidate
                    best_ball = max(valid_balls, key=lambda b: b.confidence)
                    ball_trail.clear()  # Reset trail on large jump
            else:
                best_ball = max(valid_balls, key=lambda b: b.confidence)

        if best_ball is not None:
            ball_center = ((best_ball.bbox[0] + best_ball.bbox[2]) / 2.0, (best_ball.bbox[1] + best_ball.bbox[3]) / 2.0)
            
            # Check jump from previous point
            if len(ball_trail) > 0:
                dist = math.hypot(ball_center[0] - ball_trail[-1][0], ball_center[1] - ball_trail[-1][1])
                if dist > 140.0:
                    ball_trail.clear()

            ball_trail.append(ball_center)
            last_valid_ball_pos = ball_center

        # ----------------- DRAW LAYERS -----------------
        # Layer 1: Player motion trails (ONLY active players, team colored)
        TacticalVisualizer.draw_player_trails(
            annotated_frame,
            player_trails,
            spatial_analytics.player_teams,
            active_ids=active_ids
        )

        # Layer 2: Ball motion trail (Pure WHITE line)
        TacticalVisualizer.draw_ball_trail(annotated_frame, ball_trail)

        # Layer 3: Tactical foot ellipses and ID tags
        for player, team_id in current_frame_players:
            tactical_color = TacticalVisualizer.TEAM_COLORS.get(team_id, (200, 200, 200))
            TacticalVisualizer.draw_player_ellipse(
                annotated_frame,
                player.bbox,
                tactical_color,
                tracker_id=player.tracker_id
            )

        # Layer 4: Ball tracker pointer
        if best_ball is not None:
            TacticalVisualizer.draw_ball_pointer(annotated_frame, best_ball.bbox)

        # Write frame
        writer.write(annotated_frame)
        pbar.update(1)

        if progress_callback and f_idx % 10 == 0:
            progress_callback(f_idx + 1, frames_to_process, f"Tracking Match Action... Frame {f_idx + 1}/{frames_to_process}")

    pbar.close()
    writer.release()
    print(f"\n🎥 Analyzed video saved to: {output_video_path}")

    # 5. Step 3: Spatial Post-Processing
    print("\n🗺️ Generating Heatmaps & Formations (Camera & 2D Top-Down)...")
    if progress_callback:
        progress_callback(frames_to_process, frames_to_process, "Consolidating tracks and generating 2D analytics...")

    # 5a. Consolidate fragmented tracklets (camera pans / occlusions)
    #     Wider gap and bigger spatial tolerance now that we have track_buffer=90
    spatial_analytics.consolidate_tracks(max_frame_gap=60, max_dist_px=120.0)

    # 5b. Re-apply final consolidated team assignments from vote history
    for tid in list(spatial_analytics.player_teams.keys()):
        if tid in player_team_votes and player_team_votes[tid]:
            spatial_analytics.player_teams[tid] = max(
                player_team_votes[tid], key=player_team_votes[tid].get
            )

    # 5c. Re-assign outlier-labelled tracks that slipped through
    #     (edge-case: a GK whose distinctive kit was classified as outlier
    #      during consolidation; use position to decide their team)
    for tid in list(spatial_analytics.player_teams.keys()):
        if spatial_analytics.player_teams[tid] < 0:
            pts = spatial_analytics.player_tracks_2d.get(tid, [])
            if pts:
                avg_x = float(np.mean([p[0] for p in pts]))
                avg_y = float(np.mean([p[1] for p in pts]))
                # GK must be located in defending goal area in both X and Y
                if 200.0 <= avg_y <= 480.0 and (avg_x < 220 or avg_x > 830):
                    reassigned = team_assigner.infer_goalkeeper_team(avg_x)
                    spatial_analytics.player_teams[tid] = reassigned
                else:
                    # Referee or coach outlier -> remove
                    spatial_analytics.player_teams.pop(tid, None)
                    spatial_analytics.player_tracks.pop(tid, None)
                    spatial_analytics.player_tracks_2d.pop(tid, None)
                    spatial_analytics.player_frames.pop(tid, None)
            else:
                # Cannot determine position → remove
                spatial_analytics.player_teams.pop(tid, None)
                spatial_analytics.player_tracks.pop(tid, None)
                spatial_analytics.player_tracks_2d.pop(tid, None)
                spatial_analytics.player_frames.pop(tid, None)

    # 5d. Strictly enforce FIFA squad limit (11 per team, 1 GK + 10 outfield)
    spatial_analytics.enforce_team_squad_limits(max_players_per_team=11)

    # Minimum frames to be considered a stable (non-ghost) detection
    MIN_STABLE_FRAMES = 8

    # Camera Perspective Heatmaps
    team0_heatmap = spatial_analytics.generate_heatmap(team_id=0)
    team0_path = os.path.join(output_dir, "heatmap_team_0.png")
    cv2.imwrite(team0_path, team0_heatmap)

    team1_heatmap = spatial_analytics.generate_heatmap(team_id=1)
    team1_path = os.path.join(output_dir, "heatmap_team_1.png")
    cv2.imwrite(team1_path, team1_heatmap)

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

    # Camera Perspective Average Formations (stable players only)
    formation_path = os.path.join(output_dir, "average_positions.png")
    avg_positions = spatial_analytics.get_average_positions(min_frames=MIN_STABLE_FRAMES)
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

    # ----------------- 2D TOP-DOWN RECTANGULAR PITCH ARTIFACTS -----------------
    # 2D Average Formations (stable players only)
    avg_positions_2d = spatial_analytics.get_average_positions_2d(min_frames=MIN_STABLE_FRAMES)
    pitch_2d_formation_img = pitch_view.draw_2d_formations(avg_positions_2d, spatial_analytics.player_teams)
    pitch_2d_formation_path = os.path.join(output_dir, "pitch_2d_formation.png")
    cv2.imwrite(pitch_2d_formation_path, pitch_2d_formation_img)

    # 2D Team 0 Heatmap (stable players only)
    team0_pts_2d = spatial_analytics.get_2d_positions(team_id=0, min_frames=MIN_STABLE_FRAMES)
    pitch_2d_team0_img = pitch_view.generate_2d_heatmap(team0_pts_2d)
    pitch_2d_team0_path = os.path.join(output_dir, "pitch_2d_heatmap_team0.png")
    cv2.imwrite(pitch_2d_team0_path, pitch_2d_team0_img)

    # 2D Team 1 Heatmap (stable players only)
    team1_pts_2d = spatial_analytics.get_2d_positions(team_id=1, min_frames=MIN_STABLE_FRAMES)
    pitch_2d_team1_img = pitch_view.generate_2d_heatmap(team1_pts_2d)
    pitch_2d_team1_path = os.path.join(output_dir, "pitch_2d_heatmap_team1.png")
    cv2.imwrite(pitch_2d_team1_path, pitch_2d_team1_img)

    # 2D Combined Heatmap (stable players only)
    all_pts_2d = spatial_analytics.get_2d_positions(min_frames=MIN_STABLE_FRAMES)
    pitch_2d_all_img = pitch_view.generate_2d_heatmap(all_pts_2d)
    pitch_2d_all_path = os.path.join(output_dir, "pitch_2d_heatmap_all.png")
    cv2.imwrite(pitch_2d_all_path, pitch_2d_all_img)

    # Save spatial dataset to JSON for instant dashboard reloading
    spatial_data_path = os.path.join(output_dir, "spatial_data.json")
    spatial_analytics.save_to_json(spatial_data_path)

    print(f"📊 2D Rectangular Heatmaps & Formations saved to {output_dir}")
    print(f"✅ Total unique players tracked: {len(avg_positions)}")
    print("=" * 60)
    print("🎉 Pipeline Completed Successfully!")
    print("=" * 60)

    return {
        "output_video_path": output_video_path,
        "spatial_analytics": spatial_analytics,
        "spatial_data_path": spatial_data_path,
        "pitch_view": pitch_view,
        "first_frame": first_frame_saved,
        "first_frame_path": first_frame_path,
        "team0_heatmap_path": team0_path,
        "team1_heatmap_path": team1_path,
        "overlay_path": overlay_path,
        "formation_path": formation_path,
        "pitch_2d_formation_path": pitch_2d_formation_path,
        "pitch_2d_team0_path": pitch_2d_team0_path,
        "pitch_2d_team1_path": pitch_2d_team1_path,
        "pitch_2d_all_path": pitch_2d_all_path,
        "video_info": video_info,
        "total_players": len(avg_positions)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Footballistic Match Analytics")
    parser.add_argument("--video", type=str, default="data/sample_videos/sample.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--frames", type=int, default=0, help="Max frames to process (0 for all)")
    parser.add_argument("--model", type=str, default="weights/yolo11m.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence")
    args = parser.parse_args()

    run_pipeline(video_path=args.video, output_dir=args.output, max_frames=args.frames, model_name=args.model, confidence=args.conf)
