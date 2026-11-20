"""
Footballistic: Main Pipeline Entrypoint (2-Pass Broadcast Analytics Engine).

Pass 1 — High-Recall Detection & Camera-Motion-Compensated Tracking:
  - Non-convex pitch segmentation crops out stadium stands/crowd & dugouts.
  - High-resolution pitch-cropped YOLO (imgsz=1280, conf=0.14 players / 0.06 ball)
    catches BOTH close and distant far-touchline players as well as small footballs.
  - Pitch-only detections feed into FootballBoTSORT (BoT-SORT + sparseOptFlow GMC
    camera motion compensation + Stage-3 Foot-Center & Lab Jersey Re-ID), keeping
    player IDs locked across fast camera pans and zero-IoU distant sprints.
  - Dedicated BallTracker (GMC-Kalman + 1:1 native-res 320px ROI zoom + local GMC
    motion/white-blob recovery) tracks the football continuously.

Global Post-Processing — Broadcast-Grade Accuracy:
  - Camera-motion-compensated global tracklet consolidation + long-gap FIFA-prior
    same-team re-stitching (eliminating duplicate IDs across camera pans/exits).
  - Track-level grass-excluded jersey classification (referees & dugout staff cleanly
    rejected; goalkeepers identified by defensive extremity and assigned to the team
    defending that half).
  - Clean low jersey-style IDs (#1 GK + #2..#11 for Team 0; #13 GK + #12,#14..#22 for Team 1).
  - GMC-compensated gap interpolation for players (<=12 frames) & ball (<=24 frames).

Pass 2 — Fast Video Rendering & 2D Spatial Analytics:
  - Renders analyzed_match.mp4 with 100% consistent team colors, stable IDs, GMC-pinned
    player trails, and a crisp pure WHITE football trajectory trail + pointer.
  - Exports 2D bird's-eye formations, heatmaps, and spatial_data.json.
"""

import os
# Ensure Ultralytics config is local before importing detector or ultralytics
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath(".ultralytics")

from collections import defaultdict, deque
import math
import argparse
from typing import Callable, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from src.detection.detector import FootballDetector
from src.detection.pitch_detector import PitchDetector
from src.team_assignment.color_assigner import TeamColorAssigner
from src.tracking.football_tracker import (
    FootballBoTSORT,
    consolidate_player_tracks,
    interpolate_player_gaps,
    project_point_across_frames,
)
from src.tracking.ball_tracker import BallTracker
from src.analysis.heatmap import SpatialAnalytics
from src.homography.pitch_view import PitchView2D
from src.utils.video_utils import VideoHandler, create_video_writer
from src.utils.drawing import TacticalVisualizer


def _stitch_same_team_non_overlapping(
    tracks: Dict[int, Dict[int, List[float]]],
    track_teams: Dict[int, int],
    goalkeeper_ids: Set[int],
    frame_warps: Dict[int, np.ndarray],
    max_outfield_per_team: int = 10,
    max_long_gap: int = 240
) -> Tuple[Dict[int, Dict[int, List[float]]], Dict[int, int], Set[int]]:
    """
    FIFA 11-Player Prior Re-Stitcher:
    Since each team has at most 10 outfield players on the pitch, if a team still has
    more than 10 outfield tracklets after local consolidation (e.g. a player left the
    camera frame during a pan and re-entered 4-9 seconds later), stitch the best
    non-overlapping same-team pairs using GMC-projected camera proximity.
    """
    tracks = {tid: dict(fmap) for tid, fmap in tracks.items()}
    teams = dict(track_teams)
    gks = set(goalkeeper_ids)

    for team_id in (0, 1):
        while True:
            outfield = [
                tid for tid, t in teams.items()
                if t == team_id and tid not in gks and tid in tracks
            ]
            if len(outfield) <= max_outfield_per_team:
                break

            outfield.sort(key=lambda tid: min(tracks[tid].keys()))
            best_pair = None
            best_score = float("inf")

            for i, id_b in enumerate(outfield):
                fb_keys = sorted(tracks[id_b].keys())
                f_start_b = fb_keys[0]
                b_start = tracks[id_b][f_start_b]
                pos_b = ((b_start[0] + b_start[2]) * 0.5, b_start[3])

                for id_a in outfield[:i]:
                    fa_keys = sorted(tracks[id_a].keys())
                    f_end_a = fa_keys[-1]
                    gap = f_start_b - f_end_a
                    if gap < 1 or gap > max_long_gap:
                        continue

                    b_end = tracks[id_a][f_end_a]
                    pos_a = ((b_end[0] + b_end[2]) * 0.5, b_end[3])
                    proj_a = project_point_across_frames(pos_a, f_end_a, f_start_b, frame_warps)
                    dist_px = math.hypot(pos_b[0] - proj_a[0], pos_b[1] - proj_a[1])

                    # Generous physical envelope for long off-screen/pan gaps
                    max_dist = min(900.0, 90.0 + 11.0 * gap)
                    if dist_px > max_dist:
                        continue

                    score = (dist_px / max_dist) + 0.15 * (gap / max_long_gap)
                    if score < best_score:
                        best_score = score
                        best_pair = (id_a, id_b)

            if best_pair is None:
                break

            id_a, id_b = best_pair
            tracks[id_a].update(tracks[id_b])
            del tracks[id_b]
            teams.pop(id_b, None)

    return tracks, teams, gks


def _reindex_clean_squad_ids(
    tracks: Dict[int, Dict[int, List[float]]],
    track_teams: Dict[int, int],
    goalkeeper_ids: Set[int],
) -> Tuple[Dict[int, Dict[int, List[float]]], Dict[int, int], Set[int]]:
    """
    Remaps surviving player tracks to clean, readable jersey-style IDs:
      - Team 0: #1 (Goalkeeper), #2..#11 (Top 10 Outfielders)
      - Team 1: #13 (Goalkeeper), #12, #14..#22 (Top 10 Outfielders)
    Matches PitchView2D & Streamlit dashboard goalkeeper conventions (#1, #13).
    """
    clean_tracks: Dict[int, Dict[int, List[float]]] = {}
    clean_teams: Dict[int, int] = {}
    clean_gks: Set[int] = set()

    team_slots = {
        0: (1, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        1: (13, [12, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
    }

    for team_id in (0, 1):
        gk_slot, out_slots = team_slots[team_id]
        t_gks = [
            tid for tid in tracks
            if track_teams.get(tid) == team_id and tid in goalkeeper_ids
        ]
        t_out = [
            tid for tid in tracks
            if track_teams.get(tid) == team_id and tid not in goalkeeper_ids
        ]

        # Keep the longest real GK track (if visible on camera)
        t_gks.sort(key=lambda tid: len(tracks[tid]), reverse=True)
        if t_gks:
            gk_orig = t_gks[0]
            clean_tracks[gk_slot] = tracks[gk_orig]
            clean_teams[gk_slot] = team_id
            clean_gks.add(gk_slot)

        # Keep top 10 outfield players (sorted by duration, then earliest frame for determinism)
        t_out.sort(key=lambda tid: (-len(tracks[tid]), min(tracks[tid].keys())))
        for slot_id, orig_tid in zip(out_slots, t_out[:10]):
            clean_tracks[slot_id] = tracks[orig_tid]
            clean_teams[slot_id] = team_id

    return clean_tracks, clean_teams, clean_gks


def run_pipeline(
    video_path: str = "data/sample_videos/sample.mp4",
    output_dir: str = "output",
    max_frames: int = 0,  # 0 for full video
    model_name: str = "weights/yolo11m.pt",
    confidence: float = 0.14,
    ball_confidence: float = 0.06,
    imgsz: int = 1280,
    player_trail_len: int = 18,
    ball_trail_len: int = 38,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict:
    """
    Executes the full 2-pass computer vision analysis pipeline.
    """
    print("=" * 64)
    print("⚽ FOOTBALLISTIC: HIGH-PRECISION MATCH ANALYTICS ENGINE")
    print("=" * 64)

    os.makedirs(output_dir, exist_ok=True)
    video = VideoHandler(video_path)
    video_info = video.get_info()
    cam_w, cam_h = video_info["width"], video_info["height"]
    print(f"📹 Input Video: {video_path}")
    print(f"📊 Info: {cam_w}x{cam_h} @ {video_info['fps']} FPS | Total: {video_info['total_frames']} frames ({video_info['duration_sec']:.1f}s)")

    frames_to_process = min(max_frames, video_info["total_frames"]) if max_frames > 0 else video_info["total_frames"]
    print(f"⏱️  Frames to analyze: {frames_to_process} ({frames_to_process / video_info['fps']:.1f}s)")
    print(f"🔍 Detector: {model_name} | imgsz={imgsz} | player_conf={confidence} | ball_conf={ball_confidence}")

    if progress_callback:
        progress_callback(0, frames_to_process, "Initializing High-Res Detector, GMC BoT-SORT & Ball Tracker...")

    # 1. Initialize modules
    detector = FootballDetector(
        model_path=model_name,
        confidence=confidence,
        ball_confidence=ball_confidence,
        imgsz=imgsz
    )
    pitch_detector = PitchDetector()
    team_assigner = TeamColorAssigner()
    player_tracker = FootballBoTSORT(config_path="configs/football_botsort.yaml")
    ball_tracker = BallTracker()
    pitch_view = PitchView2D(pitch_width=1050, pitch_height=680)
    spatial_analytics = SpatialAnalytics(cam_w, cam_h)
    spatial_analytics.set_total_frames(frames_to_process)

    # =========================================================================
    # PASS 1: HIGH-RES DETECTION + GMC BoT-SORT TRACKING + BALL TRACKING
    # =========================================================================
    print("\n🚀 Pass 1/2: High-Resolution Pitch Detection, GMC Tracking & Ball Recovery...")
    if progress_callback:
        progress_callback(0, frames_to_process, "Pass 1/2: Detecting & Tracking Players + Ball...")

    raw_player_tracks: Dict[int, Dict[int, List[float]]] = defaultdict(dict)
    raw_player_colors: Dict[int, List[np.ndarray]] = defaultdict(list)
    frame_warps: Dict[int, np.ndarray] = {}
    frame_y_bounds: Dict[int, Optional[Tuple[int, int]]] = {}
    frame_top_edges: Dict[int, Optional[np.ndarray]] = {}
    first_frame_saved: Optional[np.ndarray] = None

    pbar = tqdm(total=frames_to_process, desc="Pass 1 (Track)", unit="fr")

    for f_idx, frame in video.frame_generator():
        if f_idx >= frames_to_process:
            break

        if first_frame_saved is None:
            first_frame_saved = frame.copy()

        # 1a. Non-convex pitch segmentation + white near-touchline model (excludes crowd, dugouts & coaches!)
        pitch_detector.get_pitch_mask(frame)
        pitch_detector.update_near_touchline(frame)
        crop_bounds = pitch_detector.get_pitch_crop_bounds(frame, top_pad=55, bottom_pad=8)
        frame_y_bounds[f_idx] = pitch_detector.pitch_y_bounds
        frame_top_edges[f_idx] = (
            pitch_detector.top_edge_by_x[::10].copy()
            if pitch_detector.top_edge_by_x is not None
            else None
        )

        # 1b. High-res pitch-cropped YOLO detection (players + football)
        detection = detector.detect_frame(frame, crop_bounds=crop_bounds)

        # 1c. Filter to ONLY inside-pitch player candidates BEFORE tracking
        #     (ensures zero crowd spectators, coaches below the touchline, or dugout staff enter the tracker!)
        pitch_bboxes: List[List[float]] = []
        pitch_confs: List[float] = []
        pitch_bgrs: List[Optional[np.ndarray]] = []
        pitch_labs: List[Optional[np.ndarray]] = []

        for player in detection.players:
            bbox = player.bbox
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            # Keep tiny far-touchline players (down to 7x14 px), drop tiny noise specks
            if bw < 7 or bh < 14 or bh > cam_h * 0.45:
                continue

            foot = ((bbox[0] + bbox[2]) * 0.5, bbox[3])
            if not pitch_detector.is_inside_pitch(foot, tolerance_px=0.0):
                continue
            if pitch_detector.is_outside_near_touchline(foot, bh):
                continue

            bgr = team_assigner.extract_jersey_color(frame, bbox)
            lab = team_assigner.bgr_to_lab(bgr) if bgr is not None else None
            pitch_bboxes.append(bbox)
            pitch_confs.append(player.confidence)
            pitch_bgrs.append(bgr)
            pitch_labs.append(lab)

        # 1d. Update Camera-Motion-Compensated BoT-SORT (+ Stage-3 Lab Re-ID)
        tracked_players, warp_matrix = player_tracker.update_players(
            frame, pitch_bboxes, pitch_confs, det_labs=pitch_labs
        )
        frame_warps[f_idx] = warp_matrix

        tracked_bboxes_this_frame: List[List[float]] = []
        for tid, t_bbox, _score, det_idx in tracked_players:
            raw_player_tracks[tid][f_idx] = t_bbox
            tracked_bboxes_this_frame.append(t_bbox)
            if 0 <= det_idx < len(pitch_bgrs) and pitch_bgrs[det_idx] is not None:
                raw_player_colors[tid].append(pitch_bgrs[det_idx])
            else:
                bgr_fb = team_assigner.extract_jersey_color(frame, t_bbox)
                if bgr_fb is not None:
                    raw_player_colors[tid].append(bgr_fb)

        # 1e. Update Dedicated Ball Tracker (YOLO + GMC-Kalman + 1:1 ROI Zoom + Motion Blob)
        valid_balls = pitch_detector.filter_ball_candidates(detection.balls)
        ball_tracker.update(
            frame=frame,
            frame_idx=f_idx,
            valid_balls=valid_balls,
            warp_matrix=warp_matrix,
            detector=detector,
            pitch_detector=pitch_detector,
            player_bboxes=tracked_bboxes_this_frame
        )

        pbar.update(1)
        if progress_callback and f_idx % 5 == 0:
            # Pass 1 = 0..85% of progress bar
            progress_callback(
                int((f_idx + 1) * 0.85),
                frames_to_process,
                f"Pass 1/2 Tracking Players & Ball... Frame {f_idx + 1}/{frames_to_process}"
            )

    pbar.close()
    print(f"📦 Raw pitch tracks before consolidation: {len(raw_player_tracks)}")

    # =========================================================================
    # GLOBAL POST-PROCESSING: CONSOLIDATE + TEAM/GK/REF CLASSIFY + SMOOTH BALL
    # =========================================================================
    print("\n🧠 Global Post-Processing: GMC Track Stitching, Team/GK Classification & Ball Smoothing...")
    if progress_callback:
        progress_callback(int(frames_to_process * 0.86), frames_to_process, "Consolidating tracks, classifying teams & smoothing ball...")

    # 2a. Camera-motion-compensated + Lab-appearance global tracklet stitching
    merged_tracks, merged_colors = consolidate_player_tracks(
        raw_player_tracks,
        raw_player_colors,
        frame_warps,
        max_frame_gap=95,
        max_lab_dist=28.0
    )

    # 2b. Drop ephemeral noise specks (< min_frames) and extreme sideline/coach tracks
    min_track_frames = max(3, min(8, frames_to_process // 15))
    valid_tracks: Dict[int, Dict[int, List[float]]] = {}
    valid_colors: Dict[int, List[np.ndarray]] = {}
    track_positions_2d: Dict[int, List[Tuple[float, float]]] = {}

    scale_y_m = pitch_view.PITCH_WIDTH_M / max(1.0, float(pitch_view.field_h))

    for tid, fmap in merged_tracks.items():
        if len(fmap) < min_track_frames:
            continue

        pts_2d: List[Tuple[float, float]] = []
        cam_ys: List[float] = []
        top_dists: List[float] = []
        for f_i, bb in fmap.items():
            fx = (bb[0] + bb[2]) * 0.5
            fy = bb[3]
            cam_ys.append(fy)
            te = frame_top_edges.get(f_i)
            if te is not None and len(te) > 0:
                bin_x = int(np.clip(round(fx / 10.0), 0, len(te) - 1))
                ty = int(te[bin_x])
                if ty > 0:
                    top_dists.append(fy - ty)
            p2d = pitch_view.map_camera_to_2d(
                fx, fy, cam_w=cam_w, cam_h=cam_h, pitch_y_bounds=frame_y_bounds.get(f_i)
            )
            pts_2d.append(p2d)

        avg_ym = float(np.mean([(p[1] - pitch_view.field_y1) * scale_y_m for p in pts_2d]))
        avg_cam_y = float(np.mean(cam_ys))
        std_cam_y = float(np.std(cam_ys))

        # Check vertical distance below the far touchline grass edge (catches far-side linesmen!)
        avg_top_dist = float(np.mean(top_dists)) if top_dists else 999.0
        is_far_linesman = (avg_top_dist <= 22.0 and std_cam_y < 10.5) or (avg_top_dist <= 13.0)

        # Reject coaches / near-side linesmen / far-side linesmen / dugout staff stuck along touchlines
        if avg_ym >= 65.8 or avg_ym <= 1.5 or avg_cam_y >= cam_h - 24 or is_far_linesman:
            continue

        valid_tracks[tid] = fmap
        valid_colors[tid] = merged_colors.get(tid, [])
        track_positions_2d[tid] = pts_2d

    # 2c. Track-level Team, Goalkeeper & Referee Classification
    track_teams, goalkeeper_ids = team_assigner.classify_tracks(
        valid_colors,
        track_positions_2d,
        field_x1=float(pitch_view.field_x1),
        field_w=float(pitch_view.field_w),
        field_y1=float(pitch_view.field_y1),
        field_h=float(pitch_view.field_h),
    )

    # Remove rejected Referees & non-player outliers (team_id == -1)
    player_only_tracks = {tid: fmap for tid, fmap in valid_tracks.items() if track_teams.get(tid, -1) >= 0}
    player_only_teams = {tid: track_teams[tid] for tid in player_only_tracks}
    player_only_gks = {tid for tid in goalkeeper_ids if tid in player_only_tracks}
    ref_rejected = len(valid_tracks) - len(player_only_tracks)
    print(f"👕 Classified {len(player_only_tracks)} player tracks ({len(player_only_gks)} GKs) | Rejected {ref_rejected} referee/non-player tracks.")

    # 2d. FIFA 11-player prior long-gap stitching for any remaining non-overlapping fragments
    stitched_tracks, stitched_teams, stitched_gks = _stitch_same_team_non_overlapping(
        player_only_tracks,
        player_only_teams,
        player_only_gks,
        frame_warps,
        max_outfield_per_team=10,
        max_long_gap=240
    )

    # 2e. Assign clean jersey-style IDs (#1 GK + #2..#11 for Team 0; #13 GK + #12,#14..#22 for Team 1)
    final_tracks, final_teams, final_gks = _reindex_clean_squad_ids(
        stitched_tracks, stitched_teams, stitched_gks
    )

    # 2f. Smooth GMC-compensated gap interpolation for short occlusions / white-board fadeouts (<= 18 frames)
    final_tracks = interpolate_player_gaps(final_tracks, frame_warps, max_gap=18)

    # 2g. Finalize Ball Trajectory (outlier pruning + rescue + GMC-compensated interpolation)
    final_ball_positions = ball_tracker.finalize_trajectory(
        total_frames=frames_to_process,
        frame_warps=frame_warps,
        max_interp_gap=24
    )
    ball_cov = (len(final_ball_positions) / max(1, frames_to_process)) * 100.0
    print(f"⚽ Ball tracked across {len(final_ball_positions)}/{frames_to_process} frames ({ball_cov:.1f}% coverage).")

    # 2h. Populate SpatialAnalytics with clean, consolidated, interpolated tracks
    spatial_analytics.goalkeeper_ids = set(final_gks)
    frame_to_players: Dict[int, List[Tuple[int, List[float], int, bool]]] = defaultdict(list)

    for pid, fmap in final_tracks.items():
        team_id = final_teams[pid]
        is_gk = pid in final_gks
        for f_i in sorted(fmap.keys()):
            bb = fmap[f_i]
            fx = (bb[0] + bb[2]) * 0.5
            fy = bb[3]
            pos_2d = pitch_view.map_camera_to_2d(
                fx, fy, cam_w=cam_w, cam_h=cam_h, pitch_y_bounds=frame_y_bounds.get(f_i)
            )
            spatial_analytics.add_position(pid, bb, team_id, pos_2d=pos_2d, frame_idx=f_i)
            frame_to_players[f_i].append((pid, bb, team_id, is_gk))

    # Store tracked ball coordinates in SpatialAnalytics too
    for f_i in sorted(final_ball_positions.keys()):
        bx, by, _brad, _interp = final_ball_positions[f_i]
        b_2d = pitch_view.map_camera_to_2d(
            bx, by, cam_w=cam_w, cam_h=cam_h, pitch_y_bounds=frame_y_bounds.get(f_i)
        )
        spatial_analytics.ball_track.append((f_i, bx, by))
        spatial_analytics.ball_track_2d.append((f_i, b_2d[0], b_2d[1]))

    # Enforce strict FIFA 11v11 in 2D analytics (synthesizes off-screen GK in 2D only if unseen)
    spatial_analytics.enforce_team_squad_limits(max_players_per_team=11)

    # =========================================================================
    # PASS 2: FAST BROADCAST VIDEO RENDERING (CONSISTENT IDS + WHITE BALL TRAIL)
    # =========================================================================
    print("\n🎥 Pass 2/2: Rendering Annotated Match Video with GMC-Pinned Trails & White Ball Track...")
    output_video_path = os.path.join(output_dir, "analyzed_match.mp4")
    writer = create_video_writer(output_video_path, video_info["fps"], cam_w, cam_h)

    player_trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=player_trail_len))
    ball_trail: deque = deque(maxlen=ball_trail_len)
    last_ball_frame: int = -999

    pbar2 = tqdm(total=frames_to_process, desc="Pass 2 (Render)", unit="fr")

    for f_idx, frame in video.frame_generator():
        if f_idx >= frames_to_process:
            break

        annotated_frame = frame.copy()
        warp = frame_warps.get(f_idx)

        # Warp all existing trails by camera motion H so trails stay pinned to the pitch grass!
        for p_trail in player_trails.values():
            TacticalVisualizer.warp_trail_points(p_trail, warp)
        TacticalVisualizer.warp_trail_points(ball_trail, warp)

        # Update active player trails
        players_now = frame_to_players.get(f_idx, [])
        active_ids: Set[int] = set()
        for pid, bb, team_id, _is_gk in players_now:
            foot = ((bb[0] + bb[2]) * 0.5, bb[3])
            active_ids.add(pid)
            if len(player_trails[pid]) > 0:
                prev_p = player_trails[pid][-1]
                if math.hypot(foot[0] - prev_p[0], foot[1] - prev_p[1]) > 85.0:
                    player_trails[pid].clear()
            player_trails[pid].append(foot)

        # Update WHITE ball trail
        ball_now = final_ball_positions.get(f_idx)
        if ball_now is not None:
            bx, by, brad, is_interp = ball_now
            if f_idx - last_ball_frame > 10:
                ball_trail.clear()
            elif len(ball_trail) > 0:
                prev_b = ball_trail[-1]
                if math.hypot(bx - prev_b[0], by - prev_b[1]) > 190.0:
                    ball_trail.clear()
            ball_trail.append((bx, by))
            last_ball_frame = f_idx
        elif f_idx - last_ball_frame > 8:
            ball_trail.clear()

        # Layer 1: Active player motion trails (team-colored)
        TacticalVisualizer.draw_player_trails(
            annotated_frame,
            player_trails,
            final_teams,
            active_ids=active_ids
        )

        # Layer 2: Pure WHITE ball motion trail (drawn beneath ball ring, above grass)
        TacticalVisualizer.draw_ball_trail(annotated_frame, ball_trail)

        # Layer 3: Tactical foot ellipses & clean stable IDs (#1..#22)
        for pid, bb, team_id, is_gk in players_now:
            tactical_color = TacticalVisualizer.TEAM_COLORS.get(team_id, (200, 200, 200))
            TacticalVisualizer.draw_player_ellipse(
                annotated_frame,
                bb,
                tactical_color,
                tracker_id=pid,
                is_goalkeeper=is_gk
            )

        # Layer 4: Ball white ring + downward tactical pointer
        if ball_now is not None:
            TacticalVisualizer.draw_ball_pointer(
                annotated_frame,
                (ball_now[0], ball_now[1]),
                radius=int(round(ball_now[2])),
                interpolated=ball_now[3]
            )

        writer.write(annotated_frame)
        pbar2.update(1)

        if progress_callback and f_idx % 20 == 0:
            pct_frames = int(frames_to_process * 0.86 + (f_idx + 1) * 0.14)
            progress_callback(
                min(frames_to_process, pct_frames),
                frames_to_process,
                f"Pass 2/2 Rendering Video... Frame {f_idx + 1}/{frames_to_process}"
            )

    pbar2.close()
    writer.release()
    print(f"\n🎥 Analyzed video saved to: {output_video_path}")

    # =========================================================================
    # GENERATE HEATMAPS, 2D FORMATIONS & SPATIAL DATASET JSON
    # =========================================================================
    print("\n🗺️ Generating Heatmaps & Formations (Camera & 2D Top-Down)...")
    MIN_STABLE_FRAMES = 1  # Already noise-filtered & 11v11 enforced above

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

    # Camera Perspective Average Formations
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

    # 2D Top-Down Rectangular Pitch Artifacts
    avg_positions_2d = spatial_analytics.get_average_positions_2d(min_frames=MIN_STABLE_FRAMES)
    pitch_2d_formation_img = pitch_view.draw_2d_formations(avg_positions_2d, spatial_analytics.player_teams)
    pitch_2d_formation_path = os.path.join(output_dir, "pitch_2d_formation.png")
    cv2.imwrite(pitch_2d_formation_path, pitch_2d_formation_img)

    team0_pts_2d = spatial_analytics.get_2d_positions(team_id=0, min_frames=MIN_STABLE_FRAMES)
    pitch_2d_team0_img = pitch_view.generate_2d_heatmap(team0_pts_2d)
    pitch_2d_team0_path = os.path.join(output_dir, "pitch_2d_heatmap_team0.png")
    cv2.imwrite(pitch_2d_team0_path, pitch_2d_team0_img)

    team1_pts_2d = spatial_analytics.get_2d_positions(team_id=1, min_frames=MIN_STABLE_FRAMES)
    pitch_2d_team1_img = pitch_view.generate_2d_heatmap(team1_pts_2d)
    pitch_2d_team1_path = os.path.join(output_dir, "pitch_2d_heatmap_team1.png")
    cv2.imwrite(pitch_2d_team1_path, pitch_2d_team1_img)

    all_pts_2d = spatial_analytics.get_2d_positions(min_frames=MIN_STABLE_FRAMES)
    pitch_2d_all_img = pitch_view.generate_2d_heatmap(all_pts_2d)
    pitch_2d_all_path = os.path.join(output_dir, "pitch_2d_heatmap_all.png")
    cv2.imwrite(pitch_2d_all_path, pitch_2d_all_img)

    # Save spatial dataset to JSON for instant dashboard reloading
    spatial_data_path = os.path.join(output_dir, "spatial_data.json")
    spatial_analytics.save_to_json(spatial_data_path)

    print(f"📊 2D Rectangular Heatmaps & Formations saved to {output_dir}")
    print(f"✅ Total unique players tracked: {len(avg_positions)} (11 vs 11 FIFA Standard)")
    print("=" * 64)
    print("🎉 Pipeline Completed Successfully!")
    print("=" * 64)

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
    parser.add_argument("--conf", type=float, default=0.14, help="Player detection confidence")
    parser.add_argument("--ball-conf", type=float, default=0.06, help="Ball detection confidence")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference resolution")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.frames,
        model_name=args.model,
        confidence=args.conf,
        ball_confidence=args.ball_conf,
        imgsz=args.imgsz
    )
