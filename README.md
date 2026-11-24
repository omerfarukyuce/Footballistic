<div align="center">

# Footballistic

### Broadcast-grade football analytics from a single camera feed

Detect and track every player and the ball, classify teams and goalkeepers, project play onto a 2D FIFA pitch,<br>
and extract calibrated pass telemetry, speed zones, sprints, structural compactness, and pressing intensity.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/Detection-YOLO11%20%2F%20YOLOv8-0F172A?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracking-GMC%20BoT--SORT-10B981?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-334155?style=flat-square)](LICENSE)

<br>

<img src="docs/assets/demo_sample_12s.gif" alt="Footballistic demo: 12 seconds of tracked broadcast footage" width="880">

<sub>First 12 seconds of an analysed Bundesliga broadcast clip — stable player IDs, team colours, camera-compensated movement trails, and a white ball trail.
<a href="docs/assets/demo_sample_12s.mp4">Download full-quality MP4 &rarr;</a></sub>

</div>

---

## Overview

Elite match analytics (Opta, StatsBomb, Second Spectrum) relies on manual event annotation or dedicated multi-camera optical arrays inside stadiums.
**Footballistic** explores how much of that tactical and physical intelligence can be recovered automatically from a single broadcast camera feed.

| Capability | Details |
|---|---|
| **Full-pitch player tracking** | Detects both near and far-touchline players and preserves IDs across fast camera pans |
| **Dedicated ball tracking** | Camera-compensated Kalman filter + native-resolution ROI zoom (~95% frame lock on the demo clip) |
| **Team & role classification** | Grass-subtracted jersey clustering separates Home, Away, Goalkeepers, referees, and touchline staff |
| **Optical-Flow GMC + FIFA scale** | Per-frame optical-flow camera warps combined with automatic standing-height perspective scaling (~1.7% error vs. FIFA center circle) |
| **Pass-by-pass telemetry** | Exact kick-to-reception length (m), average & peak ball speed (km/h), flight time, and outcome with per-pass freeze-frames |
| **Physical & tactical dynamics** | FIFA 5-zone speed breakdown, sprint bursts (`≥ 21 km/h`), defensive line height, block compactness, and ball pressing intensity |
| **Obsidian Match Centre** | Minimalist dark Streamlit studio with bespoke HTML data tables, interactive pass stepping (`←`/`→`), and CSV export |

---

## Dashboard Tour

### 01 &middot; Pass-by-Pass Tactical Inspector
Every detected pass receives a dedicated 2D freeze-frame showing all 22 players at the exact kick frame, the real curved ball trajectory, passer/receiver halos, calibrated length and speed, and a synchronized broadcast snapshot for direct visual verification. Step chronologically through passes with the `&larr;` / `&rarr;` controls.

<p align="center">
  <img src="docs/assets/dashboard_pass_completed.png" alt="Completed 25.9m long pass (#2 to #3) in the tactical inspector" width="49%">
  <img src="docs/assets/dashboard_pass_intercepted.png" alt="Intercepted 14.9m pass (#7 to #20) in the tactical inspector" width="49%">
</p>
<p align="center"><sub><b>Left:</b> completed 25.9 m pass (#2 &rarr; #3, 42 km/h). <b>Right:</b> pass by #7 intercepted by opponent #20 (14.9 m, 71 km/h, dashed red path + &times; marker).</sub></p>

### 02 &middot; All-Passes Match Overview & Telemetry Log
All passes rendered onto one bird's-eye pitch — numbered circles mark kick origins matching the telemetry log (`green` = completed, `dashed red` = intercepted) — paired with the full Obsidian pass telemetry table (**From &rarr; To**, **Outcome**, **Length**, **Avg / Peak Speed**, **Duration**, and **Target Zone**), exportable as CSV.

<p align="center">
  <img src="docs/assets/dashboard_pass_overview.png" alt="All passes match overview map" width="84%">
</p>
<p align="center">
  <img src="docs/assets/dashboard_pass_log.png" alt="Complete pass telemetry log table" width="88%">
</p>

### 03 &middot; Team Tactical Analytics, Structural Dynamics & Pressing
11v11 average formations, convex-hull team shape, collision-free centroid markers, collective occupancy heatmaps, channel / pitch-depth distribution bars, and time-series structural compactness (defensive, midfield, and attacking lines) alongside nearest-opponent ball pressing intensity (`≤ 4.5 m` high-press zone).

<p align="center">
  <img src="docs/assets/dashboard_team_home.png" alt="Team 0 Home tactical analytics" width="49%">
  <img src="docs/assets/dashboard_team_away.png" alt="Team 1 Away tactical analytics" width="49%">
</p>
<p align="center">
  <img src="docs/assets/dashboard_team_home_dynamics.png" alt="Team 0 line heights, block envelope and pressing intensity" width="49%">
  <img src="docs/assets/dashboard_team_comparative_dynamics.png" alt="Comparative 11v11 compactness and both-teams pressing intensity" width="49%">
</p>
<p align="center"><sub><b>Top:</b> Home &amp; Away formations, centroids, and heatmaps. <b>Bottom:</b> single-team block envelope (left) and comparative 11v11 compactness &amp; pressing intensity (right).</sub></p>

### 04 &middot; Player Deep-Dive, FIFA Speed Zones & Physical Leaderboard
Inspect any tracked player's 2D trajectory, individual KDE heatmap, calibrated distance, average/top speed, peak acceleration, compact velocity profile chart paired with the **FIFA 5-Zone Intensity Breakdown** (Walking through Sprinting + detected sprint burst timestamps), and the sortable **Match Physical Leaderboard**.

<p align="center">
  <img src="docs/assets/dashboard_player_home.png" alt="Player #6 deep-dive for Team 0" width="49%">
  <img src="docs/assets/dashboard_player_away.png" alt="Player #12 deep-dive for Team 1" width="49%">
</p>
<p align="center">
  <img src="docs/assets/dashboard_player_speed_home.png" alt="Player #6 instantaneous speed profile and FIFA 5-zone breakdown with sprint burst" width="49%">
  <img src="docs/assets/dashboard_player_speed_away.png" alt="Player #12 instantaneous speed profile and FIFA 5-zone breakdown" width="49%">
</p>
<p align="center">
  <img src="docs/assets/dashboard_leaderboard_home.png" alt="Match physical leaderboard for Team 0 Home" width="49%">
  <img src="docs/assets/dashboard_leaderboard_away.png" alt="Match physical leaderboard for Team 1 Away" width="49%">
</p>
<p align="center"><sub><b>Top:</b> trajectories &amp; heatmaps (#6 Home, #12 Away). <b>Middle:</b> instantaneous speed profiles &amp; FIFA 5-zone sprint breakdown. <b>Bottom:</b> Match Physical Leaderboards with active-player highlight &amp; inline intensity bars.</sub></p>

---

## Pipeline Architecture

```mermaid
flowchart LR
    A["Broadcast Video"] --> B["Pitch Segmentation<br/>Turf Mask & Touchline"]
    B --> C["High-Res YOLO<br/>Players & Ball (1280px)"]
    C --> D["GMC BoT-SORT<br/>+ CIE Lab Re-ID"]
    C --> E["Ball Tracker<br/>Kalman + ROI Zoom"]
    D --> F["Team / GK / Ref<br/>Colour Classifier"]
    F --> G["Track Consolidation<br/>11v11 Prior & IDs"]
    E --> H["Optical-Flow GMC<br/>& FIFA Perspective Scale"]
    G --> H
    H --> I["Pass Engine & Phase 3<br/>Speed Zones · Compactness · Press"]
    I --> J["Obsidian Match Centre<br/>+ H.264 Annotated Replay"]
```

The engine runs in two passes: **Pass 1** performs detection, camera-compensated tracking, global tracklet stitching, optical-flow warp extraction, and team classification (cached to `pass1_cache.pkl`); **Pass 2** renders the globally consistent H.264 broadcast replay and exports `spatial_data.json` (with per-frame camera warps and FIFA height calibration) plus per-pass 2D freeze-frames for instant dashboard loading.

---

## Engineering Challenges

Broadcast football presents tough failure modes for off-the-shelf vision pipelines. Key issues addressed in this codebase:

| Challenge | Solution |
|---|---|
| **Distant far-touchline players missed by default detectors** | Inference runs on the dynamically pitch-cropped region at `imgsz=1280` with a low confidence floor (`0.14`) and second-stage recovery |
| **Spectators, LED ad boards, dugouts, and coaches flagged as players** | Non-convex turf HSV segmentation (excluding emerald LED panels) combined with a RANSAC white near-touchline boundary filter |
| **Fast camera pans cause tracker ID switches & fake player speed** | BoT-SORT with `sparseOptFlow` global motion compensation (GMC), Stage-3 CIE Lab Re-ID, 11v11 consolidation, and per-frame optical-flow warp subtraction in kinematics |
| **Unknown camera zoom inflates distances when assuming full 105 m FOV** | Automatic standing-player height regression (`h(y) / 2.22 m`) calibrates local horizontal `px/m` at every depth row, matching the FIFA 18.30 m center circle within **1.7%** |
| **Tiny, fast, frequently occluded football & static penalty-spot false positives** | Dedicated `BallTracker` using a GMC-Kalman filter, 1:1 native-resolution `320px` ROI zoom, motion/white-blob recovery, and unattended static penalty-spot rejection |
| **White kits blending into ad boards / neon-green kits blending into turf** | Local-turf-adaptive colour subtraction inside jersey crops plus lopsided-cluster protection in K-Means |
| **Long aerial switches falsely broken by bystanders underneath** | Kinematic fly-over filter verifying minimum in-spell deceleration and trajectory bend before, during, and after contact |
| **Halo-to-halo pass timing inflating ball speeds** | Exact kick-frame and reception-frame refinement pinpoints true departure and arrival timestamps for realistic speed (km/h) |

---

## Evaluation on Demo Clips

| Metric | `sample.mp4` (Broadcast) | `sample2.mp4` (Wide Tactical) |
|---|:---:|:---:|
| Auto-calibrated horizontal FOV | **69.5 m** *(FIFA ground truth: 70.7 m)* | **89.0 m** |
| Players tracked | **22** (11 v 11) | **22** (11 v 11) |
| Ball frame coverage | **94.9%** (712 / 750 frames) | ~85–91% |
| Passes detected | **9** (8 completed &middot; 1 intercepted) | **5** (4 completed &middot; 1 intercepted) |
| Pass accuracy | 88.9% | 80.0% |
| Mean pass length / speed | 21.6 m &middot; 46.8 km/h | 30.4 m &middot; 40.5 km/h |
| Longest pass | **50.9 m** cross-field switch (`#8` &rarr; `#7`) | **71.9 m** long clearance (`#2` &times; `#12`) |
| Fastest player (top speed) | **#6** &middot; 28.7 km/h *(26.5 m HSR)* | **#16** &middot; 26.8 km/h *(27.4 m HSR)* |

> Metrics reflect two 30-second validation clips used during development rather than a large-scale benchmark.

### Metric Calibration (Distance & Speed)
1. **Optical-flow camera subtraction:** Each frame's 2&times;3 background affine warp (`frame_warps`) removes pure camera pan, tilt, and zoom before measuring player or ball displacement.
2. **Perspective Jacobian:** Horizontal `px/m` is calibrated at every image row `y` from the trimmed linear height profile of standing players (`h(y) / 2.22 m`), while pitch width (`68.0 m`) uses a projective foreshortening curve (`persp_k = 1.75`) anchored to the on-pitch touchline bounds.
3. **Pass & physical telemetry:** Pass length and speed integrate calibrated ball displacement from `kick_frame` to `receive_frame`; player trajectories use centered ~0.5 s smoothing + 1.2 km/h noise-floor gating for FIFA 5-zone and sprint burst extraction.

---

## Quick Start

```bash
git clone https://github.com/omerfarukyuce/Footballistic.git
cd Footballistic

python -m venv venv
venv\Scripts\activate          # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

**1. Prepare footage and weights**  
Place a wide-angle or broadcast `.mp4` clip in `data/sample_videos/` (or download the demo clip with `python scripts/download_sample.py`) and ensure a YOLO checkpoint such as `yolo11m.pt` is available under `weights/`.

**2. Run the two-pass analytics pipeline**
```bash
python main.py --video data/sample_videos/sample.mp4 --output output
```

**3. Launch the Streamlit Match Centre**
```bash
streamlit run app.py
```

> Processed outputs (`output*/`), sample videos, and model weights are git-ignored; run step 2 once locally before opening the dashboard.

---

## Roadmap

### Phase 1 &mdash; Core Vision Pipeline &nbsp;`Completed`
- [x] High-resolution pitch-cropped YOLO detection with turf, LED-board, and touchline filtering
- [x] Camera-motion-compensated BoT-SORT with Lab Re-ID and 11v11 track consolidation
- [x] Dedicated ball tracker with ROI zoom, penalty-spot rejection, and white trajectory trail
- [x] Team, goalkeeper, and referee classification
- [x] 2D bird's-eye pitch mapping, formations, heatmaps, and atomic H.264 replay rendering

### Phase 2 &mdash; Event Detection & Pass Intelligence &nbsp;`Completed`
- [x] Dynamic frame-level player and team ball possession engine
- [x] Physics-verified pass detection (completed and intercepted) with calibrated length, speed, and per-pass 2D freeze-frame visuals
- [x] Interceptions, ball recoveries, progressive passes, final-third entries, and box entries

### Phase 3 &mdash; Physical & Tactical Metrics &nbsp;`Completed`
- [x] Optical-flow GMC & FIFA standing-height calibrated distance, average speed, pitch presence, and third/channel splits
- [x] FIFA 5-zone speed breakdown (Walking through Sprinting), peak speed, peak acceleration, HSR (`> 18 km/h`), sprint counts (`≥ 21 km/h`), and instantaneous velocity profile charts
- [x] Time-series defensive line height, midfield-to-defense gap, outfield block compactness envelope, and ball-carrier pressing intensity (`≤ 4.5 m` high-press zone)

### Phase 4 &mdash; Interactive Match Centre &nbsp;`In Progress`
- [x] Minimalist Obsidian dark Streamlit studio with pass inspector (`←`/`→`), bespoke HTML telemetry tables, match overview map, and CSV export
- [ ] One-click printable HTML / PDF match scouting report

### Phase 5 &mdash; Pro-Grade Calibration &nbsp;`Planned`
- [ ] Automatic pitch-keypoint homography for exact metric projection across arbitrary camera zoom/pan
- [ ] Jersey-number OCR linking tracker IDs to matchday squad numbers

---

## Current Limitations

- **Perspective approximation:** While optical-flow GMC + standing-height scaling matches the center circle to within ~2% on broadcast feeds, full per-frame line-keypoint homography (Phase 5) is still needed for extreme camera angles or heavy zoom changes.
- **2D ball tracking:** Aerial elevation is not modelled; lofted passes are projected onto the ground plane.
- **Anonymous IDs:** Player labels (`#1`–`#22`) are persistent per-clip identifiers rather than recognized shirt numbers.
- **Validation scope:** Tested thoroughly on two 30-second broadcast/tactical clips; broader multi-stadium evaluation is planned.
- **Hardware:** Runs on CPU out of the box; a CUDA GPU significantly accelerates Pass 1 inference.

---

## Repository Structure

```text
Footballistic/
├── .streamlit/config.toml            # Obsidian dark studio theme configuration
├── configs/football_botsort.yaml     # GMC BoT-SORT tracker settings
├── docs/assets/                      # 2x Retina dashboard screenshots & 12s demo clip
├── scripts/download_sample.py        # Sample match clip downloader
├── src/
│   ├── detection/                    # High-res YOLO · pitch & touchline segmentation
│   ├── tracking/                     # GMC BoT-SORT · ball tracker · global stitcher
│   ├── team_assignment/              # Lab K-Means team, GK & referee classifier
│   ├── homography/pitch_view.py      # 2D pitch, heatmaps, single-pass & overview visuals
│   ├── analysis/                     # Spatial metrics · pass engine · speed zones & pressing
│   └── utils/                        # Broadcast visualizer · PyAV H.264 encoder
├── main.py                           # Two-pass CV pipeline entrypoint
├── app.py                            # Streamlit Obsidian Match Centre UI
└── requirements.txt
```

---

## Contributing

Contributions, issues, and feature requests are welcome via the [issues page](https://github.com/omerfarukyuce/Footballistic/issues).

## License

Released under the [MIT License](LICENSE).  
Sample clips are short broadcast excerpts (DFL Bundesliga / Roboflow football dataset) included for demonstration purposes only.
