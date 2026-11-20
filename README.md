# ⚽ Footballistic: AI-Powered Football Match Analytics with Computer Vision

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8%20%2F%20v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![BoT-SORT](https://img.shields.io/badge/Tracker-GMC%20BoT--SORT-emerald.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Footballistic** is an open-source football (soccer) analytics engine powered by Deep Learning and Computer Vision. It automatically extracts persistent player tracking, camera-motion-compensated ball trajectories, team & goalkeeper identification, 2D tactical bird's-eye formations, heatmaps, and match statistics from single tactical/wide-angle camera feeds.

---

## 📌 Project Overview

Traditional match analytics (e.g., Opta, StatsBomb) rely heavily on human annotators clicking timestamps and pitch coordinates, or expensive multi-camera optical installations inside elite stadiums.

**Footballistic** aims to democratize professional-grade football intelligence by combining high-resolution object detection, camera-motion-compensated multi-object tracking, and spatial analytics on standard wide-angle match video.

---

## 🚀 Roadmap & Core Modules

### 📍 Phase 1: Core Vision Pipeline & Spatial Analytics ✅ (Completed)
- [x] **Repository & Modular Architecture:** Clean separation of detection, tracking, team classification, homography, spatial analytics, and dashboard UI
- [x] **High-Resolution Pitch-Cropped Detection:** `YOLO11m` / `YOLOv8m` at `1280px` cropped dynamically to the playing field (`PitchDetector` with non-convex turf segmentation, green LED ad-board immunity, and white near-touchline RANSAC detection to exclude spectators, dugouts, coaches, and linesmen)
- [x] **Camera-Motion-Compensated Tracking (`GMC BoT-SORT` + Lab Re-ID):** `sparseOptFlow` affine camera-pan compensation, Stage-3 foot-distance + CIE Lab jersey re-association for small distant players, global GMC tracklet consolidation, FIFA 11-player prior re-stitching, and smooth occlusion-gap interpolation (`#1`–`#22` stable IDs)
- [x] **Dedicated Football Ball Tracker & White Comet Trail:** GMC-Kalman filter, 1:1 native-resolution `320px` ROI zoom recovery, local motion/white-blob recovery, unattended static penalty-spot rejection, and GMC-pinned pure white trajectory rendering
- [x] **Team, Goalkeeper & Referee Classification:** Grass-excluded torso color extraction, track-level iterative trimmed K-Means (`Lab`), referee outlier rejection, and defensive-extremity Goalkeeper identification (`#1 GK`, `#13 GK`)
- [x] **2D Bird's-Eye Pitch Mapping & Spatial Analytics:** Projective perspective foreshortening onto a standard $105\text{m} \times 68\text{m}$ pitch, Gaussian KDE heatmaps, convex-hull tactical shapes, team compactness/centroids, and strict 11v11 formations
- [x] **2-Pass Broadcast Video Renderer:** Globally consistent team colors, stable player IDs, GMC-pinned player & white ball trails encoded to web-ready H.264

### 📍 Phase 2: Event Detection & Individual Pass Intelligence ✅ (Completed)
- [x] **Ball Possession Engine:** Frame-by-frame dynamic player & team ball ownership, possession percentage (`%`), continuous possession timeline, and team control spells
- [x] **Physics-Verified Individual Pass Breakdown:** Kinematic aerial fly-over rejection + exact kick-to-reception timing for both **Completed** and **Incomplete / Intercepted** passes — outputting **From → To player IDs**, pass type (`Long Switch`, `Box Entry`, `Progressive`, `Short`, `Intercepted`), perspective-calibrated **length (m)** on a $105\text{m} \times 68\text{m}$ FIFA pitch, **average & peak ball speed (km/h)**, and **1 dedicated 2D freeze-frame visual per pass** showing the actual ball trajectory and all 22 players
- [x] **Turnovers, Interceptions & Recoveries:** Opponent possession transitions, ball-winner attribution, and high-turnover zone mapping
- [x] **Shots, Crosses & Final-Third / Box Entries:** High-velocity goal-mouth ball vector spotting, flank crosses, and penalty-area entries

### 📍 Phase 3: Advanced Physical & Tactical Metrics
- [x] **Core Physical Metrics:** Estimated distance covered (`m`), average speed (`km/h`), pitch presence (`%`), and tactical third/channel distribution per player
- [ ] **Speed Zones & Sprint Analytics:** Instantaneous speed smoothing, top speed (`km/h`), high-speed running (`> 18 km/h`), and sprint counts (`> 21 km/h`)
- [ ] **Tactical Shape Dynamics:** Defensive line height, midfield-to-defense gap, team compactness over time, and pressing intensity (nearest-defender pressure distance)

### 📍 Phase 4: Interactive Tactical Match Center
- [x] **Minimalist Streamlit Studio (`app.py`):** Instant pre-computed dual-match loader, model switcher (`YOLO11` / `YOLOv8`), custom video upload, Team Analytics, Player Deep-Dive, H.264 annotated replay, and CSV export
- [x] **Pass-by-Pass Tactical Inspector Tab:** Interactive single-pass 2D freeze-frame viewer (paired with real broadcast mid-pass snapshots), dual-team possession split bar, complete All-Passes telemetry table (`From → To`, `Outcome`, `Length m`, `Speed km/h`), and CSV export
- [ ] **Automated Match Report Export:** One-click printable HTML / PDF scouting report summarizing formations, heatmaps, passing networks, and KPIs

### 📍 Phase 5: Pro-Grade Calibration (Future Stretch)
- [ ] **Automatic Pitch Keypoint Homography:** Per-frame line/intersection keypoint detection for exact metric projective homography across any camera pan/zoom
- [ ] **Jersey Number Recognition (OCR):** Optional cropped back-number recognition to link track IDs to real matchday squad numbers

---

## 📁 Repository Structure

```text
Footballistic/
├── configs/
│   ├── football_botsort.yaml       # GMC BoT-SORT config (sparseOptFlow + 120f buffer)
│   └── football_bytetrack.yaml     # Legacy ByteTrack config
├── data/
│   └── sample_videos/              # Tactical wide-angle & broadcast sample clips
├── weights/                        # YOLO11m / YOLOv8m model checkpoints
├── src/
│   ├── detection/
│   │   ├── detector.py             # High-res pitch-cropped YOLO + 1:1 ROI ball zoom
│   │   └── pitch_detector.py       # Non-convex turf mask & RANSAC near-touchline detector
│   ├── tracking/
│   │   ├── football_tracker.py     # GMC BoT-SORT + Stage-3 Lab Re-ID + global stitcher
│   │   └── ball_tracker.py         # GMC-Kalman + ROI zoom + static-spot rejection
│   ├── team_assignment/
│   │   └── color_assigner.py       # Grass-excluded Lab K-Means, GK & referee classifier
│   ├── homography/
│   │   └── pitch_view.py           # Projective 2D top-down pitch, heatmaps & formations
│   ├── analysis/
│   │   ├── heatmap.py              # Spatial analytics, 11v11 squad enforcement & JSON I/O
│   │   └── events.py               # Pass telemetry (length/speed), possession & event engine
│   └── utils/
│       ├── drawing.py              # Broadcast ellipses, GMC-pinned trails & white ball trail
│       └── video_utils.py          # Video streaming & PyAV H.264 encoding
├── output/                         # Pre-computed Match 1 analytics & video
├── output_sample2/                 # Pre-computed Match 2 analytics & video
├── main.py                         # 2-pass computer vision pipeline entrypoint
├── app.py                          # Streamlit interactive tactical match center
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/omerfarukyuce/Footballistic.git
cd Footballistic
```

### 2. Set up virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Pipeline & Interactive Web Dashboard

### Run the 2-pass CV pipeline on a video
```bash
python main.py --video data/sample_videos/sample2.mp4 --output output_sample2 --frames 0
```

### Launch the interactive Streamlit Match Center
```bash
streamlit run app.py
```

From the dashboard, you can:
- **Switch instantly** between pre-computed sample matches or **upload** your own match video.
- **Select** between `YOLO11m` and `YOLOv8m` detection models.
- **Inspect Team Analytics:** 11v11 2D formations, team centroids, compactness, channel preference, and collective heatmaps.
- **Deep-Dive into Individual Players:** individual 2D trajectories, heatmaps, estimated distance, speed, and tactical zone.
- **Watch the Annotated Replay** with stable player IDs, GMC-pinned movement trails, and a pure white ball trajectory trail, and **export** spatial data to CSV.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/omerfarukyuce/Footballistic/issues).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
