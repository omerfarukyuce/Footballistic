# ⚽ Footballistic: AI-Powered Football Match Analytics with Computer Vision

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%20%2F%20v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Supervision](https://img.shields.io/badge/Supervision-Roboflow-purple.svg)](https://github.com/roboflow/supervision)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Footballistic** is an open-source football (soccer) analytics engine powered by Deep Learning and Computer Vision. It automatically extracts player tracking, team classification, tactical pitch radar mappings, heatmaps, and comprehensive match statistics from tactical/wide-angle match footage.

---

## 📌 Project Overview

Traditional match analytics (e.g., Opta, StatsBomb) rely heavily on human annotators clicking timestamps and pitch coordinates, or expensive multi-camera optical setups installed in elite stadiums. 

**Footballistic** aims to democratize professional-grade football analytics by leveraging state-of-the-art vision models on single tactical/wide-angle camera feeds.

---

## 🚀 Roadmap & Core Modules

### 📍 Phase 1: Core Foundation & Spatial Analytics (Current Focus)
- [x] Repository initialization & architectural structure
- [ ] **Object Detection:** Detect players, referees, goalkeepers, and the ball (`YOLOv8` / `YOLOv11`)
- [ ] **Multi-Object Tracking:** Assign persistent IDs across video frames (`ByteTrack`)
- [ ] **Team Identification:** Unsupervised jersey color clustering (`K-Means`) to classify teams & referees
- [ ] **Perspective Transform (Homography):** Map camera coordinates onto a 2D top-down standard pitch model ($105m \times 68m$)
- [ ] **Heatmaps:** Generate player and team spatial density distributions (Gaussian KDE)
- [ ] **Average Formations:** Calculate dynamic average player positions and tactical shapes

### 📍 Phase 2: Event & Match Action Spotting
- [ ] **Ball Possession:** Proximity-based dynamic ball ownership detection
- [ ] **Passing Network:** Completed passes, incomplete passes, and passing lanes
- [ ] **Shots & Crosses:** Ball trajectory velocity and goal-mouth vector analysis

### 📍 Phase 3: Physical & Tactical Metrics
- [ ] **Physical Metrics:** Distance covered (km), sprint counts, and instantaneous speed
- [ ] **Duels & Interceptions:** Ground duels, aerial duels, and recoveries

### 📍 Phase 4: Interactive Dashboard
- [ ] **Streamlit Match Center:** Visual dashboards, interactive replays, and PDF/HTML match reports

---

## 📁 Repository Structure

```text
Footballistic/
├── data/
│   └── sample_videos/        # Tactical camera footage and test clips
├── weights/                  # Pre-trained and fine-tuned model checkpoints
├── src/
│   ├── detection/            # Player, referee, and ball detection modules
│   ├── tracking/             # Multi-object tracking (ByteTrack / BoT-SORT)
│   ├── team_assignment/      # K-Means jersey color clustering
│   ├── homography/           # 2D pitch keypoint mapping and radar view
│   ├── analysis/             # Heatmaps, average positions, and event metrics
│   └── utils/                # Video I/O, geometry, and drawing utilities
├── output/                   # Processed videos, radar animations, and heatmaps
├── main.py                   # Main pipeline video processor
├── app.py                    # Streamlit interactive analytics dashboard
├── requirements.txt
├── .gitignore
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

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/omerfarukyuce/Footballistic/issues).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
