"""
High-recall object detection module for football players, goalkeepers, and the ball using YOLO.

Key upgrades for wide-angle & broadcast stadium footage:
  - Pitch-cropped high-resolution inference (imgsz=1280 by default) so YOLO dedicates
    100% of its spatial resolution to the playing surface instead of stadium crowds,
    dramatically improving recall on distant far-touchline players and tiny footballs.
  - Separate confidence thresholds for players (0.14) and the football (0.06), because
    COCO class 32 (sports ball) typically scores 0.08-0.30 on small moving balls.
  - Native 1:1 resolution ROI zoom (`detect_ball_in_roi`) for recovering small/blurred
    footballs around a Kalman-predicted location at minimal CPU cost.
"""

import os
# Ensure Ultralytics config is local before importing YOLO
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath(".ultralytics")

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackedPlayer:
    tracker_id: Optional[int]
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float


@dataclass
class DetectedBall:
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float


@dataclass
class FrameDetection:
    players: List[TrackedPlayer]
    balls: List[DetectedBall]


class FootballDetector:
    """Detects players (class 0) and the football (class 32) at high spatial resolution."""

    def __init__(
        self,
        model_path: str = "weights/yolo11m.pt",
        confidence: float = 0.14,
        ball_confidence: float = 0.06,
        imgsz: int = 1280,
        iou: float = 0.55,
        device: Optional[str] = None,
        tracker_config: str = "configs/football_botsort.yaml"
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.ball_confidence = min(ball_confidence, confidence)
        self.imgsz = imgsz
        self.iou = iou
        self.device = device
        self.tracker_config = tracker_config if os.path.exists(tracker_config) else "botsort.yaml"

    def detect_frame(
        self,
        frame: np.ndarray,
        crop_bounds: Optional[Tuple[int, int, int, int]] = None
    ) -> FrameDetection:
        """
        Runs high-resolution YOLO detection on the pitch crop (or full frame).
        Separates players (class 0) and football candidates (class 32) with
        independent confidence thresholds and remaps coordinates back to full frame.
        """
        ox, oy = 0, 0
        if crop_bounds is not None:
            x1, y1, x2, y2 = crop_bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(x1 + 1, min(w, int(x2)))
            y2 = max(y1 + 1, min(h, int(y2)))
            roi = frame[y1:y2, x1:x2]
            ox, oy = x1, y1
        else:
            roi = frame

        base_conf = min(self.confidence, self.ball_confidence)
        results = self.model.predict(
            roi,
            imgsz=self.imgsz,
            conf=base_conf,
            iou=self.iou,
            classes=[0, 32],
            device=self.device,
            verbose=False
        )[0]

        players: List[TrackedPlayer] = []
        balls: List[DetectedBall] = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            cls_ids = boxes.cls.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for i in range(len(boxes)):
                cls_id = int(cls_ids[i])
                conf = float(confs[i])
                bx1 = float(xyxy[i][0]) + ox
                by1 = float(xyxy[i][1]) + oy
                bx2 = float(xyxy[i][2]) + ox
                by2 = float(xyxy[i][3]) + oy
                bbox = [bx1, by1, bx2, by2]

                if cls_id == 0 and conf >= self.confidence:
                    players.append(TrackedPlayer(tracker_id=None, bbox=bbox, confidence=conf))
                elif cls_id == 32 and conf >= self.ball_confidence:
                    balls.append(DetectedBall(bbox=bbox, confidence=conf))

        return FrameDetection(players=players, balls=balls)

    def detect_ball_in_roi(
        self,
        frame: np.ndarray,
        center_x: float,
        center_y: float,
        roi_size: int = 320,
        conf: float = 0.05
    ) -> List[DetectedBall]:
        """
        Native 1:1 pixel-resolution ball search inside a compact crop around
        a predicted ball coordinate. Gives ~3x higher pixel detail on tiny or
        fast-moving footballs than full-frame detection at minimal compute cost.
        """
        fh, fw = frame.shape[:2]
        half = roi_size // 2
        x1 = max(0, int(round(center_x)) - half)
        y1 = max(0, int(round(center_y)) - half)
        x2 = min(fw, x1 + roi_size)
        y2 = min(fh, y1 + roi_size)
        # Keep square crop when near right/bottom frame border
        x1 = max(0, x2 - roi_size)
        y1 = max(0, y2 - roi_size)

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0 or patch.shape[0] < 32 or patch.shape[1] < 32:
            return []

        results = self.model.predict(
            patch,
            imgsz=roi_size,
            conf=conf,
            iou=0.45,
            classes=[32],
            device=self.device,
            verbose=False
        )[0]

        balls: List[DetectedBall] = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                bbox = [
                    float(xyxy[i][0]) + x1,
                    float(xyxy[i][1]) + y1,
                    float(xyxy[i][2]) + x1,
                    float(xyxy[i][3]) + y1,
                ]
                balls.append(DetectedBall(bbox=bbox, confidence=float(confs[i])))

        return balls

    def detect_and_track(self, frame: np.ndarray, persist: bool = True) -> FrameDetection:
        """
        Backwards-compatible convenience wrapper using Ultralytics built-in tracking.
        Note: The main Footballistic pipeline uses `detect_frame()` + `FootballBoTSORT`
        so stadium spectators are filtered OUT before tracking.
        """
        results = self.model.track(
            frame,
            persist=persist,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            classes=[0, 32],
            tracker=self.tracker_config,
            device=self.device,
            verbose=False
        )[0]

        players: List[TrackedPlayer] = []
        balls: List[DetectedBall] = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            cls_ids = boxes.cls.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)

            for i in range(len(boxes)):
                cls_id = int(cls_ids[i])
                bbox = xyxy[i].tolist()
                conf = float(confs[i])
                t_id = int(ids[i]) if ids[i] is not None else None

                if cls_id == 0:
                    players.append(TrackedPlayer(tracker_id=t_id, bbox=bbox, confidence=conf))
                elif cls_id == 32:
                    balls.append(DetectedBall(bbox=bbox, confidence=conf))

        return FrameDetection(players=players, balls=balls)
