"""
Object detection and tracking module for football players, referees, and the ball using YOLO.
"""

import os
# Ensure Ultralytics config is local before importing YOLO
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath(".ultralytics")

from dataclasses import dataclass
from typing import List, Optional
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
    """Detects and tracks players and the ball across video frames."""

    def __init__(
        self,
        model_path: str = "weights/yolo11m.pt",
        confidence: float = 0.35,
        iou: float = 0.5,
        device: Optional[str] = None,
        tracker_config: str = "configs/football_bytetrack.yaml"
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.tracker_config = tracker_config if os.path.exists(tracker_config) else "bytetrack.yaml"

    def detect_and_track(self, frame: np.ndarray, persist: bool = True) -> FrameDetection:
        """
        Runs YOLO with native ByteTrack tracking.
        Separates players (class 0) with persistent tracking IDs and the ball (class 32).
        """
        results = self.model.track(
            frame,
            persist=persist,
            conf=self.confidence,
            iou=self.iou,
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

                if cls_id == 0:  # person / player
                    players.append(TrackedPlayer(tracker_id=t_id, bbox=bbox, confidence=conf))
                elif cls_id == 32:  # sports ball
                    balls.append(DetectedBall(bbox=bbox, confidence=conf))

        return FrameDetection(players=players, balls=balls)
