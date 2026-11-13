"""
Video reading and writing utility functions.
"""

import os
from typing import Generator, Tuple
import cv2


class VideoHandler:
    """Helper class to stream frames and save processed videos."""

    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_info(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_sec": self.total_frames / self.fps if self.fps > 0 else 0
        }

    def frame_generator(self) -> Generator[Tuple[int, cv2.typing.MatLike], None, None]:
        """Yields frame index and frame matrix."""
        frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()


def create_video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int
) -> cv2.VideoWriter:
    """Creates an OpenCV VideoWriter using mp4v or avc1 codec."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
