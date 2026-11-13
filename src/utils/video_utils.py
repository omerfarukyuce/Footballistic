"""
Video reading and writing utility functions with PyAV H.264 encoding support.
"""

import os
from typing import Generator, Tuple
import cv2
import numpy as np

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False


class VideoHandler:
    """Helper class to stream frames and retrieve video metadata."""

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


class PyAVVideoWriter:
    """High-compatibility H.264 Video Writer using PyAV (FFmpeg bindings)."""

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = output_path
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream("h264", rate=int(round(fps)))
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"preset": "fast", "crf": "23"}

    def write(self, frame_bgr: np.ndarray):
        frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def release(self):
        if self.container:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
            self.container = None


class OpenCVVideoWriterWrapper:
    """Fallback OpenCV VideoWriter wrapper with release method."""

    def __init__(self, writer: cv2.VideoWriter):
        self.writer = writer

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()
            self.writer = None


def create_video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int
):
    """
    Creates an H.264 VideoWriter.
    Uses PyAV for perfect cross-platform HTML5 & Windows Media Player support.
    Falls back to cv2.VideoWriter if PyAV is not installed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if HAS_AV:
        try:
            return PyAVVideoWriter(output_path, fps, width, height)
        except Exception as e:
            print(f"Warning: PyAV video writer initialization failed ({e}), falling back to OpenCV.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cv_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return OpenCVVideoWriterWrapper(cv_writer)
