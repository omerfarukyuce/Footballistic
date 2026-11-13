"""
Multi-object tracking module using ByteTrack for persistent player and ball IDs.
"""

from typing import Optional
import supervision as sv


class FootballTracker:
    """Tracks players and referee across frames using ByteTrack."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 25
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the tracker with current frame detections and assigns tracker_id.
        Note: Tracking is typically applied to players (person class).
        """
        tracked_detections = self.tracker.update_with_detections(detections)
        return tracked_detections
