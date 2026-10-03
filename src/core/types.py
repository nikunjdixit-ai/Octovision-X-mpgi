from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Any
import numpy as np


@dataclass
class BoundingBox:
    """Represents an axis-aligned bounding box with confidence score."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    class_id: int = 0

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Ground contact foot position for spatial zone intrusion detection."""
        return (self.x1 + self.x2) // 2, self.y2

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_tlwh(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.width, self.height)


@dataclass
class TrackedPerson:
    """
    Represents a detected and tracked person in a specific camera stream session.
    Guarantees globally namespaced identity via stream_track_uid to prevent
    ID collisions across different cameras or recording sessions.
    """
    camera_id: str
    session_id: str
    track_id: int
    bbox: BoundingBox
    frame_id: int
    timestamp: str = ""
    stream_track_uid: str = field(init=False)

    def __post_init__(self):
        self.stream_track_uid = f"{self.camera_id}_{self.session_id}_{self.track_id}"

    def extract_crop(self, frame: np.ndarray, min_w: int = 10, min_h: int = 20) -> Optional[np.ndarray]:
        """Extracts the person image crop directly in RAM without disk I/O."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x1 = max(0, min(self.bbox.x1, w - 1))
        y1 = max(0, min(self.bbox.y1, h - 1))
        x2 = max(0, min(self.bbox.x2, w))
        y2 = max(0, min(self.bbox.y2, h))

        if (x2 - x1) < min_w or (y2 - y1) < min_h:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop.copy() if crop.size > 0 else None


@dataclass
class FramePacket:
    """Container for raw video frames passing through the pipeline."""
    frame_id: int
    timestamp: float
    timestamp_str: str
    frame: np.ndarray
    shape: Tuple[int, int, int]
    camera_id: str
    session_id: str


@dataclass
class StreamConfig:
    """Configuration for video/stream ingestion."""
    source_type: str = "file"  # "file", "webcam", "rtsp"
    source_path: str = ""      # file path, webcam index (as str or int), or RTSP URL
    camera_id: str = "CAM_01"
    session_id: str = "SESSION_01"
    fps: float = 30.0
    width: int = 1280
    height: int = 720
    reconnect_interval_sec: float = 3.0
    buffer_size: int = 2
    drop_stale_frames: bool = True
    transport: str = "tcp"                   # "tcp" or "udp" for RTSP streams
    connection_timeout_sec: float = 5.0      # Socket connection timeout for RTSP probe/reconnect
    target_fps: Optional[float] = None       # Decoupled processing rate for downstream inference

