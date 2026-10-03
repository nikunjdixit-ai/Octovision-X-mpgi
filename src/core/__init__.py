"""Core CV pipeline types, streaming abstractions, and lifecycle contracts."""
from .types import BoundingBox, TrackedPerson, FramePacket, StreamConfig
from .stream import BaseCameraStream, FileVideoStream, WebcamStream, RTSPStream, create_camera_stream, sanitize_url
from .config import CameraConfig, load_camera_config, load_cameras_config, ConfigurationError

__all__ = [
    "BoundingBox",
    "TrackedPerson",
    "FramePacket",
    "StreamConfig",
    "BaseCameraStream",
    "FileVideoStream",
    "WebcamStream",
    "RTSPStream",
    "create_camera_stream",
    "sanitize_url",
    "CameraConfig",
    "load_camera_config",
    "load_cameras_config",
    "ConfigurationError",
]
