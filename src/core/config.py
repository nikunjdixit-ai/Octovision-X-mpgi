from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .types import StreamConfig


class ConfigurationError(Exception):
    """Raised when configuration parsing or validation fails."""
    pass


ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")


def resolve_env_vars(value: Any) -> Any:
    """
    Recursively resolves environment variables in strings, lists, and dictionaries.
    Supports both ${VAR} and ${VAR:-default} fallback syntax.
    """
    if isinstance(value, str):
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default_val = match.group(2)
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            if default_val is not None:
                return default_val
            raise ConfigurationError(
                f"Required environment variable '${var_name}' is not set and has no default value."
            )

        return ENV_PATTERN.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    return value


@dataclass
class CameraConfig:
    """Structured configuration for a single camera stream and its processing parameters."""
    camera_id: str
    name: str = ""
    source_type: str = "file"
    source_path: str = ""
    session_id: str = "SESSION_01"
    fps: float = 30.0
    width: int = 1280
    height: int = 720
    transport: str = "tcp"
    connection_timeout_sec: float = 5.0
    target_fps: Optional[float] = None
    reconnect_interval_sec: float = 3.0
    buffer_size: int = 2
    drop_stale_frames: bool = True

    def to_stream_config(self) -> StreamConfig:
        """Converts to the core StreamConfig consumed by BaseCameraStream."""
        return StreamConfig(
            source_type=self.source_type,
            source_path=self.source_path,
            camera_id=self.camera_id,
            session_id=self.session_id,
            fps=self.fps,
            width=self.width,
            height=self.height,
            reconnect_interval_sec=self.reconnect_interval_sec,
            buffer_size=self.buffer_size,
            drop_stale_frames=self.drop_stale_frames,
            transport=self.transport,
            connection_timeout_sec=self.connection_timeout_sec,
            target_fps=self.target_fps,
        )


VALID_SOURCE_TYPES = {
    "file", "video", "mp4",
    "webcam", "camera", "cam",
    "rtsp", "ip_camera", "stream",
}


def _parse_camera_dict(data: Dict[str, Any]) -> CameraConfig:
    """Parses a single dictionary into a validated CameraConfig object."""
    # 1. Handle nested sections (camera, stream, processing) or flat keys
    camera_sec = data.get("camera", {}) if isinstance(data.get("camera"), dict) else {}
    stream_sec = data.get("stream", {}) if isinstance(data.get("stream"), dict) else {}
    proc_sec = data.get("processing", {}) if isinstance(data.get("processing"), dict) else {}

    # Camera ID
    camera_id = (
        camera_sec.get("camera_id")
        or data.get("camera_id")
        or camera_sec.get("id")
        or data.get("id")
    )
    if not camera_id:
        raise ConfigurationError("Missing required field 'camera_id' in camera configuration.")
    camera_id = str(camera_id).strip()

    name = str(camera_sec.get("name") or data.get("name") or camera_id)

    # Stream Type & Path
    source_type = (
        stream_sec.get("source_type")
        or data.get("source_type")
        or camera_sec.get("source_type")
        or "file"
    )
    source_type = str(source_type).lower().strip()
    if source_type not in VALID_SOURCE_TYPES:
        raise ConfigurationError(
            f"Invalid source_type '{source_type}' for camera '{camera_id}'. "
            f"Must be one of: {sorted(VALID_SOURCE_TYPES)}"
        )

    # Source Path / URL resolution
    source_path = (
        stream_sec.get("source_path")
        or stream_sec.get("rtsp_url")
        or stream_sec.get("url")
        or data.get("source_path")
        or data.get("rtsp_url")
        or camera_sec.get("source_path")
        or ""
    )
    source_path = str(source_path).strip()

    # Session ID
    session_id = str(
        camera_sec.get("session_id")
        or data.get("session_id")
        or "SESSION_01"
    ).strip()

    # Dimensions & FPS
    try:
        width = int(proc_sec.get("width") or stream_sec.get("width") or data.get("width") or 1280)
        height = int(proc_sec.get("height") or stream_sec.get("height") or data.get("height") or 720)
        fps = float(stream_sec.get("fps") or camera_sec.get("fps") or data.get("fps") or 30.0)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Invalid numeric stream parameters for camera '{camera_id}': {exc}")

    # Target FPS throttling
    raw_target_fps = proc_sec.get("target_fps") or data.get("target_fps")
    target_fps = float(raw_target_fps) if raw_target_fps is not None else None

    # RTSP specific settings
    transport = str(
        stream_sec.get("transport")
        or data.get("transport")
        or "tcp"
    ).lower().strip()
    if transport not in ("tcp", "udp"):
        transport = "tcp"

    reconnect_interval = float(
        stream_sec.get("reconnect_interval_sec")
        or camera_sec.get("reconnect_interval_sec")
        or data.get("reconnect_interval_sec")
        or 3.0
    )
    timeout_sec = float(
        stream_sec.get("connection_timeout_sec")
        or data.get("connection_timeout_sec")
        or 5.0
    )
    buffer_size = int(
        stream_sec.get("buffer_size")
        or camera_sec.get("buffer_size")
        or data.get("buffer_size")
        or 2
    )
    drop_stale = bool(
        stream_sec.get("drop_stale_frames", True)
        if "drop_stale_frames" in stream_sec
        else data.get("drop_stale_frames", True)
    )

    return CameraConfig(
        camera_id=camera_id,
        name=name,
        source_type=source_type,
        source_path=source_path,
        session_id=session_id,
        fps=fps,
        width=width,
        height=height,
        transport=transport,
        connection_timeout_sec=timeout_sec,
        target_fps=target_fps,
        reconnect_interval_sec=reconnect_interval,
        buffer_size=buffer_size,
        drop_stale_frames=drop_stale,
    )


def load_camera_config(source: Union[str, Path, Dict[str, Any]]) -> CameraConfig:
    """
    Loads, resolves environment variables, and validates a single camera configuration.
    Accepts a path to a YAML file or a pre-loaded dictionary.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
        except Exception as exc:
            raise ConfigurationError(f"Failed to parse YAML file {path}: {exc}")
    elif isinstance(source, dict):
        raw_data = source
    else:
        raise ConfigurationError(f"Unsupported configuration source type: {type(source)}")

    if not isinstance(raw_data, dict):
        raise ConfigurationError("Camera configuration must be a valid mapping / dictionary.")

    resolved_data = resolve_env_vars(raw_data)
    return _parse_camera_dict(resolved_data)


def load_cameras_config(source: Union[str, Path, Dict[str, Any], List[Any]]) -> List[CameraConfig]:
    """
    Loads, resolves environment variables, and validates a multi-camera configuration.
    Supports YAML with a 'cameras:' list or a direct list of camera specifications.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
        except Exception as exc:
            raise ConfigurationError(f"Failed to parse YAML file {path}: {exc}")
    elif isinstance(source, (dict, list)):
        raw_data = source
    else:
        raise ConfigurationError(f"Unsupported configuration source type: {type(source)}")

    resolved_data = resolve_env_vars(raw_data)

    if isinstance(resolved_data, dict):
        if "cameras" in resolved_data and isinstance(resolved_data["cameras"], list):
            items = resolved_data["cameras"]
        else:
            # Single camera dictionary, return as single-item list
            return [_parse_camera_dict(resolved_data)]
    elif isinstance(resolved_data, list):
        items = resolved_data
    else:
        raise ConfigurationError("Multi-camera configuration must be a list or dict with 'cameras' list.")

    camera_configs = []
    seen_ids = set()
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ConfigurationError(f"Camera definition at index {idx} must be a dictionary.")
        cfg = _parse_camera_dict(item)
        if cfg.camera_id in seen_ids:
            raise ConfigurationError(f"Duplicate camera_id '{cfg.camera_id}' detected at index {idx}.")
        seen_ids.add(cfg.camera_id)
        camera_configs.append(cfg)

    return camera_configs
