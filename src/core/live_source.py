from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union
from urllib.parse import quote

from src.core.stream import BaseCameraStream
from src.core.types import FramePacket

logger = logging.getLogger("trace.stream.live")

# `--source camera` reads the classroom camera from these variables (in .env or the environment)
CAMERA_SOURCE_NAMES = ("camera", "rtsp", "ip-camera")
RTSP_URL_VAR = "CAMERA_RTSP_URL"


def load_env_file(path: Union[str, Path] = ".env") -> int:
    """
    Loads KEY=VALUE lines from a local .env file into os.environ (existing variables win).
    Keeps camera passwords out of the code and the command line. Returns the number of keys set.
    """
    path = Path(path)
    if not path.is_file():
        return 0
    loaded = 0
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def camera_url_from_env(env: Optional[Mapping[str, str]] = None) -> str:
    """
    Builds the classroom camera's RTSP URL from the environment:
    - CAMERA_RTSP_URL, used as is, or
    - CAMERA_IP + CAMERA_PASSWORD (+ optional CAMERA_USER, CAMERA_PORT, CAMERA_CHANNEL, CAMERA_SUBTYPE),
      in the Dahua / Imou format rtsp://user:pass@ip:554/cam/realmonitor?channel=1&subtype=0.
    """
    env = os.environ if env is None else env
    url = (env.get(RTSP_URL_VAR) or "").strip()
    if url:
        return url

    ip = (env.get("CAMERA_IP") or "").strip()
    password = env.get("CAMERA_PASSWORD") or ""
    if not ip or not password:
        raise ValueError(
            f"Camera not configured: set {RTSP_URL_VAR}, or CAMERA_IP and CAMERA_PASSWORD, in .env "
            "(see .env.example)."
        )
    user = (env.get("CAMERA_USER") or "admin").strip()
    port = (env.get("CAMERA_PORT") or "554").strip()
    channel = (env.get("CAMERA_CHANNEL") or "1").strip()
    subtype = (env.get("CAMERA_SUBTYPE") or "0").strip()
    return (
        f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}@{ip}:{port}"
        f"/cam/realmonitor?channel={channel}&subtype={subtype}"
    )


def resolve_source(source: str, env: Optional[Mapping[str, str]] = None) -> Tuple[str, str]:
    """
    Maps the --source value to (source_type, source_path):
    - "camera"            -> ("rtsp", URL from .env)
    - "rtsp://..."        -> ("rtsp", the URL)
    - "0", "1", ...       -> ("webcam", device index)
    - anything else       -> ("file", the path)
    """
    value = (source or "").strip()
    if value.lower() in CAMERA_SOURCE_NAMES:
        return "rtsp", camera_url_from_env(env)
    if value.lower().startswith(("rtsp://", "rtsps://")):
        return "rtsp", value
    if value.isdigit():
        return "webcam", value
    return "file", value


class LiveStream:
    """
    Wraps a live camera stream (RTSP or webcam) for the attendance run:
    - waits up to `max_wait_sec` for the next frame instead of ending the run on the first
      empty read (RTSP needs a few seconds to connect and reconnects on its own after a drop);
    - rebases timestamps to seconds since the first frame, so time-in-class and the live view
      read like a video clock; `timestamp_str` keeps the wall-clock time for the attendance report;
    - ends the run after `max_duration_sec`, if set.
    """

    def __init__(
        self,
        stream: BaseCameraStream,
        max_wait_sec: float = 30.0,
        max_duration_sec: Optional[float] = None,
        poll_sec: float = 0.05,
    ):
        self.stream = stream
        self.max_wait_sec = float(max_wait_sec)
        self.max_duration_sec = max_duration_sec
        self.poll_sec = poll_sec
        self._t0: Optional[float] = None
        self.ended_reason = ""

    def __getattr__(self, name):
        # camera_id, session_id, fps, is_running, start, release, probe, dropped_frames, ...
        if name == "stream":
            raise AttributeError(name)
        return getattr(self.stream, name)

    def read(self) -> Optional[FramePacket]:
        if self._t0 is not None and self.max_duration_sec and self._elapsed() >= self.max_duration_sec:
            self.ended_reason = f"reached --duration ({self.max_duration_sec / 60:.1f} min)"
            return None

        deadline = time.monotonic() + self.max_wait_sec
        while True:
            packet = self.stream.read()
            if packet is not None and packet.frame is not None:
                break
            if not self.stream.is_running:
                self.ended_reason = "camera stream stopped"
                return None
            if time.monotonic() >= deadline:
                self.ended_reason = f"no frames from the camera for {self.max_wait_sec:.0f}s"
                logger.warning("Live stream: %s", self.ended_reason)
                return None
            time.sleep(self.poll_sec)

        if self._t0 is None:
            self._t0 = packet.timestamp
        return replace(packet, timestamp=max(0.0, packet.timestamp - self._t0))

    def _elapsed(self) -> float:
        return time.time() - self._t0 if self._t0 is not None else 0.0
