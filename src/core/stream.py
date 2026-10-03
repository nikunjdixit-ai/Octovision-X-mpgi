from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

from .types import FramePacket, StreamConfig

logger = logging.getLogger("trace.stream")


def sanitize_url(url: str) -> str:
    """Masks credentials in RTSP/HTTP URLs for safe logging."""
    if not url:
        return ""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)


class BaseCameraStream(ABC):
    """Abstract base class for all video and camera ingestion sources."""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.camera_id = config.camera_id
        self.session_id = config.session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_id = 0
        self._fps: float = config.fps
        self._width: int = config.width
        self._height: int = config.height
        self._is_running: bool = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def fps(self) -> float:
        return self._fps if self._fps > 0 else 30.0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @abstractmethod
    def start(self) -> bool:
        """Starts the stream connection or worker thread."""
        pass

    @abstractmethod
    def read(self) -> Optional[FramePacket]:
        """Returns the next FramePacket, or None if stream ended or unavailable."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Closes the stream and frees hardware/network resources."""
        pass

    @abstractmethod
    def probe(self, timeout_sec: Optional[float] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Lightweight preflight connection check.
        Returns (success: bool, diagnosis_message: str, metadata: dict).
        """
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def _format_timestamp(self, elapsed_seconds: float) -> str:
        td = timedelta(seconds=elapsed_seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        millis = int((elapsed_seconds - total_seconds) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


class FileVideoStream(BaseCameraStream):
    """Ingests local video files (MP4, AVI, MOV, MKV) reliably."""

    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        file_path = self.config.source_path
        if not os.path.exists(file_path):
            logger.error(f"Video file does not exist: {file_path}")
            self._is_running = False
            return False

        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {file_path}")
            self._is_running = False
            return False

        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or self.config.fps
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._is_running = True
        logger.info(f"Opened FileVideoStream '{file_path}' ({self._width}x{self._height} @ {self._fps:.2f} FPS)")
        return True

    def read(self) -> Optional[FramePacket]:
        if not self._is_running or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.info("FileVideoStream reached end-of-file.")
            self._is_running = False
            return None

        self.frame_id += 1
        elapsed_sec = self.frame_id / self.fps
        timestamp_str = self._format_timestamp(elapsed_sec)

        return FramePacket(
            frame_id=self.frame_id,
            timestamp=elapsed_sec,
            timestamp_str=timestamp_str,
            frame=frame,
            shape=frame.shape,
            camera_id=self.camera_id,
            session_id=self.session_id,
        )

    def release(self) -> None:
        self._is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info(f"Released FileVideoStream ({self.camera_id}).")

    def probe(self, timeout_sec: Optional[float] = None) -> Tuple[bool, str, Dict[str, Any]]:
        file_path = self.config.source_path
        if not os.path.exists(file_path):
            return False, f"Video file does not exist: {file_path}", {}
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False, f"Failed to open video file: {file_path}", {}
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, f"Video file is empty or corrupted: {file_path}", {}
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or self.config.fps
        cap.release()
        return True, "OK", {"width": w, "height": h, "fps": round(fps, 2)}



class WebcamStream(BaseCameraStream):
    """Ingests live frames from local USB or laptop webcams."""

    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self.cap: Optional[cv2.VideoCapture] = None
        try:
            self.device_index = int(config.source_path)
        except (ValueError, TypeError):
            self.device_index = 0

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            logger.error(f"Cannot open webcam device index: {self.device_index}")
            self._is_running = False
            return False

        if self.config.width > 0 and self.config.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._is_running = True
        logger.info(f"Opened WebcamStream index {self.device_index} ({self._width}x{self._height} @ {self._fps:.2f} FPS)")
        return True

    def read(self) -> Optional[FramePacket]:
        if not self._is_running or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Webcam frame capture failed.")
            return None

        self.frame_id += 1
        now_ts = time.time()
        timestamp_str = datetime.fromtimestamp(now_ts).strftime("%H:%M:%S.%f")[:-3]

        return FramePacket(
            frame_id=self.frame_id,
            timestamp=now_ts,
            timestamp_str=timestamp_str,
            frame=frame,
            shape=frame.shape,
            camera_id=self.camera_id,
            session_id=self.session_id,
        )

    def release(self) -> None:
        self._is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info(f"Released WebcamStream ({self.camera_id}).")

    def probe(self, timeout_sec: Optional[float] = None) -> Tuple[bool, str, Dict[str, Any]]:
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            return False, f"Cannot access webcam device index {self.device_index}", {}
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, f"Cannot capture frame from webcam index {self.device_index}", {}
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()
        return True, "OK", {"width": w, "height": h, "fps": round(fps, 2)}



class RTSPStream(BaseCameraStream):
    """
    Ingests live RTSP / IP camera streams with asynchronous grabber thread,
    frame dropping to eliminate latency lag, and automatic reconnection.
    Hardened for enterprise NVRs with TCP transport and configurable timeouts.
    """

    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self.rtsp_url = config.source_path
        self.transport = (config.transport or "tcp").lower().strip()
        if self.transport not in ("tcp", "udp"):
            self.transport = "tcp"
        self.connection_timeout_sec = config.connection_timeout_sec or 5.0
        self.frame_queue: queue.Queue = queue.Queue(maxsize=max(1, config.buffer_size))
        self.reconnect_interval = config.reconnect_interval_sec
        self.drop_stale_frames = config.drop_stale_frames
        self.dropped_frames: int = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connected = False

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        """Creates a VideoCapture instance configured with FFmpeg TCP transport and timeouts."""
        transport_opt = f"rtsp_transport;{self.transport}"
        timeout_us = int(self.connection_timeout_sec * 1_000_000)
        # Configure OpenCV FFmpeg environment variable for RTSP
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"{transport_opt}|stimeout;{timeout_us}"

        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.connection_timeout_sec * 1000))
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.connection_timeout_sec * 1000))
        return cap

    def start(self) -> bool:
        self._stop_event.clear()
        self._is_running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()
        safe_url = sanitize_url(self.rtsp_url)
        logger.info(
            f"Started asynchronous RTSP grabber thread for '{safe_url}' "
            f"(transport={self.transport}, timeout={self.connection_timeout_sec}s)"
        )
        return True

    def _connect(self) -> Optional[cv2.VideoCapture]:
        safe_url = sanitize_url(self.rtsp_url)
        logger.info(f"Attempting connection to RTSP stream: {safe_url}")
        cap = self._create_capture()
        if cap is not None and cap.isOpened():
            self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._connected = True
            logger.info(f"RTSP stream connected: {safe_url} ({self._width}x{self._height} @ {self._fps:.1f} FPS)")
            return cap
        else:
            self._connected = False
            logger.warning(f"RTSP connection failed: {safe_url}")
            return None

    def _capture_worker(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        while not self._stop_event.is_set():
            if cap is None or not cap.isOpened():
                cap = self._connect()
                if cap is None or not cap.isOpened():
                    time.sleep(self.reconnect_interval)
                    continue

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("RTSP stream read dropped. Reconnecting...")
                if cap is not None:
                    cap.release()
                cap = None
                self._connected = False
                time.sleep(self.reconnect_interval)
                continue

            self.frame_id += 1
            now_ts = time.time()
            timestamp_str = datetime.fromtimestamp(now_ts).strftime("%H:%M:%S.%f")[:-3]

            packet = FramePacket(
                frame_id=self.frame_id,
                timestamp=now_ts,
                timestamp_str=timestamp_str,
                frame=frame,
                shape=frame.shape,
                camera_id=self.camera_id,
                session_id=self.session_id,
            )

            # Frame queue management to prevent latency buildup
            if self.drop_stale_frames:
                try:
                    self.frame_queue.put_nowait(packet)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()  # Drop oldest frame
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass
                    try:
                        self.frame_queue.put_nowait(packet)
                    except queue.Full:
                        self.dropped_frames += 1
            else:
                try:
                    self.frame_queue.put(packet, timeout=0.1)
                except queue.Full:
                    pass

        if cap is not None:
            cap.release()
        logger.info("RTSP worker thread terminated cleanly.")

    def read(self) -> Optional[FramePacket]:
        if not self._is_running:
            return None
        try:
            return self.frame_queue.get(timeout=0.5)
        except queue.Empty:
            return None

    def probe(self, timeout_sec: Optional[float] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Lightweight preflight connection check for RTSP streams.
        Verifies endpoint reachability, credentials, and initial frame without spawning worker threads.
        """
        timeout = timeout_sec or self.connection_timeout_sec
        safe_url = sanitize_url(self.rtsp_url)
        res_container: Dict[str, Any] = {"success": False, "message": "", "meta": {}}

        def _probe_worker():
            try:
                cap = self._create_capture()
                if cap is None or not cap.isOpened():
                    res_container["message"] = f"Failed to connect to RTSP endpoint: {safe_url}"
                    return
                ret, frame = cap.read()
                if not ret or frame is None:
                    res_container["message"] = "Connected to RTSP stream, but failed to retrieve initial frame."
                    cap.release()
                    return
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
                cap.release()
                res_container["success"] = True
                res_container["message"] = "OK"
                res_container["meta"] = {"width": w, "height": h, "fps": round(fps, 2)}
            except Exception as exc:
                res_container["message"] = f"RTSP probe error: {exc}"

        probe_thread = threading.Thread(target=_probe_worker, daemon=True)
        probe_thread.start()
        probe_thread.join(timeout=timeout)

        if probe_thread.is_alive():
            return False, f"Connection attempt timed out after {timeout:.1f}s ({safe_url})", {}

        return res_container["success"], res_container["message"], res_container["meta"]

    def release(self) -> None:
        self._is_running = False
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        logger.info(f"Released RTSPStream ({self.camera_id}).")


def create_camera_stream(config: StreamConfig) -> BaseCameraStream:
    """Factory function returning the appropriate camera stream implementation."""
    source_type = config.source_type.lower().strip()
    if source_type in ("file", "video", "mp4"):
        return FileVideoStream(config)
    elif source_type in ("webcam", "camera", "cam"):
        return WebcamStream(config)
    elif source_type in ("rtsp", "ip_camera", "stream"):
        return RTSPStream(config)
    else:
        raise ValueError(f"Unsupported source_type '{config.source_type}'. Use 'file', 'webcam', or 'rtsp'.")
