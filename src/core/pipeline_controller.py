from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import FramePacket, TrackedPerson, StreamConfig
from src.core.stream import BaseCameraStream, create_camera_stream
from src.core.metrics import PipelinePerformanceTracker
from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker

logger = logging.getLogger("trace.pipeline")

SubscriberCallback = Callable[[FramePacket, List[TrackedPerson]], Any]


class PipelineController:
    """
    Reusable central Computer Vision pipeline controller.
    Coordinates: Camera Ingestion -> Detection -> Tracking -> Structured Event Dispatch.
    Powers both CCTV Incident Intelligence and Automated Classroom Attendance.
    """

    def __init__(
        self,
        stream: Optional[BaseCameraStream] = None,
        config: Optional[StreamConfig] = None,
        detector: Optional[PersonDetector] = None,
        tracker: Optional[PersonTracker] = None,
        model_path: str = "yolov8n.pt",
        camera_id: str = "CAM_01",
        session_id: str = "SESSION_01",
        conf_thresh: float = 0.40,
        iou_thresh: float = 0.45,
        target_fps: Optional[float] = None,
    ):
        self.camera_id = camera_id
        self.session_id = session_id
        self.config = config
        self.target_fps = target_fps or (config.target_fps if config else None)
        self._min_frame_interval: Optional[float] = (
            (1.0 / self.target_fps) if (self.target_fps and self.target_fps > 0) else None
        )
        self._last_process_time: float = 0.0

        # 1. Initialize Stream
        if stream is not None:
            self.stream = stream
            self.camera_id = stream.camera_id
            self.session_id = stream.session_id
        elif config is not None:
            self.camera_id = config.camera_id
            self.session_id = config.session_id
            self.stream = create_camera_stream(config)
        else:
            raise ValueError("PipelineController requires either a stream instance or a StreamConfig.")

        # 2. Initialize Detector (reuse provided or create once)
        if detector is not None:
            self.detector = detector
        else:
            self.detector = PersonDetector(
                model_path=model_path,
                conf=conf_thresh,
                iou=iou_thresh
            )

        # 3. Initialize Tracker (reuse provided or create once)
        if tracker is not None:
            self.tracker = tracker
            self.tracker.camera_id = self.camera_id
            self.tracker.session_id = self.session_id
        else:
            self.tracker = PersonTracker(
                model_path=model_path,
                conf=conf_thresh,
                iou=iou_thresh,
                camera_id=self.camera_id,
                session_id=self.session_id
            )

        self._subscribers: List[SubscriberCallback] = []
        self._is_running: bool = False
        self.processed_frames: int = 0
        self.total_detections: int = 0
        self.performance = PipelinePerformanceTracker(camera_id=self.camera_id)


    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def subscribers(self) -> List[SubscriberCallback]:
        return list(self._subscribers)

    def subscribe(self, callback: SubscriberCallback) -> None:
        """Registers a callback hook to receive (FramePacket, List[TrackedPerson]) after each frame."""
        if not callable(callback):
            raise TypeError("Subscriber callback must be callable.")
        if callback not in self._subscribers:
            self._subscribers.append(callback)
            logger.debug(f"Registered subscriber callback: {callback}")

    def unsubscribe(self, callback: SubscriberCallback) -> None:
        """Removes a registered callback hook."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.debug(f"Unregistered subscriber callback: {callback}")

    def _notify_subscribers(self, packet: FramePacket, tracked_persons: List[TrackedPerson]) -> None:
        """Invokes all subscribers safely without letting one failure break the entire pipeline."""
        for callback in self._subscribers:
            try:
                callback(packet, tracked_persons)
            except Exception as exc:
                logger.error(f"Error in pipeline subscriber callback {callback}: {exc}", exc_info=False)

    def start(self) -> bool:
        """Starts stream capture and resets tracker state."""
        if self._is_running:
            return True

        if not self.stream.is_running:
            if not self.stream.start():
                logger.error(f"Failed to start camera stream ({self.camera_id}).")
                self._is_running = False
                return False

        self.tracker.reset(self.session_id)
        self._is_running = True
        logger.info(f"PipelineController started for camera={self.camera_id}, session={self.session_id}")
        return True

    def reset(self, new_session_id: Optional[str] = None) -> None:
        """Resets the tracker Kalman state and session identifier."""
        if new_session_id is not None:
            self.session_id = new_session_id
            self.stream.session_id = new_session_id

        self.tracker.reset(self.session_id)
        self.processed_frames = 0
        self.total_detections = 0
        logger.info(f"PipelineController state reset for session={self.session_id}")

    def step(self) -> Tuple[Optional[FramePacket], List[TrackedPerson]]:
        """
        Processes a single frame through the pipeline:
        Reads frame -> ByteTrack tracking -> Dispatches to subscribers -> Returns results.
        """
        if not self._is_running:
            if not self.start():
                return None, []

        packet = self.stream.read()
        if packet is None or packet.frame is None:
            logger.info(f"Stream completed or frame unavailable for camera={self.camera_id}.")
            self._is_running = False
            return None, []

        # Throttling / decimation if target_fps is configured
        if self._min_frame_interval is not None and self._last_process_time > 0:
            elapsed_since_last = time.time() - self._last_process_time
            if elapsed_since_last < self._min_frame_interval:
                time.sleep(self._min_frame_interval - elapsed_since_last)

        self.processed_frames += 1

        t0 = time.time()
        # Track persons directly in RAM without saving crops to disk
        tracked_persons = self.tracker.track_frame(
            frame=packet.frame,
            frame_id=packet.frame_id,
            timestamp=packet.timestamp_str
        )
        latency_ms = (time.time() - t0) * 1000.0

        self.performance.record_frame(latency_ms)
        self.performance.dropped_frames = getattr(self.stream, "dropped_frames", 0)
        self._last_process_time = time.time()

        self.total_detections += len(tracked_persons)

        # Dispatch structured event to all registered hooks
        self._notify_subscribers(packet, tracked_persons)

        return packet, tracked_persons

    def run(self, max_frames: Optional[int] = None) -> int:
        """
        Runs the pipeline continuously until the stream terminates or max_frames is reached.
        """
        if not self.start():
            return 0

        frames_run = 0
        try:
            while self._is_running:
                if max_frames is not None and frames_run >= max_frames:
                    break

                packet, _ = self.step()
                if packet is None:
                    break

                frames_run += 1
        finally:
            self.stop()

        return frames_run

    def stop(self) -> None:
        """Stops the pipeline and cleanly releases the camera stream."""
        self._is_running = False
        if self.stream is not None:
            self.stream.release()
        logger.info(f"PipelineController stopped ({self.camera_id}). Processed {self.processed_frames} frames.")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Returns comprehensive throughput, latency, and hardware utilization telemetry."""
        return self.performance.summary()

    def format_performance_report(self) -> str:
        """Formats performance summary as a readable diagnostic string."""
        return self.performance.format_summary()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
