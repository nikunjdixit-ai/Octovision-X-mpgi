import logging
import threading
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class CVPipelineAdapter:
    """
    Adapter integrating the backend API with the core CV AI pipeline (src/).
    Uses lazy initialization to avoid loading ML models during import.
    """
    def __init__(self, camera_id: str, session_id: str):
        self.camera_id = camera_id
        self.session_id = session_id
        self._source_type = None
        self._source_path = None
        self._registry_dir = None
        self._controller = None
        self._attendance_engine = None
        self._identity_engine = None
        self._registry = None
        self._reid_model = None
        self._config = None
        self._thread = None
        self._running = False
        self._error = None
        self._results = None
        self._state = "queued"
        self._state_lock = threading.RLock()
        self._stop_requested = False

    def configure_file(self, video_path: str, registry_dir: str) -> None:
        self._source_type = "file"
        self._source_path = video_path
        self._registry_dir = registry_dir

    def configure_rtsp(self, rtsp_url: str, registry_dir: str) -> None:
        self._source_type = "rtsp"
        self._source_path = rtsp_url
        self._registry_dir = registry_dir

    def configure_webcam(self, device_index: int | str = 0, registry_dir: str = "") -> None:
        self._source_type = "webcam"
        self._source_path = str(device_index)
        self._registry_dir = registry_dir

    def _validate_configuration(self) -> None:
        if self._source_type not in {"file", "rtsp", "webcam"}:
            raise ValueError("Pipeline source is not configured or supported.")

        if not isinstance(self._source_path, str) or not self._source_path.strip():
            raise ValueError("Pipeline source path is required.")

        if not isinstance(self._registry_dir, str) or not self._registry_dir.strip():
            raise ValueError("Student registry directory is required.")

    def start(self) -> None:
       with self._state_lock:
           if self._state in {"initializing", "running", "stopping"}:
               return

           if self._state != "queued":
               raise RuntimeError(
                   f"Cannot start pipeline in {self._state} state."
               )

           self._state = "initializing"
           self._error = None
           self._results = None
           self._stop_requested = False

       controller = None

       try:
           self._validate_configuration()

           from src.core.types import StreamConfig
           from src.core.stream import create_camera_stream
           from src.core.live_source import LiveStream
           from src.core.pipeline_controller import PipelineController
           from src.reid.reid_model import ReIDModel
           from src.identity.enrollment import StudentRegistry
           from src.identity.identity_engine import IdentityEngine
           from src.attendance.attendance_engine import AttendanceEngine, AttendanceConfig

           reid_model = ReIDModel()
           registry = StudentRegistry(reid_model=reid_model)
           registry.load(self._registry_dir)

           identity_engine = IdentityEngine(
               registry=registry,
               reid_model=reid_model
           )

           attendance_engine = AttendanceEngine(
               registry=registry,
               identity_engine=identity_engine,
               config=AttendanceConfig(),
               session_id=self.session_id,
               camera_id=self.camera_id
           )

           config = StreamConfig(
               source_type=self._source_type,
               source_path=self._source_path,
               camera_id=self.camera_id,
               session_id=self.session_id
           )

           if self._source_type in {"rtsp", "webcam"}:
               raw_stream = create_camera_stream(config)
               live_stream = LiveStream(raw_stream, max_wait_sec=30.0)
               controller = PipelineController(stream=live_stream)
           else:
               controller = PipelineController(config=config)

           controller.subscribe(attendance_engine.process_frame)

           with self._state_lock:
               if self._stop_requested:
                   self._state = "stopped"
                   self._running = False
                   cancelled = True
               else:
                   self._reid_model = reid_model
                   self._registry = registry
                   self._identity_engine = identity_engine
                   self._attendance_engine = attendance_engine
                   self._config = config
                   self._controller = controller

                   self._thread = threading.Thread(
                       target=self._run_loop,
                       daemon=True
                   )
                   self._running = True
                   self._state = "running"
                   self._thread.start()
                   cancelled = False

           if cancelled:
               try:
                   controller.stop()
               except Exception as e:
                   logger.warning("Error stopping cancelled pipeline: %s", e)
               return

       except Exception as e:
           if controller is not None:
               try:
                   controller.stop()
               except Exception as stop_error:
                   logger.warning("Error stopping controller: %s", stop_error)

           with self._state_lock:
               self._error = e
               self._running = False
               self._state = "stopped" if self._stop_requested else "failed"
               self._controller = None
               self._thread = None

           raise RuntimeError(f"Failed to start pipeline: {e}") from e

    def _run_loop(self) -> None:
       try:
           frames_processed = self._controller.run()

           if frames_processed <= 0:
               raise RuntimeError(
                   "Pipeline completed without processing any frames."
               )

           results = self._attendance_engine.get_session_summary()

           with self._state_lock:
               self._results = results
               self._state = (
                   "stopped" if self._stop_requested else "completed"
               )

       except Exception as e:
           logger.exception("Pipeline error")

           with self._state_lock:
               self._error = e
               self._state = (
                   "stopped" if self._stop_requested else "failed"
               )

       finally:
           with self._state_lock:
               self._running = False

    def stop(self) -> None:
       with self._state_lock:
           if self._state in {"completed", "failed", "stopped"}:
               if self._state == "stopped":
                   if self._thread is None or not self._thread.is_alive():
                       self._thread = None
                       self._controller = None
               return

           if self._state == "queued":
               self._stop_requested = True
               self._state = "stopped"
               self._running = False
               return

           if self._state == "initializing":
               self._stop_requested = True
               self._state = "stopping"
               return

           should_stop_controller = self._state == "running"
           self._stop_requested = True
           self._state = "stopping"

           controller = self._controller
           worker = self._thread

           if worker is None:
               return

       if should_stop_controller and controller is not None:
           try:
               controller.stop()
           except Exception as e:
               logger.warning("Error stopping controller: %s", e)

       if worker is threading.current_thread():
           return

       worker.join(timeout=5.0)

       if worker.is_alive():
           return

       with self._state_lock:
           self._running = False
           self._state = "stopped"
           self._controller = None
           self._thread = None

    def get_results(self) -> Dict[str, Any]:
        return self._results if self._results is not None else {}

    def get_status(self) -> str:
       with self._state_lock:
           if self._state in {"initializing", "running", "stopping"}:
               return "running"

           if self._state == "completed":
               return "done"

           return self._state
