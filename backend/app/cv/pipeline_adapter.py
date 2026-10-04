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
        self._controller = None
        self._attendance_engine = None
        self._identity_engine = None
        self._registry = None
        self._reid_model = None
        self._config = None
        self._thread = None
        self._running = False
        self._error = None

    def configure_file(self, video_path: str, registry_dir: str) -> None:
        """Configures the pipeline for an MP4/MOV file with a specific class registry."""
        self._source_type = "file"
        self._source_path = video_path
        self._registry_dir = registry_dir

    def configure_rtsp(self, rtsp_url: str, registry_dir: str) -> None:
        """Configures the pipeline for an RTSP stream."""
        self._source_type = "rtsp"
        self._source_path = rtsp_url
        self._registry_dir = registry_dir

    def start(self) -> None:
        """Initializes heavy models and starts processing in a background thread."""
        if self._running:
            return

        # Lazy import of ML models
        from src.core.types import StreamConfig
        from src.core.pipeline_controller import PipelineController
        from src.reid.reid_model import ReIDModel
        from src.identity.enrollment import StudentRegistry
        from src.identity.identity_engine import IdentityEngine
        from src.attendance.attendance_engine import AttendanceEngine, AttendanceConfig

        try:
            self._reid_model = ReIDModel()
            self._registry = StudentRegistry(reid_model=self._reid_model)
            self._registry.load(self._registry_dir)

            self._identity_engine = IdentityEngine(
                registry=self._registry,
                reid_model=self._reid_model
            )

            attendance_config = AttendanceConfig()
            self._attendance_engine = AttendanceEngine(
                registry=self._registry,
                identity_engine=self._identity_engine,
                config=attendance_config,
                session_id=self.session_id,
                camera_id=self.camera_id
            )

            self._config = StreamConfig(
                source_type=self._source_type,
                source_path=self._source_path,
                camera_id=self.camera_id,
                session_id=self.session_id
            )

            self._controller = PipelineController(config=self._config)
            self._controller.subscribe(self._attendance_engine.process_frame)

            self._running = True
            self._error = None
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        except Exception as e:
            self.stop()
            raise RuntimeError(f"Failed to start pipeline: {e}") from e

    def _run_loop(self) -> None:
        try:
            self._controller.run()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._error = e
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
        if self._controller:
            try:
                self._controller.stop()
            except Exception as e:
                logger.warning(f"Error stopping controller: {e}")
            finally:
                self._controller = None

    def get_results(self) -> Dict[str, Any]:
        """Returns the normalized session summary from the attendance engine."""
        if not self._attendance_engine:
            return {}
        return self._attendance_engine.get_session_summary()

    def get_status(self) -> str:
        """Returns the current pipeline status."""
        if self._running:
            return "running"
        if self._error:
            return "failed"
        if self._attendance_engine is not None:
            return "done"
        return "queued"
