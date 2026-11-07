from contextlib import contextmanager
import threading
from threading import Thread as RealThread
import pytest
from unittest.mock import MagicMock, patch
from backend.app.cv.pipeline_adapter import CVPipelineAdapter

def test_pipeline_adapter_lazy_import():
    adapter = CVPipelineAdapter("CAM1", "SES1")
    assert adapter._controller is None
    assert not adapter._running

def test_pipeline_adapter_configure_file():
    adapter = CVPipelineAdapter("CAM1", "SES1")
    adapter.configure_file("test.mp4", "registry/")
    assert adapter._source_type == "file"
    assert adapter._source_path == "test.mp4"

def test_pipeline_adapter_configure_webcam():
    adapter = CVPipelineAdapter("CAM1", "SES1")
    adapter.configure_webcam(0, "registry/")
    assert adapter._source_type == "webcam"
    assert adapter._source_path == "0"

@patch('backend.app.cv.pipeline_adapter.threading.Thread')
def test_pipeline_adapter_start_mocked(mock_thread):
    # Mocking heavy src dependencies
    with patch('src.reid.reid_model.ReIDModel'), \
         patch('src.identity.enrollment.StudentRegistry'), \
         patch('src.identity.identity_engine.IdentityEngine'), \
         patch('src.attendance.attendance_engine.AttendanceEngine'), \
         patch('src.core.pipeline_controller.PipelineController'):
        
        adapter = CVPipelineAdapter("CAM1", "SES1")
        adapter.configure_file("test.mp4", "registry/")
        adapter.start()
        
        assert adapter._running
        mock_thread.assert_called_once()
        assert adapter._controller is not None

@patch('backend.app.cv.pipeline_adapter.threading.Thread')
def test_pipeline_adapter_rtsp_wrapped_in_livestream(mock_thread):
    with patch('src.reid.reid_model.ReIDModel'), \
         patch('src.identity.enrollment.StudentRegistry'), \
         patch('src.identity.identity_engine.IdentityEngine'), \
         patch('src.attendance.attendance_engine.AttendanceEngine'), \
         patch('src.core.stream.create_camera_stream') as mock_create_stream, \
         patch('src.core.live_source.LiveStream') as mock_live_stream_cls, \
         patch('src.core.pipeline_controller.PipelineController') as mock_controller_cls:

        raw_stream = MagicMock()
        mock_create_stream.return_value = raw_stream
        wrapped_stream = MagicMock()
        mock_live_stream_cls.return_value = wrapped_stream

        adapter = CVPipelineAdapter("CAM1", "SES1")
        adapter.configure_rtsp("rtsp://192.168.1.50:554/live", "registry/")
        adapter.start()

        mock_create_stream.assert_called_once()
        mock_live_stream_cls.assert_called_once_with(raw_stream, max_wait_sec=30.0)
        mock_controller_cls.assert_called_once_with(stream=wrapped_stream)

@patch('backend.app.cv.pipeline_adapter.threading.Thread')
def test_pipeline_adapter_webcam_wrapped_in_livestream(mock_thread):
    with patch('src.reid.reid_model.ReIDModel'), \
         patch('src.identity.enrollment.StudentRegistry'), \
         patch('src.identity.identity_engine.IdentityEngine'), \
         patch('src.attendance.attendance_engine.AttendanceEngine'), \
         patch('src.core.stream.create_camera_stream') as mock_create_stream, \
         patch('src.core.live_source.LiveStream') as mock_live_stream_cls, \
         patch('src.core.pipeline_controller.PipelineController') as mock_controller_cls:

        raw_stream = MagicMock()
        mock_create_stream.return_value = raw_stream
        wrapped_stream = MagicMock()
        mock_live_stream_cls.return_value = wrapped_stream

        adapter = CVPipelineAdapter("CAM1", "SES1")
        adapter.configure_webcam(0, "registry/")
        adapter.start()

        mock_create_stream.assert_called_once()
        mock_live_stream_cls.assert_called_once_with(raw_stream, max_wait_sec=30.0)
        mock_controller_cls.assert_called_once_with(stream=wrapped_stream)

@patch('backend.app.cv.pipeline_adapter.threading.Thread')
def test_pipeline_adapter_file_not_wrapped_in_livestream(mock_thread):
    with patch('src.reid.reid_model.ReIDModel'), \
         patch('src.identity.enrollment.StudentRegistry'), \
         patch('src.identity.identity_engine.IdentityEngine'), \
         patch('src.attendance.attendance_engine.AttendanceEngine'), \
         patch('src.core.live_source.LiveStream') as mock_live_stream_cls, \
         patch('src.core.pipeline_controller.PipelineController') as mock_controller_cls:

        adapter = CVPipelineAdapter("CAM1", "SES1")
        adapter.configure_file("test.mp4", "registry/")
        adapter.start()

        mock_live_stream_cls.assert_not_called()
        mock_controller_cls.assert_called_once_with(config=adapter._config)

def test_initial_empty_read_does_not_terminate_live_stream():
    import numpy as np
    from src.core.types import FramePacket
    from src.core.live_source import LiveStream
    from src.core.pipeline_controller import PipelineController

    class FakeCameraStream:
        camera_id = "CAM1"
        session_id = "SES1"
        is_running = True

        def __init__(self, empty_reads=2):
            self.empty_reads = empty_reads
            self.read_count = 0

        def start(self):
            self.is_running = True
            return True

        def read(self):
            self.read_count += 1
            if self.read_count <= self.empty_reads:
                return None
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            return FramePacket(
                frame_id=1,
                timestamp=1700000000.0,
                timestamp_str="10:00:00.000",
                frame=frame,
                shape=frame.shape,
                camera_id=self.camera_id,
                session_id=self.session_id,
            )

        def release(self):
            self.is_running = False

    # 1. Unwrapped stream terminates on initial empty read
    raw_stream = FakeCameraStream(empty_reads=1)
    with patch('src.core.pipeline_controller.PersonDetector'), \
         patch('src.core.pipeline_controller.PersonTracker'):
        unwrapped_controller = PipelineController(stream=raw_stream)
        unwrapped_controller.start()
        packet, _ = unwrapped_controller.step()
        assert packet is None
        assert not unwrapped_controller.is_running

    # 2. Wrapped in LiveStream, initial empty read is waited out and processing continues
    raw_stream = FakeCameraStream(empty_reads=1)
    live_stream = LiveStream(raw_stream, max_wait_sec=2.0, poll_sec=0.01)
    with patch('src.core.pipeline_controller.PersonDetector'), \
         patch('src.core.pipeline_controller.PersonTracker'):
        live_controller = PipelineController(stream=live_stream)
        live_controller.start()
        packet, _ = live_controller.step()
        assert packet is not None
        assert live_controller.is_running
        assert packet.timestamp == 0.0  # rebased to 0.0


def test_pipeline_adapter_stop():
    adapter = CVPipelineAdapter("CAM1", "SES1")
    adapter._running = True
    adapter.stop()
    assert not adapter._running

@contextmanager
def mock_cv_pipeline():
    with patch('src.reid.reid_model.ReIDModel') as reid, \
         patch('src.identity.enrollment.StudentRegistry') as reg, \
         patch('src.identity.identity_engine.IdentityEngine') as ident, \
         patch('src.attendance.attendance_engine.AttendanceEngine') as att, \
         patch('src.core.pipeline_controller.PipelineController') as ctrl:
        yield {"reid": reid, "controller": ctrl, "attendance": att}

def test_lifecycle_source_validation():
    adapter = CVPipelineAdapter("CAM1", "SES1")
    assert adapter.get_status() == "queued"
    with pytest.raises(RuntimeError):
        adapter.start()
    assert adapter.get_status() == "failed"

def test_lifecycle_initialization_failure():
    with mock_cv_pipeline() as mocks:
        mocks["reid"].side_effect = RuntimeError("Weights missing")
        adapter = CVPipelineAdapter("CAM1", "SES1")
        adapter.configure_file("test.mp4", "registry/")
        with pytest.raises(RuntimeError):
            adapter.start()
        assert adapter.get_status() == "failed"
        assert adapter._thread is None

@pytest.mark.parametrize("run_behavior,expected_status", [
    (10, "done"),
    (0, "failed"),
    (RuntimeError("Worker failure"), "failed"),
])
def test_lifecycle_processing_outcomes(run_behavior, expected_status):
    with mock_cv_pipeline() as mocks:
        if isinstance(run_behavior, Exception):
            mocks["controller"].return_value.run.side_effect = run_behavior
        else:
            mocks["controller"].return_value.run.return_value = run_behavior

        adapter = CVPipelineAdapter("CAM1", "SES1")
        adapter.configure_file("test.mp4", "registry/")
        adapter.start()
        adapter._thread.join(timeout=2.0)

        assert not adapter._thread.is_alive()
        assert adapter.get_status() == expected_status

def test_lifecycle_queued_adapter_stop():
    adapter = CVPipelineAdapter("CAM1", "SES1")
    adapter.configure_file("test.mp4", "registry/")
    adapter.stop()
    adapter.stop()
    assert adapter.get_status() == "stopped"
    with pytest.raises(RuntimeError):
        adapter.start()
    assert adapter.get_status() == "stopped"

def test_lifecycle_running_adapter_stop():
    worker_started = threading.Event()
    stop_called = threading.Event()

    def fake_run():
        worker_started.set()
        stop_called.wait(timeout=2.0)
        return 10

    with mock_cv_pipeline() as mocks:
        mocks["controller"].return_value.run.side_effect = fake_run
        mocks["controller"].return_value.stop.side_effect = stop_called.set

        adapter = CVPipelineAdapter("CAM1", "SES1")
        adapter.configure_file("test.mp4", "registry/")
        adapter.start()

        assert worker_started.wait(timeout=2.0)
        worker_thread = adapter._thread

        adapter.stop()
        adapter.stop()

        assert not worker_thread.is_alive()
        assert adapter.get_status() == "stopped"
        assert mocks["controller"].return_value.stop.call_count == 1

@patch('backend.app.cv.pipeline_adapter.threading.Thread')
def test_lifecycle_concurrent_start(mock_thread):
    first_init_started = threading.Event()
    allow_init_finish = threading.Event()
    second_start_returned = threading.Event()
    init_call_count = 0
    init_lock = threading.Lock()

    def fake_reid_init(*args, **kwargs):
        nonlocal init_call_count
        with init_lock:
            init_call_count += 1
        first_init_started.set()
        allow_init_finish.wait(timeout=2.0)

    with mock_cv_pipeline() as mocks:
        mocks["reid"].side_effect = fake_reid_init
        adapter = CVPipelineAdapter("CAM1", "SES1")
        adapter.configure_file("test.mp4", "registry/")

        barrier = threading.Barrier(2)
        errors = []

        def caller1():
            try:
                barrier.wait(timeout=2.0)
                adapter.start()
            except Exception as e:
                errors.append(e)

        def caller2():
            try:
                barrier.wait(timeout=2.0)
                first_init_started.wait(timeout=2.0)
                adapter.start()
                second_start_returned.set()
            except Exception as e:
                errors.append(e)

        t1 = RealThread(target=caller1)
        t2 = RealThread(target=caller2)
        t1.start()
        t2.start()

        try:
            assert first_init_started.wait(timeout=2.0)
            assert second_start_returned.wait(timeout=1.0)
        finally:
            allow_init_finish.set()
            t1.join(timeout=2.0)
            t2.join(timeout=2.0)

        assert not t1.is_alive()
        assert not t2.is_alive()
        assert not errors
        assert init_call_count == 1
        assert mock_thread.call_count == 1

def test_lifecycle_join_timeout():
    worker_started = threading.Event()
    release_worker = threading.Event()

    with mock_cv_pipeline() as mocks:
        mocks["controller"].return_value.run.side_effect = lambda: (
            worker_started.set(),
            release_worker.wait(timeout=2.0),
            10,
        )[2]

        adapter = CVPipelineAdapter("CAM1", "SES1")
        adapter.configure_file("test.mp4", "registry/")
        adapter.start()

        assert worker_started.wait(timeout=2.0)
        worker_thread = adapter._thread

        real_join = worker_thread.join
        worker_thread.join = lambda timeout=None: real_join(timeout=0.02)

        try:
            adapter.stop()
            assert worker_thread.is_alive()
            assert adapter.get_status() == "running"
            assert adapter._thread is not None
            assert adapter._controller is not None
        finally:
            release_worker.set()
            real_join(timeout=2.0)

def test_lifecycle_result_caching():
    with mock_cv_pipeline() as mocks:
        mocks["attendance"].return_value.get_session_summary.return_value = {"present": 5}
        mocks["controller"].return_value.run.return_value = 10

        adapter = CVPipelineAdapter("CAM1", "SES1")
        adapter.configure_file("test.mp4", "registry/")
        adapter.start()
        adapter._thread.join(timeout=2.0)

        first_results = adapter.get_results()
        second_results = adapter.get_results()

        assert first_results == second_results
        assert mocks["attendance"].return_value.get_session_summary.call_count == 1
