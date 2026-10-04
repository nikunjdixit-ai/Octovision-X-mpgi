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

def test_pipeline_adapter_stop():
    adapter = CVPipelineAdapter("CAM1", "SES1")
    adapter._running = True
    adapter.stop()
    assert not adapter._running
