import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.live_source import LiveStream, camera_url_from_env, load_env_file, resolve_source
from src.core.stream import sanitize_url
from src.core.types import FramePacket


def test_resolve_source_kinds():
    env = {"CAMERA_IP": "192.168.1.50", "CAMERA_PASSWORD": "pw"}
    assert resolve_source("camera", env) == (
        "rtsp",
        "rtsp://admin:pw@192.168.1.50:554/cam/realmonitor?channel=1&subtype=0",
    )
    assert resolve_source("rtsp://u:p@10.0.0.2:554/live", env) == ("rtsp", "rtsp://u:p@10.0.0.2:554/live")
    assert resolve_source("0", env) == ("webcam", "0")
    assert resolve_source("data/class.mp4", env) == ("file", "data/class.mp4")


def test_camera_url_encodes_password_and_prefers_full_url():
    url = camera_url_from_env({"CAMERA_IP": "192.168.1.50", "CAMERA_PASSWORD": "a@b:c#1", "CAMERA_SUBTYPE": "1"})
    assert url == "rtsp://admin:a%40b%3Ac%231@192.168.1.50:554/cam/realmonitor?channel=1&subtype=1"
    assert "a%40b" not in sanitize_url(url)

    full = "rtsp://x:y@10.0.0.9/stream1"
    assert camera_url_from_env({"CAMERA_RTSP_URL": full, "CAMERA_IP": "1.2.3.4", "CAMERA_PASSWORD": "z"}) == full


def test_camera_not_configured():
    with pytest.raises(ValueError, match="CAMERA_IP"):
        camera_url_from_env({"CAMERA_IP": "192.168.1.50"})


def test_load_env_file_keeps_existing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('# camera\nTRACE_T_IP=192.168.1.50\nexport TRACE_T_PW="se cret"\nTRACE_T_SET=new\n', encoding="utf-8")
    monkeypatch.setenv("TRACE_T_SET", "old")
    monkeypatch.delenv("TRACE_T_IP", raising=False)
    monkeypatch.delenv("TRACE_T_PW", raising=False)

    assert load_env_file(env_file) == 2
    import os

    assert os.environ["TRACE_T_IP"] == "192.168.1.50"
    assert os.environ["TRACE_T_PW"] == "se cret"
    assert os.environ["TRACE_T_SET"] == "old"
    assert load_env_file(tmp_path / "missing.env") == 0


class _FakeStream:
    """Returns None for the first reads (still connecting), then frames with wall-clock timestamps."""

    camera_id, session_id = "CAM", "S"

    def __init__(self, timestamps, empty_reads=3, running=True):
        self.timestamps = list(timestamps)
        self.empty_reads = empty_reads
        self.is_running = running

    def read(self):
        if self.empty_reads:
            self.empty_reads -= 1
            return None
        if not self.timestamps:
            return None
        ts = self.timestamps.pop(0)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        return FramePacket(1, ts, "10:00:00.000", frame, frame.shape, self.camera_id, self.session_id)


def test_live_stream_waits_for_frames_and_rebases_time():
    live = LiveStream(_FakeStream([1_700_000_000.0, 1_700_000_002.5]), max_wait_sec=5, poll_sec=0)
    first, second = live.read(), live.read()
    assert first.timestamp == 0.0 and second.timestamp == 2.5
    assert second.timestamp_str == "10:00:00.000"
    assert live.camera_id == "CAM"


def test_live_stream_gives_up_when_camera_is_silent():
    live = LiveStream(_FakeStream([], empty_reads=10**6), max_wait_sec=0.05, poll_sec=0)
    assert live.read() is None
    assert "no frames" in live.ended_reason


def test_live_stream_stops_after_duration():
    now = time.time()
    live = LiveStream(_FakeStream([now, now + 1, now + 2], empty_reads=0), max_duration_sec=60, poll_sec=0)
    assert live.read() is not None
    assert live.read() is not None
    live._t0 -= 61  # pretend the session started a minute ago
    assert live.read() is None
    assert "--duration" in live.ended_reason
