import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attendance.attendance_engine import AttendanceConfig, AttendanceEngine
from src.core.types import BoundingBox, FramePacket, TrackedPerson
from src.identity.enrollment import StudentProfile
from src.identity.identity_engine import (
    STATUS_CONFIRMED,
    STATUS_TENTATIVE,
    STATUS_UNKNOWN,
    IdentityDecision,
)
from src.visualization.live_view import GREEN, GREY, YELLOW, LiveAttendanceView

CAM, SESS = "CAM_CLASSROOM_01", "SESSION_TEST"


class _Registry:
    def list_students(self):
        return [
            StudentProfile(student_id="01_Asha_Rao", name="Asha Rao", roll_number="01"),
            StudentProfile(student_id="02_Ravi_Kumar", name="Ravi Kumar", roll_number="02"),
        ]


class _Identity:
    """Track 1 is Asha (confirmed), track 2 is still being checked, track 3 is not on the roster."""

    STATUSES = {1: STATUS_CONFIRMED, 2: STATUS_TENTATIVE, 3: STATUS_UNKNOWN}

    def __init__(self):
        self.last = {}

    def _decision(self, track_id, frame_id):
        status = self.STATUSES[track_id]
        return IdentityDecision(
            stream_track_uid=f"{CAM}_{SESS}_{track_id}",
            camera_id=CAM,
            session_id=SESS,
            track_id=track_id,
            status=status,
            student_id="01_Asha_Rao" if status == STATUS_CONFIRMED else None,
            name="Asha Rao" if status == STATUS_CONFIRMED else status,
            similarity=0.82,
            frame_id=frame_id,
        )

    def process_frame(self, packet, tracked_people):
        out = [self._decision(p.track_id, packet.frame_id) for p in tracked_people]
        self.last = {d.stream_track_uid: d for d in out}
        return out

    def get_track_state(self, uid):
        return self.last.get(uid)

    def reset(self):
        self.last = {}


def _people(fid):
    return [
        TrackedPerson(camera_id=CAM, session_id=SESS, track_id=t, bbox=BoundingBox(20 + 120 * i, 40, 100 + 120 * i, 300), frame_id=fid)
        for i, t in enumerate((1, 2, 3))
    ]


def _packet(fid, frame):
    return FramePacket(fid, fid / 10.0, "", frame, frame.shape, CAM, SESS)


def _engine():
    identity = _Identity()
    engine = AttendanceEngine(registry=_Registry(), identity_engine=identity, config=AttendanceConfig(), session_id=SESS, camera_id=CAM)
    return engine, identity


def test_labels_follow_identity_status():
    engine, identity = _engine()
    view = LiveAttendanceView(engine, identity, show=False)
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    people = _people(1)
    engine.process_frame(_packet(1, frame), people)

    labels = [view._person_label(p) for p in people]
    assert labels == [("Asha Rao", GREEN), ("checking...", YELLOW), ("Unknown", GREY)]

    unseen = TrackedPerson(camera_id=CAM, session_id=SESS, track_id=9, bbox=BoundingBox(0, 0, 10, 30), frame_id=1)
    assert view._person_label(unseen) == ("checking...", YELLOW)


def test_render_adds_panel_and_scales_wide_frames():
    engine, identity = _engine()
    view = LiveAttendanceView(engine, identity, show=False, max_video_width=640, panel_width=300)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    engine.process_frame(_packet(1, frame), _people(1))

    canvas = view.render(_packet(1, frame), _people(1))
    assert canvas.shape == (360, 640 + 300, 3)
    # the input frame is not drawn on
    assert not frame.any()


def test_saves_live_view_video(tmp_path):
    engine, identity = _engine()
    out = tmp_path / "live" / "live_view.mp4"
    view = LiveAttendanceView(engine, identity, show=False, save_path=out, fps=10)
    frame = np.full((240, 320, 3), 60, dtype=np.uint8)
    for fid in range(1, 21):
        packet = _packet(fid, frame)
        people = _people(fid)
        engine.process_frame(packet, people)
        view(packet, people)

    assert view.close() == out
    cap = cv2.VideoCapture(str(out))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frames == 20


def test_close_without_frames_returns_none(tmp_path):
    engine, identity = _engine()
    view = LiveAttendanceView(engine, identity, show=False, save_path=tmp_path / "never.mp4")
    assert view.close() is None


def test_panel_lists_present_students():
    engine, identity = _engine()
    view = LiveAttendanceView(engine, identity, show=False)
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    engine.process_frame(_packet(1, frame), _people(1))

    statuses = {r.student_id: r.status for r in engine.get_attendance_records()}
    assert statuses == {"01_Asha_Rao": "PRESENT", "02_Ravi_Kumar": "ABSENT"}
    panel = view._panel(360, _packet(1, frame))
    assert panel.shape == (360, view.panel_width, 3)
