import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attendance.attendance_engine import AttendanceConfig, AttendanceEngine
from src.core.types import BoundingBox, FramePacket, TrackedPerson
from src.identity.enrollment import StudentProfile
from src.identity.identity_engine import (
    STATUS_CONFIRMED,
    STATUS_LOW_CONFIDENCE,
    STATUS_TENTATIVE,
    STATUS_UNKNOWN,
    IdentityDecision,
)
from src.verification import (
    ABSENT,
    POSSIBLY_PRESENT,
    REVIEW,
    VERIFIED,
    AttendanceVerifier,
    VerificationConfig,
)

FPS = 10.0
CAM, SESS = "CAM_CLASSROOM_01", "SESSION_TEST"


class _Registry:
    def __init__(self, profiles: List[StudentProfile]):
        self._profiles = profiles

    def list_students(self) -> List[StudentProfile]:
        return list(self._profiles)


class _ScriptedIdentity:
    """Returns pre-scripted IdentityDecisions per frame instead of running the models."""

    def __init__(self, script: Dict[int, List[dict]]):
        self.script = script

    def process_frame(self, packet, tracked_people):
        out = []
        for d in self.script.get(packet.frame_id, []):
            out.append(
                IdentityDecision(
                    stream_track_uid=f"{CAM}_{SESS}_{d['track']}",
                    camera_id=CAM,
                    session_id=SESS,
                    track_id=d["track"],
                    status=d["status"],
                    student_id=d.get("sid") if d["status"] == STATUS_CONFIRMED else None,
                    name=d.get("name", d.get("sid", STATUS_UNKNOWN)),
                    similarity=d.get("sim", 0.8),
                    candidate_student_id=d.get("cand", d.get("sid")),
                    candidate_name=d.get("cand"),
                    frame_id=packet.frame_id,
                    inference_performed=d.get("inf", True),
                    conflict_demoted=d.get("demoted", False),
                )
            )
        return out

    def reset(self):
        pass


def _profile(sid: str, name: str, roll: str) -> StudentProfile:
    return StudentProfile(
        student_id=sid,
        name=name,
        roll_number=roll,
        source_images=[{"filename": "01.jpg"}],
        enrollment_metadata={"folder_name": sid},
    )


def _run(script: Dict[int, List[dict]], n_frames: int, config: VerificationConfig = None):
    registry = _Registry([
        _profile("01_Asha_Rao", "Asha Rao", "01"),
        _profile("02_Ravi_Kumar", "Ravi Kumar", "02"),
        _profile("03_Meera_Shah", "Meera Shah", "03"),
        _profile("04_Kabir_Jain", "Kabir Jain", "04"),
    ])
    engine = AttendanceEngine(
        registry=registry,
        identity_engine=_ScriptedIdentity(script),
        config=AttendanceConfig(),
        session_id=SESS,
        camera_id=CAM,
    )
    verifier = AttendanceVerifier(engine, config or VerificationConfig(min_time_in_class_sec=2.0))
    frame = np.full((120, 160, 3), 90, dtype=np.uint8)
    for fid in range(1, n_frames + 1):
        packet = FramePacket(
            frame_id=fid,
            timestamp=fid / FPS,
            timestamp_str="",
            frame=frame,
            shape=frame.shape,
            camera_id=CAM,
            session_id=SESS,
        )
        people = [
            TrackedPerson(camera_id=CAM, session_id=SESS, track_id=d["track"],
                          bbox=BoundingBox(10, 10, 60, 110), frame_id=fid)
            for d in script.get(fid, [])
        ]
        verifier.process_frame(packet, people)
    return engine, verifier


def _by_id(records):
    return {r.student_id: r for r in records}


def test_statuses_cover_verified_review_possible_and_absent():
    script = {}
    for fid in range(1, 51):
        script.setdefault(fid, []).append({"track": 1, "status": STATUS_CONFIRMED, "sid": "01_Asha_Rao"})
    # Ravi confirmed in a single frame only -> PRESENT but REVIEW
    script.setdefault(5, []).append({"track": 2, "status": STATUS_CONFIRMED, "sid": "02_Ravi_Kumar"})
    # Meera is repeatedly the closest unconfirmed match -> POSSIBLY_PRESENT
    for fid in (10, 20, 30):
        script.setdefault(fid, []).append(
            {"track": 3, "status": STATUS_TENTATIVE, "cand": "03_Meera_Shah", "sim": 0.68}
        )

    engine, verifier = _run(script, n_frames=50)
    recs = _by_id(verifier.build_records())

    assert recs["01_Asha_Rao"].verification == VERIFIED
    assert recs["01_Asha_Rao"].attendance_status == "PRESENT"
    assert recs["02_Ravi_Kumar"].attendance_status == "PRESENT"
    assert recs["02_Ravi_Kumar"].verification == REVIEW
    assert any("only" in r for r in recs["02_Ravi_Kumar"].reasons)
    assert recs["03_Meera_Shah"].attendance_status == "ABSENT"
    assert recs["03_Meera_Shah"].verification == POSSIBLY_PRESENT
    assert recs["04_Kabir_Jain"].verification == ABSENT


def test_timeline_splits_on_gap():
    script = {}
    for fid in list(range(1, 21)) + list(range(401, 421)):  # 2 s, then a 38 s gap, then 2 s
        script[fid] = [{"track": 7 if fid < 400 else 9, "status": STATUS_CONFIRMED, "sid": "01_Asha_Rao"}]

    _, verifier = _run(script, n_frames=420)
    rec = _by_id(verifier.build_records())["01_Asha_Rao"]

    assert len(rec.segments) == 2
    assert rec.segments[0]["track_ids"] == [7]
    assert rec.segments[1]["track_ids"] == [9]
    assert rec.track_ids == [7, 9]
    assert rec.seen_from == "00:00:00" and rec.seen_to == "00:00:42"


def test_one_person_matched_to_two_students_is_flagged():
    script = {}
    for fid in range(1, 31):
        sid = "01_Asha_Rao" if fid <= 15 else "02_Ravi_Kumar"
        script[fid] = [{"track": 4, "status": STATUS_CONFIRMED, "sid": sid}]

    _, verifier = _run(script, n_frames=30, config=VerificationConfig(min_time_in_class_sec=0.5))
    recs = _by_id(verifier.build_records())

    assert recs["01_Asha_Rao"].verification == REVIEW
    assert recs["02_Ravi_Kumar"].verification == REVIEW
    assert any("02_Ravi_Kumar" in r for r in recs["01_Asha_Rao"].reasons)


def test_conflict_demotion_is_flagged():
    script = {}
    for fid in range(1, 41):
        script[fid] = [{"track": 1, "status": STATUS_CONFIRMED, "sid": "01_Asha_Rao"}]
    script[12].append({"track": 5, "status": STATUS_LOW_CONFIDENCE, "cand": "01_Asha_Rao", "demoted": True})

    _, verifier = _run(script, n_frames=40)
    rec = _by_id(verifier.build_records())["01_Asha_Rao"]

    assert rec.verification == REVIEW
    assert any("track 5" in r for r in rec.reasons)


def test_cached_decisions_do_not_count_as_candidate_hits():
    script = {fid: [{"track": 3, "status": STATUS_TENTATIVE, "cand": "03_Meera_Shah", "inf": False}]
              for fid in range(1, 20)}

    _, verifier = _run(script, n_frames=19)
    assert _by_id(verifier.build_records())["03_Meera_Shah"].verification == ABSENT


def test_attendance_matches_engine_without_verifier():
    script = {fid: [{"track": 1, "status": STATUS_CONFIRMED, "sid": "01_Asha_Rao"}] for fid in range(1, 30)}
    engine, _ = _run(script, n_frames=29)

    plain = AttendanceEngine(
        registry=engine.registry, identity_engine=_ScriptedIdentity(script),
        config=AttendanceConfig(), session_id=SESS, camera_id=CAM,
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for fid in range(1, 30):
        plain.process_frame(FramePacket(fid, fid / FPS, "", frame, frame.shape, CAM, SESS), [])

    assert [r.to_dict() for r in engine.get_attendance_records()] == [r.to_dict() for r in plain.get_attendance_records()]


def test_recording_failure_never_breaks_attendance(monkeypatch):
    script = {fid: [{"track": 1, "status": STATUS_CONFIRMED, "sid": "01_Asha_Rao"}] for fid in range(1, 10)}

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(AttendanceVerifier, "record", boom)
    engine, _ = _run(script, n_frames=9)
    assert _by_id(engine.get_attendance_records())["01_Asha_Rao"].status == "PRESENT"


def test_export_writes_reports_and_evidence(tmp_path):
    photos = tmp_path / "students" / "01_Asha_Rao"
    photos.mkdir(parents=True)
    cv2.imwrite(str(photos / "01.jpg"), np.full((200, 100, 3), 200, dtype=np.uint8))

    video = tmp_path / "class.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 120))
    for _ in range(60):
        writer.write(np.full((120, 160, 3), 90, dtype=np.uint8))
    writer.release()

    script = {fid: [{"track": 1, "status": STATUS_CONFIRMED, "sid": "01_Asha_Rao"}] for fid in range(1, 50)}
    for fid in (10, 20, 30):
        script[fid].append({"track": 3, "status": STATUS_TENTATIVE, "cand": "03_Meera_Shah", "sim": 0.66})

    _, verifier = _run(script, n_frames=49)
    csv_path, json_path, _ = verifier.export(tmp_path / "verification", photos_dir=tmp_path / "students", video_path=video)

    asha = tmp_path / "verification" / "01_Asha_Rao"
    assert (asha / "frame.jpg").exists()
    assert (asha / "proof.jpg").exists()
    assert (asha / "clip.mp4").exists()
    assert (tmp_path / "verification" / "03_Meera_Shah" / "closest_frame.jpg").exists()
    assert not (tmp_path / "verification" / "04_Kabir_Jain").exists()

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["summary"] == {VERIFIED: 1, REVIEW: 0, POSSIBLY_PRESENT: 1, ABSENT: 2}
    assert csv_path.read_text(encoding="utf-8").startswith("student_id,name,roll_no")


def test_config_validation():
    with pytest.raises(ValueError):
        VerificationConfig(segment_gap_sec=0)
    with pytest.raises(ValueError):
        VerificationConfig(possible_presence_hits=0)
