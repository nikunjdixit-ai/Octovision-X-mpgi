from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.attendance.attendance_engine import STATUS_PRESENT, AttendanceEngine, format_seconds_hms
from src.core.types import FramePacket, TrackedPerson
from src.identity.identity_engine import (
    STATUS_CONFIRMED,
    STATUS_LOW_CONFIDENCE,
    STATUS_TENTATIVE,
    IdentityDecision,
)

logger = logging.getLogger("trace.verification")

VERIFIED = "VERIFIED"
REVIEW = "REVIEW"
POSSIBLY_PRESENT = "POSSIBLY_PRESENT"
ABSENT = "ABSENT"

REPORT_FIELDNAMES = [
    "student_id",
    "name",
    "roll_no",
    "attendance_status",
    "verification",
    "seen_from",
    "seen_to",
    "time_in_class_sec",
    "segments",
    "best_similarity",
    "confirmed_observations",
    "track_ids",
    "reasons",
    "evidence_dir",
]


@dataclass
class VerificationConfig:
    """
    Thresholds for the verification pass that runs alongside AttendanceEngine.
    """
    # A PRESENT student seen for less time than this is flagged for review
    min_time_in_class_sec: float = 5.0
    # ... or confirmed in fewer frames than this
    min_confirmed_observations: int = 3
    # An ABSENT student who was the top (unconfirmed) candidate this many times is POSSIBLY_PRESENT
    possible_presence_hits: int = 3
    # Observations further apart than this start a new timeline segment
    segment_gap_sec: float = 30.0
    # Seconds of video kept before/after the best frame in clip.mp4
    clip_padding_sec: float = 3.0
    snapshot_max_width: int = 1280

    def __post_init__(self) -> None:
        if self.min_time_in_class_sec < 0.0:
            raise ValueError("min_time_in_class_sec must be >= 0.0.")
        if self.min_confirmed_observations < 1:
            raise ValueError("min_confirmed_observations must be >= 1.")
        if self.possible_presence_hits < 1:
            raise ValueError("possible_presence_hits must be >= 1.")
        if self.segment_gap_sec <= 0.0:
            raise ValueError("segment_gap_sec must be > 0.0.")


@dataclass
class _Segment:
    start_sec: float
    end_sec: float
    track_ids: List[int] = field(default_factory=list)


@dataclass
class _Snapshot:
    similarity: float
    frame_id: int
    timestamp_sec: float
    track_id: int
    frame: np.ndarray
    crop: Optional[np.ndarray]


@dataclass
class _StudentEvidence:
    segments: List[_Segment] = field(default_factory=list)
    confirmed_observations: int = 0
    last_confirmed_frame_id: Optional[int] = None
    best: Optional[_Snapshot] = None
    candidate_hits: int = 0
    best_candidate: Optional[_Snapshot] = None
    conflict_tracks: set = field(default_factory=set)


@dataclass
class VerificationRecord:
    """Verification result for one enrolled student."""
    student_id: str
    name: str
    roll_no: str
    attendance_status: str
    verification: str
    seen_from: str = ""
    seen_to: str = ""
    time_in_class_sec: float = 0.0
    segments: List[Dict[str, Any]] = field(default_factory=list)
    best_similarity: float = 0.0
    confirmed_observations: int = 0
    track_ids: List[int] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    evidence_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "name": self.name,
            "roll_no": self.roll_no,
            "attendance_status": self.attendance_status,
            "verification": self.verification,
            "seen_from": self.seen_from,
            "seen_to": self.seen_to,
            "time_in_class_sec": self.time_in_class_sec,
            "segments": list(self.segments),
            "best_similarity": self.best_similarity,
            "confirmed_observations": self.confirmed_observations,
            "track_ids": list(self.track_ids),
            "reasons": list(self.reasons),
            "evidence_dir": self.evidence_dir,
        }

    def to_csv_row(self) -> Dict[str, Any]:
        row = self.to_dict()
        row["segments"] = "; ".join(f"{s['from']}-{s['to']}" for s in self.segments)
        row["track_ids"] = " ".join(str(t) for t in self.track_ids)
        row["reasons"] = "; ".join(self.reasons)
        return row


class AttendanceVerifier:
    """
    Verification and proof layer on top of AttendanceEngine.

    Runs as the PipelineController subscriber in place of `attendance_engine.process_frame`:
    every frame is passed to AttendanceEngine unchanged, and the returned IdentityDecisions
    are recorded to build, per student:
    - a timeline of when they were in class (segments split by gaps),
    - the best matching frame and crop as proof,
    - checks a person should look at before trusting the result.

    The official PRESENT/ABSENT decision stays with AttendanceEngine; this layer only adds
    a verification label next to it:
    - VERIFIED          PRESENT and no warning signs.
    - REVIEW            PRESENT, but seen only briefly, in few frames, the same person was
                        also matched to another student, or another person also claimed them.
    - POSSIBLY_PRESENT  ABSENT, but repeatedly the closest (unconfirmed) match for someone.
    - ABSENT            ABSENT with no supporting evidence.
    """

    def __init__(
        self,
        attendance_engine: AttendanceEngine,
        config: Optional[VerificationConfig] = None,
    ):
        self.attendance_engine = attendance_engine
        self.config = config if config is not None else VerificationConfig()
        self._evidence: Dict[str, _StudentEvidence] = {}
        # track_id -> student_ids it was confirmed as (a track should stay one student)
        self._track_students: Dict[int, set] = {}

    def reset(self) -> None:
        self._evidence.clear()
        self._track_students.clear()

    def _get(self, student_id: str) -> _StudentEvidence:
        if student_id not in self._evidence:
            self._evidence[student_id] = _StudentEvidence()
        return self._evidence[student_id]

    def process_frame(
        self,
        packet: FramePacket,
        tracked_people: List[TrackedPerson],
    ) -> List[IdentityDecision]:
        """
        PipelineController-compatible subscriber hook (`callback(packet, tracked_people)`).
        Attendance is updated first; a failure while recording evidence never affects it.
        """
        decisions = self.attendance_engine.process_frame(packet, tracked_people)
        try:
            self.record(packet, tracked_people, decisions)
        except Exception as exc:
            logger.warning("Verification recording skipped frame %s: %s", getattr(packet, "frame_id", "?"), exc)
        return decisions

    def record(
        self,
        packet: FramePacket,
        tracked_people: List[TrackedPerson],
        decisions: List[IdentityDecision],
    ) -> None:
        if packet is None or packet.frame is None or not decisions:
            return

        people_by_uid = {p.stream_track_uid: p for p in tracked_people}
        roster = {r.student_id for r in self.attendance_engine.get_attendance_records()}

        for dec in decisions:
            if dec is None:
                continue
            person = people_by_uid.get(dec.stream_track_uid)

            if dec.status == STATUS_CONFIRMED and dec.student_id in roster:
                self._record_confirmed(packet, person, dec)
            elif dec.conflict_demoted and dec.candidate_student_id in roster:
                # Another person was also matched to this student at the same moment
                self._get(dec.candidate_student_id).conflict_tracks.add(int(dec.track_id))
            elif (
                dec.status in (STATUS_TENTATIVE, STATUS_LOW_CONFIDENCE)
                and dec.inference_performed
                and dec.candidate_student_id in roster
            ):
                self._record_candidate(packet, person, dec)

    def _record_confirmed(
        self,
        packet: FramePacket,
        person: Optional[TrackedPerson],
        dec: IdentityDecision,
    ) -> None:
        ev = self._get(dec.student_id)
        track_id = int(dec.track_id)
        self._track_students.setdefault(track_id, set()).add(dec.student_id)

        if ev.last_confirmed_frame_id == packet.frame_id:
            return
        ev.last_confirmed_frame_id = packet.frame_id
        ev.confirmed_observations += 1

        ts = float(packet.timestamp)
        last = ev.segments[-1] if ev.segments else None
        if last is not None and 0.0 <= ts - last.end_sec <= self.config.segment_gap_sec:
            last.end_sec = max(last.end_sec, ts)
            if track_id not in last.track_ids:
                last.track_ids.append(track_id)
        else:
            ev.segments.append(_Segment(start_sec=ts, end_sec=ts, track_ids=[track_id]))

        if dec.inference_performed and (ev.best is None or dec.similarity > ev.best.similarity):
            ev.best = self._snapshot(packet, person, dec, label=dec.name)

    def _record_candidate(
        self,
        packet: FramePacket,
        person: Optional[TrackedPerson],
        dec: IdentityDecision,
    ) -> None:
        ev = self._get(dec.candidate_student_id)
        ev.candidate_hits += 1
        if ev.best_candidate is None or dec.similarity > ev.best_candidate.similarity:
            label = f"{dec.candidate_name or dec.candidate_student_id}?"
            ev.best_candidate = self._snapshot(packet, person, dec, label=label)

    def _snapshot(
        self,
        packet: FramePacket,
        person: Optional[TrackedPerson],
        dec: IdentityDecision,
        label: str,
    ) -> _Snapshot:
        frame = packet.frame.copy()
        crop = person.extract_crop(packet.frame) if person is not None else None
        if person is not None:
            x1, y1, x2, y2 = person.bbox.to_xyxy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 3)
            text = f"{label}  {dec.similarity:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(0, y1 - th - 10)
            cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + th + 10), (0, 200, 0), -1)
            cv2.putText(frame, text, (x1 + 4, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        h, w = frame.shape[:2]
        if w > self.config.snapshot_max_width:
            scale = self.config.snapshot_max_width / float(w)
            frame = cv2.resize(frame, (self.config.snapshot_max_width, int(h * scale)))

        return _Snapshot(
            similarity=float(dec.similarity),
            frame_id=int(packet.frame_id),
            timestamp_sec=float(packet.timestamp),
            track_id=int(dec.track_id),
            frame=frame,
            crop=crop,
        )

    def build_records(self) -> List[VerificationRecord]:
        """Combines AttendanceEngine records with the recorded evidence into verification results."""
        records: List[VerificationRecord] = []
        for att in self.attendance_engine.get_attendance_records():
            ev = self._evidence.get(att.student_id, _StudentEvidence())
            rec = VerificationRecord(
                student_id=att.student_id,
                name=att.name,
                roll_no=att.roll_no,
                attendance_status=att.status,
                verification=ABSENT,
                confirmed_observations=ev.confirmed_observations,
            )

            if ev.segments:
                rec.seen_from = format_seconds_hms(ev.segments[0].start_sec)
                rec.seen_to = format_seconds_hms(ev.segments[-1].end_sec)
                rec.time_in_class_sec = round(sum(s.end_sec - s.start_sec for s in ev.segments), 2)
                rec.segments = [
                    {
                        "from": format_seconds_hms(s.start_sec),
                        "to": format_seconds_hms(s.end_sec),
                        "track_ids": list(s.track_ids),
                    }
                    for s in ev.segments
                ]
                rec.track_ids = sorted({t for s in ev.segments for t in s.track_ids})
            if ev.best is not None:
                rec.best_similarity = round(ev.best.similarity, 4)

            if att.status == STATUS_PRESENT:
                rec.reasons = self._review_reasons(att.student_id, rec, ev)
                rec.verification = REVIEW if rec.reasons else VERIFIED
            elif ev.candidate_hits >= self.config.possible_presence_hits:
                rec.verification = POSSIBLY_PRESENT
                best = ev.best_candidate.similarity if ev.best_candidate else 0.0
                rec.reasons = [
                    f"closest match for someone {ev.candidate_hits} times (best {best:.2f}) but never confirmed"
                ]
            records.append(rec)
        return records

    def _review_reasons(self, student_id: str, rec: VerificationRecord, ev: _StudentEvidence) -> List[str]:
        reasons: List[str] = []
        if rec.time_in_class_sec < self.config.min_time_in_class_sec:
            reasons.append(f"seen only {rec.time_in_class_sec:.1f}s")
        if ev.confirmed_observations < self.config.min_confirmed_observations:
            reasons.append(f"confirmed in only {ev.confirmed_observations} frame(s)")
        for track_id in rec.track_ids:
            others = sorted(self._track_students.get(track_id, set()) - {student_id})
            if others:
                reasons.append(f"same person (track {track_id}) was also matched to {', '.join(others)}")
        if ev.conflict_tracks:
            tracks = ", ".join(str(t) for t in sorted(ev.conflict_tracks))
            reasons.append(f"another person (track {tracks}) was also matched to this student")
        return reasons

    def export(
        self,
        output_dir: Union[str, Path],
        photos_dir: Optional[Union[str, Path]] = None,
        video_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path, List[VerificationRecord]]:
        """
        Writes verification.csv / verification.json and one evidence folder per student:
        - proof.jpg   enrollment photo next to the matched person (needs `photos_dir`)
        - frame.jpg   the best matching frame, boxed and labelled
        - clip.mp4    a few seconds around that frame (needs `video_path`)
        For POSSIBLY_PRESENT students the files are prefixed `closest_` so a person can check them.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        profiles = {p.student_id: p for p in self.attendance_engine.registry.list_students()}
        records = self.build_records()

        for rec in records:
            ev = self._evidence.get(rec.student_id)
            if ev is None:
                continue
            if rec.verification in (VERIFIED, REVIEW):
                snap, prefix = ev.best, ""
            elif rec.verification == POSSIBLY_PRESENT:
                snap, prefix = ev.best_candidate, "closest_"
            else:
                continue
            if snap is None:
                continue

            student_dir = out / rec.student_id
            student_dir.mkdir(parents=True, exist_ok=True)
            rec.evidence_dir = str(student_dir)
            cv2.imwrite(str(student_dir / f"{prefix}frame.jpg"), snap.frame)

            photo = _enrollment_photo(profiles.get(rec.student_id), photos_dir)
            proof = _side_by_side(photo, snap.crop)
            if proof is not None:
                cv2.imwrite(str(student_dir / f"{prefix}proof.jpg"), proof)
            if video_path is not None:
                _write_clip(Path(video_path), snap.frame_id, self.config.clip_padding_sec, student_dir / f"{prefix}clip.mp4")

        csv_path = out / "verification.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDNAMES)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec.to_csv_row())

        counts = {s: sum(r.verification == s for r in records) for s in (VERIFIED, REVIEW, POSSIBLY_PRESENT, ABSENT)}
        json_path = out / "verification.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "session_id": self.attendance_engine.session_id,
                    "summary": counts,
                    "students": [r.to_dict() for r in records],
                },
                fh,
                indent=2,
            )

        logger.info("Exported verification report (%d students) to %s", len(records), out)
        return csv_path, json_path, records


def _enrollment_photo(profile: Any, photos_dir: Optional[Union[str, Path]]) -> Optional[np.ndarray]:
    """First enrollment photo of a student, looked up as <photos_dir>/<folder>/<filename>."""
    if profile is None or photos_dir is None or not profile.source_images:
        return None
    folder = profile.enrollment_metadata.get("folder_name", profile.student_id)
    path = Path(photos_dir) / folder / profile.source_images[0].get("filename", "")
    img = cv2.imread(str(path)) if path.is_file() else None
    return img


def _side_by_side(photo: Optional[np.ndarray], crop: Optional[np.ndarray], height: int = 320) -> Optional[np.ndarray]:
    tiles = []
    for img in (photo, crop):
        if img is None or img.size == 0:
            continue
        h, w = img.shape[:2]
        tiles.append(cv2.resize(img, (max(1, int(w * height / float(h))), height)))
    return np.hstack(tiles) if tiles else None


def _write_clip(video_path: Path, frame_id: int, padding_sec: float, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s for evidence clip", video_path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pad = int(round(padding_sec * fps))
    # Stream frame ids start at 1; OpenCV frame positions start at 0
    start = max(0, frame_id - 1 - pad)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for _ in range(2 * pad + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    cap.release()
