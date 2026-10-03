from __future__ import annotations

import csv
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.types import FramePacket, TrackedPerson
from src.identity.enrollment import StudentProfile, StudentRegistry, sanitize_roll_number
from src.identity.identity_engine import (
    STATUS_CONFIRMED,
    IdentityDecision,
    IdentityEngine,
)

logger = logging.getLogger("trace.attendance.engine")

STATUS_PRESENT = "PRESENT"
STATUS_ABSENT = "ABSENT"

CSV_FIELDNAMES = [
    "student_id",
    "name",
    "roll_no",
    "department",
    "section",
    "first_seen",
    "last_seen",
    "duration_seconds",
    "status",
    "session_id",
    "camera_id",
    "observation_count",
]


def _natural_sort_key(value: str) -> List[Union[int, str]]:
    """Deterministic natural sort key so '01', '02', ..., '09', '10' sort numerically."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def format_seconds_hms(seconds: float) -> str:
    """Formats elapsed seconds as HH:MM:SS."""
    total_sec = max(0, int(round(seconds)))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


@dataclass
class AttendanceConfig:
    """
    Centralized configuration for temporal attendance tracking and Present/Absent reconciliation.
    """
    min_confirmed_observations: int = 1
    min_presence_duration_sec: float = 0.0
    max_observation_gap_sec: float = 30.0
    default_fps: float = 30.0

    def __post_init__(self) -> None:
        if self.min_confirmed_observations < 1:
            raise ValueError("min_confirmed_observations must be >= 1.")
        if self.min_presence_duration_sec < 0.0:
            raise ValueError("min_presence_duration_sec must be >= 0.0.")
        if self.max_observation_gap_sec <= 0.0:
            raise ValueError("max_observation_gap_sec must be > 0.0.")
        if self.default_fps <= 0.0:
            raise ValueError("default_fps must be > 0.0.")


@dataclass
class AttendanceRecord:
    """
    Authoritative attendance record for a single enrolled student within a session.
    """
    student_id: str
    name: str
    roll_number: str
    department: str
    section: str
    first_seen: str = ""
    last_seen: str = ""
    duration_seconds: float = 0.0
    status: str = STATUS_ABSENT
    session_id: str = "SESSION_01"
    camera_id: str = "CAM_01"
    observation_count: int = 0
    confidence: float = 0.0
    track_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.student_id = str(self.student_id).strip()
        self.name = str(self.name).strip()
        self.roll_number = sanitize_roll_number(self.roll_number)
        self.department = str(self.department).strip()
        self.section = str(self.section).strip()

    @property
    def roll_no(self) -> str:
        return str(self.roll_number)

    def formatted_duration(self) -> Union[int, float]:
        if self.status == STATUS_ABSENT or self.duration_seconds == 0.0:
            return 0
        rounded = round(float(self.duration_seconds), 2)
        return int(rounded) if rounded.is_integer() else rounded

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "student_id": str(self.student_id),
            "name": str(self.name),
            "roll_no": str(self.roll_number),
            "department": str(self.department),
            "section": str(self.section),
            "first_seen": self.first_seen if self.status == STATUS_PRESENT else "",
            "last_seen": self.last_seen if self.status == STATUS_PRESENT else "",
            "duration_seconds": self.formatted_duration(),
            "status": self.status,
            "session_id": str(self.session_id),
            "camera_id": str(self.camera_id),
            "observation_count": self.observation_count,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": str(self.student_id),
            "name": str(self.name),
            "roll_no": str(self.roll_number),
            "roll_number": str(self.roll_number),
            "department": str(self.department),
            "section": str(self.section),
            "status": self.status,
            "first_seen": self.first_seen if self.status == STATUS_PRESENT else "",
            "last_seen": self.last_seen if self.status == STATUS_PRESENT else "",
            "duration_seconds": self.formatted_duration(),
            "observation_count": self.observation_count,
            "confidence": round(float(self.confidence), 4),
            "track_ids": list(self.track_ids),
            "session_id": str(self.session_id),
            "camera_id": str(self.camera_id),
        }


@dataclass
class _StudentTemporalState:
    """Internal per-student session tracking state."""
    first_timestamp_sec: Optional[float] = None
    last_timestamp_sec: Optional[float] = None
    accumulated_duration_sec: float = 0.0
    last_frame_id: Optional[int] = None


class AttendanceEngine:
    """
    Session-based classroom attendance engine built on top of StudentRegistry and IdentityEngine.

    Responsibilities:
    - Initializes all enrolled students in StudentRegistry as ABSENT.
    - Consumes IdentityDecision outputs (either directly or as a PipelineController subscriber).
    - Strictly ignores UNKNOWN, LOW_CONFIDENCE, and TENTATIVE identity states.
    - Deduplicates observations across multiple ByteTrack track_ids for the same student_id.
    - Tracks first_seen, last_seen, and presence duration_seconds across non-occluded intervals.
    - Reconciles Present/Absent status and generates deterministic CSV and JSON session reports.
    """

    def __init__(
        self,
        registry: StudentRegistry,
        identity_engine: Optional[IdentityEngine] = None,
        config: Optional[AttendanceConfig] = None,
        session_id: str = "SESSION_01",
        camera_id: str = "CAM_01",
    ):
        self.registry = registry
        self.config = config if config is not None else AttendanceConfig()
        self.session_id = str(session_id)
        self.camera_id = str(camera_id)
        self.identity_engine = (
            identity_engine
            if identity_engine is not None
            else IdentityEngine(registry=self.registry)
        )

        self._records: Dict[str, AttendanceRecord] = {}
        self._temporal_states: Dict[str, _StudentTemporalState] = {}
        self.frames_evaluated: int = 0
        self._initialize_roster()

    def _initialize_roster(self) -> None:
        """Initializes every enrolled student in the registry as ABSENT."""
        self._records.clear()
        self._temporal_states.clear()

        for profile in self.registry.list_students():
            sid = profile.student_id
            self._records[sid] = AttendanceRecord(
                student_id=sid,
                name=profile.name,
                roll_number=profile.roll_number,
                department=profile.department,
                section=profile.section,
                first_seen="",
                last_seen="",
                duration_seconds=0.0,
                status=STATUS_ABSENT,
                session_id=self.session_id,
                camera_id=self.camera_id,
                observation_count=0,
                confidence=0.0,
                track_ids=[],
            )
            self._temporal_states[sid] = _StudentTemporalState()

    def reset_session(self, session_id: Optional[str] = None) -> None:
        """Resets attendance state for a new classroom session."""
        if session_id is not None:
            self.session_id = str(session_id)
        self.frames_evaluated = 0
        self.identity_engine.reset()
        self._initialize_roster()

    def process_frame(
        self,
        packet: FramePacket,
        tracked_people: List[TrackedPerson],
    ) -> List[IdentityDecision]:
        """
        PipelineController-compatible subscriber hook (`callback(packet, tracked_people)`).
        Delegates person crop identity evaluation to IdentityEngine and updates session attendance.
        """
        if packet is None or packet.frame is None:
            return []

        self.frames_evaluated += 1
        decisions = self.identity_engine.process_frame(packet, tracked_people)
        self.update_from_decisions(
            decisions=decisions,
            timestamp_sec=packet.timestamp,
            timestamp_str=packet.timestamp_str,
            frame_id=packet.frame_id,
            camera_id=packet.camera_id,
            session_id=packet.session_id,
        )
        return decisions

    def update_from_decisions(
        self,
        decisions: List[IdentityDecision],
        timestamp_sec: Optional[float] = None,
        timestamp_str: Optional[str] = None,
        frame_id: Optional[int] = None,
        camera_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Updates student attendance records from a batch of IdentityDecision objects.
        Only CONFIRMED decisions for enrolled students are processed.
        Deduplicates multiple decisions claiming the same student in the same frame.
        """
        if not decisions:
            return

        # Group CONFIRMED decisions by student_id to prevent duplicate updates in the same frame
        confirmed_by_student: Dict[str, IdentityDecision] = {}
        for dec in decisions:
            if dec is None:
                continue
            if dec.status != STATUS_CONFIRMED or not dec.student_id:
                # Strictly ignore UNKNOWN, LOW_CONFIDENCE, and TENTATIVE
                continue
            if dec.student_id not in self._records:
                continue

            existing = confirmed_by_student.get(dec.student_id)
            if existing is None or dec.similarity > existing.similarity:
                confirmed_by_student[dec.student_id] = dec

            # Still record track_id association even if another track in the same frame had higher similarity
            rec = self._records[dec.student_id]
            if dec.track_id is not None and int(dec.track_id) not in rec.track_ids:
                rec.track_ids.append(int(dec.track_id))

        for sid, dec in confirmed_by_student.items():
            eff_frame_id = frame_id if frame_id is not None else dec.frame_id
            eff_ts_sec = (
                float(timestamp_sec)
                if timestamp_sec is not None
                else (float(eff_frame_id) / self.config.default_fps)
            )
            eff_ts_str = (
                timestamp_str
                if timestamp_str
                else format_seconds_hms(eff_ts_sec)
            )
            self._apply_confirmed_observation(
                student_id=sid,
                timestamp_sec=eff_ts_sec,
                timestamp_str=eff_ts_str,
                frame_id=eff_frame_id,
                track_id=dec.track_id,
                confidence=dec.similarity,
                camera_id=camera_id or dec.camera_id,
                session_id=session_id or dec.session_id,
            )

    def _apply_confirmed_observation(
        self,
        student_id: str,
        timestamp_sec: float,
        timestamp_str: str,
        frame_id: Optional[int] = None,
        track_id: Optional[int] = None,
        confidence: float = 0.0,
        camera_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        record = self._records[student_id]
        state = self._temporal_states[student_id]

        if camera_id:
            record.camera_id = str(camera_id)
        if session_id:
            record.session_id = str(session_id)
        if track_id is not None and int(track_id) not in record.track_ids:
            record.track_ids.append(int(track_id))

        # Guard against duplicate update on the exact same frame_id
        if frame_id is not None and state.last_frame_id is not None and frame_id == state.last_frame_id:
            record.confidence = round(max(record.confidence, float(confidence)), 4)
            return

        state.last_frame_id = frame_id
        record.observation_count += 1
        record.confidence = round(max(record.confidence, float(confidence)), 4)

        if state.first_timestamp_sec is None or state.last_timestamp_sec is None:
            state.first_timestamp_sec = timestamp_sec
            state.last_timestamp_sec = timestamp_sec
            record.first_seen = timestamp_str
            record.last_seen = timestamp_str
        else:
            delta = timestamp_sec - state.last_timestamp_sec
            if 0.0 < delta <= self.config.max_observation_gap_sec:
                state.accumulated_duration_sec += delta
            if timestamp_sec >= state.last_timestamp_sec:
                state.last_timestamp_sec = timestamp_sec
                record.last_seen = timestamp_str

        record.duration_seconds = round(state.accumulated_duration_sec, 2)

        # Reconcile PRESENT status when thresholds are satisfied
        if (
            record.observation_count >= self.config.min_confirmed_observations
            and record.duration_seconds >= self.config.min_presence_duration_sec
        ):
            record.status = STATUS_PRESENT

    def get_attendance_records(self) -> List[AttendanceRecord]:
        """
        Returns deterministically sorted AttendanceRecord objects for ALL enrolled students
        (both PRESENT and ABSENT).
        """
        records = list(self._records.values())
        records.sort(key=lambda r: (_natural_sort_key(r.student_id), _natural_sort_key(r.roll_number)))
        return records

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Generates the structured session attendance summary dictionary matching the JSON report schema.
        """
        records = self.get_attendance_records()
        total_students = len(records)
        present_count = sum(1 for r in records if r.status == STATUS_PRESENT)
        absent_count = total_students - present_count
        pct = round((present_count / total_students) * 100.0, 2) if total_students > 0 else 0.0

        return {
            "session_id": self.session_id,
            "camera_id": self.camera_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_students": total_students,
            "present": present_count,
            "absent": absent_count,
            "attendance_percentage": pct,
            "students": [r.to_dict() for r in records],
        }

    def export_csv(self, output_path: Union[str, Path]) -> Path:
        """
        Writes the complete attendance report CSV containing all enrolled students
        (both PRESENT and ABSENT).
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        records = self.get_attendance_records()

        with open(out, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec.to_csv_row())

        logger.info("Exported CSV attendance report (%d students) to %s", len(records), out)
        return out

    def export_json(self, output_path: Union[str, Path]) -> Path:
        """
        Writes the structured JSON session attendance report.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_session_summary()

        with open(out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        logger.info("Exported JSON attendance session report to %s", out)
        return out

    def export_reports(
        self,
        output_dir: Union[str, Path],
        filename_prefix: str = "attendance",
        timestamp_tag: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Exports both `attendance_<timestamp>.csv` and `attendance_<timestamp>.json` into `output_dir`.
        Returns (csv_path, json_path).
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = timestamp_tag or time.strftime("%Y%m%d_%H%M%S")
        csv_path = self.export_csv(out_dir / f"{filename_prefix}_{tag}.csv")
        json_path = self.export_json(out_dir / f"{filename_prefix}_{tag}.json")
        return csv_path, json_path
