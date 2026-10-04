from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attendance.attendance_engine import AttendanceConfig, AttendanceEngine
from src.core.pipeline_controller import PipelineController
from src.core.types import StreamConfig
from src.identity.enrollment import EnrollmentError, StudentRegistry
from src.identity.identity_engine import IdentityEngine
from src.reid.reid_model import ReIDModel
from src.verification.attendance_verifier import AttendanceVerifier, VerificationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TRACE Phase 7 — Classroom MP4 Attendance Engine & CSV/JSON Report Generator"
    )
    parser.add_argument(
        "--video",
        "--source",
        dest="video",
        type=str,
        required=True,
        help="Path to input classroom MP4 video (e.g., data/raw/videos/classroom/classroom_test.mp4)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="data/enrollment/registry",
        help="Path to StudentRegistry directory containing students_registry.json and embeddings/",
    )
    parser.add_argument(
        "--students-dir",
        type=str,
        default=None,
        help="Optional student folders directory to enroll if registry does not yet exist",
    )
    parser.add_argument(
        "--students-zip",
        type=str,
        default=None,
        help="Optional student ZIP archive to enroll if registry does not yet exist",
    )
    parser.add_argument(
        "--metadata-csv",
        "--metadata",
        dest="metadata_csv",
        type=str,
        default=None,
        help="Optional CSV (student_id,name,roll_no,department,section) to update student metadata",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/attendance",
        help="Output directory or .csv file path for generated CSV and JSON attendance reports",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="CAM_CLASSROOM_01",
        help="Camera identifier for session tracking",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session identifier (defaults to SESSION_<timestamp>)",
    )
    default_yolo = "models/yolov8m.pt" if (PROJECT_ROOT / "models/yolov8m.pt").exists() else "yolov8n.pt"
    parser.add_argument(
        "--model",
        type=str,
        default=default_yolo,
        help=f"YOLO person detection weights path (default: {default_yolo})",
    )
    parser.add_argument(
        "--confirm-threshold",
        type=float,
        default=0.70,
        help="Cosine similarity threshold for IdentityEngine candidate confirmation (default: 0.70)",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.55,
        help="Cosine similarity floor for LOW_CONFIDENCE before UNKNOWN rejection",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.025,
        help="Minimum difference between top-1 and top-2 cosine similarities to qualify as confirmation vote (default: 0.025)",
    )
    parser.add_argument(
        "--min-confirmations",
        type=int,
        default=3,
        help="Minimum qualifying frame observations in IdentityEngine to confirm a track",
    )
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=2,
        help="Frame interval between OSNet extractions for unconfirmed tracks",
    )
    parser.add_argument(
        "--reverify-interval",
        type=int,
        default=15,
        help="Frame interval between OSNet re-verifications for already-confirmed tracks",
    )
    parser.add_argument(
        "--min-confirmed-observations",
        type=int,
        default=1,
        help="Minimum confirmed observations in AttendanceEngine to mark PRESENT",
    )
    parser.add_argument(
        "--min-presence-duration",
        type=float,
        default=0.0,
        help="Minimum presence duration in seconds required to mark PRESENT",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=30.0,
        help="Maximum allowed gap in seconds between observations for continuous dwell accumulation",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process (processes full video by default)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Also write a verification report with per-student proof (frame, photo comparison, clip)",
    )
    parser.add_argument(
        "--photos-dir",
        type=str,
        default=None,
        help="Student photo folders used for proof.jpg (defaults to --students-dir)",
    )
    parser.add_argument(
        "--no-clips",
        action="store_true",
        help="With --verify: skip clip.mp4 evidence (faster)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists() or not video_path.is_file():
        print(f"ERROR: Classroom video file not found: {video_path}")
        return 1

    registry_path = Path(args.registry)
    reid_model = ReIDModel()
    registry = StudentRegistry(reid_model=reid_model)

    try:
        if (registry_path / "students_registry.json").exists():
            registry.load(registry_path)
        elif args.students_zip:
            registry.registry_dir = registry_path
            registry.enroll_from_zip(
                zip_path=args.students_zip,
                persist=True,
                metadata_csv=args.metadata_csv,
            )
        elif args.students_dir:
            registry.registry_dir = registry_path
            registry.enroll_from_directory(
                students_dir=args.students_dir,
                persist=True,
                metadata_csv=args.metadata_csv,
            )
        else:
            print(
                f"ERROR: Student registry manifest not found at '{registry_path / 'students_registry.json'}'. "
                "Provide --registry, --students-dir, or --students-zip."
            )
            return 1

        if args.metadata_csv:
            registry.apply_metadata_csv(args.metadata_csv, persist=True)
    except (EnrollmentError, Exception) as exc:
        print(f"ERROR loading/preparing StudentRegistry: {exc}")
        return 1

    students = registry.list_students()
    if not students:
        print("ERROR: StudentRegistry contains 0 enrolled students.")
        return 1

    # Pre-flight verification of StudentRegistry integrity
    roll_numbers = [s.roll_number for s in students]
    if len(set(roll_numbers)) != len(roll_numbers):
        print("ERROR: Duplicate roll numbers detected in StudentRegistry.")
        return 1

    session_id = args.session_id or f"SESSION_{time.strftime('%Y%m%d_%H%M%S')}"

    print("=================================================================")
    print("             TRACE CLASSROOM ATTENDANCE ENGINE                   ")
    print("=================================================================")
    print(f"Video Source       : {video_path}")
    print(f"Student Registry   : {registry_path} ({len(students)} students enrolled)")
    print(f"Total Ref Samples  : {sum(s.num_samples for s in students)} reference images")
    print(f"Unique Roll Nos    : {len(set(roll_numbers))} verified unique")
    print(f"Camera / Session   : {args.camera_id} / {session_id}")
    print("-----------------------------------------------------------------")

    identity_engine = IdentityEngine(
        registry=registry,
        reid_model=reid_model,
        confirm_threshold=args.confirm_threshold,
        low_confidence_threshold=args.low_confidence_threshold,
        min_confirmations=args.min_confirmations,
        inference_interval_frames=args.inference_interval,
        reverify_interval_frames=args.reverify_interval,
        min_margin=args.min_margin,
    )

    attendance_config = AttendanceConfig(
        min_confirmed_observations=args.min_confirmed_observations,
        min_presence_duration_sec=args.min_presence_duration,
        max_observation_gap_sec=args.max_gap,
    )

    attendance_engine = AttendanceEngine(
        registry=registry,
        identity_engine=identity_engine,
        config=attendance_config,
        session_id=session_id,
        camera_id=args.camera_id,
    )

    stream_config = StreamConfig(
        source_type="file",
        source_path=str(video_path),
        camera_id=args.camera_id,
        session_id=session_id,
    )

    controller = PipelineController(
        config=stream_config,
        model_path=args.model,
    )
    verifier = AttendanceVerifier(attendance_engine, VerificationConfig()) if args.verify else None
    controller.subscribe(verifier.process_frame if verifier else attendance_engine.process_frame)

    t0 = time.perf_counter()
    frames_processed = controller.run(max_frames=args.max_frames)
    elapsed_sec = time.perf_counter() - t0
    fps = frames_processed / elapsed_sec if elapsed_sec > 0 else 0.0

    # Determine output CSV and JSON paths
    output_target = Path(args.output)
    if output_target.suffix.lower() == ".csv":
        csv_path = attendance_engine.export_csv(output_target)
        json_path = attendance_engine.export_json(output_target.with_suffix(".json"))
    elif output_target.name.lower() == "latest":
        csv_path = attendance_engine.export_csv(output_target / "attendance.csv")
        json_path = attendance_engine.export_json(output_target / "attendance.json")
    else:
        csv_path, json_path = attendance_engine.export_reports(output_target)
        # Also write canonical attendance.csv and attendance.json for deterministic access
        attendance_engine.export_csv(output_target / "attendance.csv")
        attendance_engine.export_json(output_target / "attendance.json")

    summary = attendance_engine.get_session_summary()
    id_stats = identity_engine.get_performance_stats()

    print("\n--- ATTENDANCE REPORT SUMMARY ---")
    print(f"{'ID':<18} | {'Name':<20} | {'Roll No.':<15} | {'Dept':<8} | {'Sec':<3} | {'First':<8} | {'Last':<8} | {'Dur(s)':<6} | {'Status'}")
    print("-" * 108)
    for rec in attendance_engine.get_attendance_records():
        print(
            f"{rec.student_id:<18} | {rec.name:<20} | {rec.roll_number:<15} | "
            f"{rec.department:<8} | {rec.section:<3} | {rec.first_seen:<8} | "
            f"{rec.last_seen:<8} | {str(rec.formatted_duration()):<6} | {rec.status}"
        )
    print("-" * 108)
    print(
        f"Total Enrolled: {summary['total_students']} | "
        f"Present: {summary['present']} | "
        f"Absent: {summary['absent']} | "
        f"Attendance: {summary['attendance_percentage']}%"
    )
    print(
        f"Performance   : {frames_processed} frames in {elapsed_sec:.2f}s ({fps:.2f} FPS) | "
        f"OSNet Inferences: {id_stats['total_inferences']} executed, {id_stats['total_skipped_frames']} skipped"
    )
    print(f"CSV Report    : {csv_path}")
    print(f"JSON Report   : {json_path}")

    if verifier is not None:
        report_dir = csv_path.parent / "verification"
        v_csv, v_json, v_records = verifier.export(
            report_dir,
            photos_dir=args.photos_dir or args.students_dir,
            video_path=None if args.no_clips else video_path,
        )
        print("\n--- VERIFICATION ---")
        for rec in v_records:
            if rec.verification in ("REVIEW", "POSSIBLY_PRESENT"):
                print(f"{rec.student_id:<18} | {rec.name:<20} | {rec.verification:<16} | {'; '.join(rec.reasons)}")
        counts = {s: sum(r.verification == s for r in v_records) for s in ("VERIFIED", "REVIEW", "POSSIBLY_PRESENT", "ABSENT")}
        print(
            f"Verified: {counts['VERIFIED']} | Review: {counts['REVIEW']} | "
            f"Possibly present: {counts['POSSIBLY_PRESENT']} | Absent: {counts['ABSENT']}"
        )
        print(f"Verification  : {v_csv}")
        print(f"Evidence      : {report_dir}/<student_id>/")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
