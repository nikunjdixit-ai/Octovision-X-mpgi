from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.identity.enrollment import EnrollmentError, StudentRegistry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TRACE Phase 6 — Student Enrollment & OSNet Embedding Registry Builder"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dir",
        type=str,
        help="Path to student directory (e.g., data/students/ containing 01_Rahul_Sharma/, etc.)",
    )
    group.add_argument(
        "--zip",
        type=str,
        help="Path to ZIP archive containing student reference folders",
    )
    parser.add_argument(
        "--registry-dir",
        type=str,
        default="data/enrollment/registry",
        help="Output directory for persisted student registry JSON and .npy embeddings",
    )
    parser.add_argument(
        "--department",
        type=str,
        default="Computer Science",
        help="Department label for enrolled students",
    )
    parser.add_argument(
        "--section",
        type=str,
        default="A",
        help="Class section label for enrolled students",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=None,
        help="Optional CSV path (student_id,name,roll_no,department,section) for authoritative student metadata",
    )

    args = parser.parse_args()

    print("=============================================")
    print("       TRACE STUDENT ENROLLMENT CLI          ")
    print("=============================================")

    t0 = time.perf_counter()
    registry = StudentRegistry(registry_dir=args.registry_dir)

    try:
        if args.zip:
            print(f"Source ZIP     : {args.zip}")
            profiles = registry.enroll_from_zip(
                zip_path=args.zip,
                department=args.department,
                section=args.section,
                persist=True,
                metadata_csv=args.metadata_csv,
            )
        else:
            print(f"Source Folder  : {args.dir}")
            profiles = registry.enroll_from_directory(
                students_dir=args.dir,
                department=args.department,
                section=args.section,
                persist=True,
                metadata_csv=args.metadata_csv,
            )
    except EnrollmentError as err:
        print(f"\nENROLLMENT FAILED: {err}")
        print("=============================================")
        return 1
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {exc}")
        print("=============================================")
        return 1

    elapsed = time.perf_counter() - t0
    total_samples = sum(p.num_samples for p in profiles)
    print(f"Registry Path  : {args.registry_dir}")
    print(f"Students Added : {len(profiles)}")
    print(f"Total Samples  : {total_samples}")
    print(f"Elapsed Time   : {elapsed:.2f}s")
    print("---------------------------------------------")
    for p in profiles:
        print(f"  [{p.roll_number}] {p.name:<22} | ID: {p.student_id:<20} | Samples: {p.num_samples}")
    print("=============================================")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
