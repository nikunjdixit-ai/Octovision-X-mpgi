# Attendance Verification & Proof

Author: Divyanshi Bajpai (AI Search Engineer)

Attendance marks each student PRESENT or ABSENT. This step answers the next
question a teacher or admin asks: **why** was this student marked present, and
**which results should a person double-check?**

```
Enrollment → Detection → Tracking → Identity → Attendance → Verification & Proof → Dashboard
```

It runs alongside the attendance engine in the same pass over the video. It
never changes the PRESENT/ABSENT result; it adds a label and evidence next to it.

## What it adds

**1. Class timeline (per student)** — when the student was in class, split
into segments whenever they are not seen for more than 30 s, with the track
IDs behind each segment. Shows late arrival, leaving early or stepping out.

**2. Verification label**

| Label | Meaning |
|---|---|
| `VERIFIED` | PRESENT, and none of the warning signs below |
| `REVIEW` | PRESENT, but worth a look: seen for under 5 s, confirmed in fewer than 3 frames, the same tracked person was also matched to another student, or another person was matched to this student at the same moment |
| `POSSIBLY_PRESENT` | ABSENT, but the student was the closest (unconfirmed) match for someone at least 3 times — check before treating as absent |
| `ABSENT` | ABSENT with no supporting evidence |

**3. Proof per student** (in `verification/<student_id>/`)

| File | Content |
|---|---|
| `frame.jpg` | The best matching frame, with the student boxed and named |
| `proof.jpg` | Enrollment photo next to the matched person, side by side |
| `clip.mp4` | ±3 s of video around that frame |

For `POSSIBLY_PRESENT` students the same files are written as
`closest_frame.jpg`, `closest_proof.jpg`, `closest_clip.mp4`.

## Usage

```bash
python scripts/run_attendance.py --video <class video> --registry data/enrollment/registry \
    --verify --photos-dir data/enrollment/students
```

- `--verify` turns the step on (off by default; without it attendance runs exactly as before).
- `--photos-dir` — the student photo folders used at enrollment, for `proof.jpg`
  (defaults to `--students-dir`).
- `--no-clips` skips `clip.mp4` for faster runs.

Output, next to the attendance report:

```
output/attendance/verification/
├── verification.csv      one row per student: label, seen from/to, segments, reasons
├── verification.json     same, plus summary counts
└── <student_id>/         proof files
```

## Design notes

- Recording evidence can never break attendance: attendance is updated first,
  and any error while saving evidence is logged and skipped.
- Only frames where the identity model actually ran count as candidate evidence;
  cached decisions between inference frames are not double-counted.
- Thresholds live in `VerificationConfig` (`src/verification/attendance_verifier.py`).
- Tests: `python -m pytest tests/test_attendance_verifier.py` (no models or GPU needed).

## Privacy

Evidence folders contain student images and video. They are written under
`output/`, which is git-ignored — never commit them, and share them only with
staff who are allowed to see attendance records.
