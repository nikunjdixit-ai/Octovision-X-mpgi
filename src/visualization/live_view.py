from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.attendance.attendance_engine import STATUS_PRESENT, AttendanceEngine, format_seconds_hms
from src.core.types import FramePacket, TrackedPerson
from src.identity.identity_engine import STATUS_CONFIRMED, STATUS_UNKNOWN, IdentityEngine

logger = logging.getLogger("trace.visualization.live")

WINDOW_NAME = "TRACE Attendance - press Q to stop"

# BGR colours
GREEN = (80, 200, 80)
YELLOW = (40, 200, 230)
GREY = (150, 150, 150)
WHITE = (240, 240, 240)
PANEL_BG = (32, 28, 26)
ROW_BG = (48, 42, 40)


class LiveAttendanceView:
    """
    Live view of the attendance run, registered as an extra PipelineController subscriber
    after AttendanceEngine (or AttendanceVerifier), so it draws the decisions of the same frame:
    - each tracked person boxed: green + name once CONFIRMED, yellow "checking..." while
      the identity is being decided, grey "Unknown" for people not matched to the roster;
    - a side panel with the class list (ABSENT -> PRESENT as students are confirmed),
      the present count and the video time.

    `show=True` opens a window (needs a display; press Q to stop early).
    `save_path` writes the same view to an .mp4 (works on headless machines such as Colab).
    """

    def __init__(
        self,
        attendance_engine: AttendanceEngine,
        identity_engine: IdentityEngine,
        show: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        fps: float = 25.0,
        max_video_width: int = 1280,
        panel_width: int = 380,
        on_quit: Optional[Callable[[], None]] = None,
    ):
        self.attendance_engine = attendance_engine
        self.identity_engine = identity_engine
        self.show = bool(show)
        self.save_path = Path(save_path) if save_path else None
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.max_video_width = int(max_video_width)
        self.panel_width = int(panel_width)
        self.on_quit = on_quit
        self._writer: Optional[cv2.VideoWriter] = None
        self._window_open = False

    def __call__(self, packet: FramePacket, tracked_people: List[TrackedPerson]) -> None:
        if packet is None or packet.frame is None:
            return
        canvas = self.render(packet, tracked_people)

        if self.save_path is not None:
            if self._writer is None:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                h, w = canvas.shape[:2]
                self._writer = cv2.VideoWriter(str(self.save_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self._writer.write(canvas)

        if self.show:
            if not self._window_open:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                self._window_open = True
            cv2.imshow(WINDOW_NAME, canvas)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")) and self.on_quit is not None:
                self.on_quit()

    def render(self, packet: FramePacket, tracked_people: List[TrackedPerson]) -> np.ndarray:
        frame = packet.frame.copy()
        for person in tracked_people:
            label, color = self._person_label(person)
            x1, y1, x2, y2 = person.bbox.to_xyxy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            _draw_tag(frame, label, (x1, y1), color)

        h, w = frame.shape[:2]
        if w > self.max_video_width:
            scale = self.max_video_width / float(w)
            frame = cv2.resize(frame, (self.max_video_width, int(h * scale)))

        panel = self._panel(frame.shape[0], packet)
        return np.hstack([frame, panel])

    def _person_label(self, person: TrackedPerson) -> Tuple[str, Tuple[int, int, int]]:
        dec = self.identity_engine.get_track_state(person.stream_track_uid)
        if dec is None:
            return "checking...", YELLOW
        if dec.status == STATUS_CONFIRMED and dec.student_id:
            return dec.name, GREEN
        if dec.status == STATUS_UNKNOWN:
            return "Unknown", GREY
        return "checking...", YELLOW

    def _panel(self, height: int, packet: FramePacket) -> np.ndarray:
        panel = np.full((height, self.panel_width, 3), PANEL_BG, dtype=np.uint8)
        records = self.attendance_engine.get_attendance_records()
        present = sum(1 for r in records if r.status == STATUS_PRESENT)

        _text(panel, "CLASS ATTENDANCE", (16, 34), 0.7, WHITE, 2)
        _text(panel, f"Video time  {format_seconds_hms(packet.timestamp)}", (16, 64), 0.5, GREY)
        _text(panel, f"Present {present} / {len(records)}", (16, 100), 0.8, GREEN if present else WHITE, 2)

        top, row_h = 120, 30
        max_rows = max(1, (height - top - 10) // row_h)
        shown = records[:max_rows]
        for i, rec in enumerate(shown):
            y = top + i * row_h
            cv2.rectangle(panel, (10, y + 3), (self.panel_width - 10, y + row_h - 3), ROW_BG, -1)
            is_present = rec.status == STATUS_PRESENT
            cv2.circle(panel, (24, y + row_h // 2), 6, GREEN if is_present else GREY, -1)
            _text(panel, _fit(f"{rec.roll_no}  {rec.name}", 26), (38, y + 20), 0.5, WHITE)
            _text(panel, rec.status, (self.panel_width - 92, y + 20), 0.45, GREEN if is_present else GREY)
        if len(records) > len(shown):
            _text(panel, f"+ {len(records) - len(shown)} more", (16, top + len(shown) * row_h + 20), 0.5, GREY)
        return panel

    def close(self) -> Optional[Path]:
        """Releases the video writer and window; returns the saved video path, if any."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._window_open:
            cv2.destroyWindow(WINDOW_NAME)
            self._window_open = False
        return self.save_path if self.save_path is not None and self.save_path.exists() else None


def _text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float, color, thickness: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _fit(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1] + "."


def _draw_tag(img: np.ndarray, label: str, corner: Tuple[int, int], color) -> None:
    x, y = corner
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    top = max(0, y - th - 12)
    cv2.rectangle(img, (x, top), (x + tw + 10, top + th + 12), color, -1)
    cv2.putText(img, label, (x + 5, top + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
