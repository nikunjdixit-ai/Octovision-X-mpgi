from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
from ultralytics import YOLO

# Add project root to sys.path if running from subfolder
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ..core.types import BoundingBox, TrackedPerson
except (ImportError, ValueError):
    from src.core.types import BoundingBox, TrackedPerson

logger = logging.getLogger("trace.tracker")


class PersonTracker:
    """
    ByteTrack-based multi-person tracking engine with namespaced track UIDs,
    session lifecycle management, and tracker reset support.
    Maintains 100% backward compatibility with legacy tracking scripts.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        tracker_type: str = "bytetrack.yaml",
        conf: float = 0.40,
        iou: float = 0.45,
        imgsz: int = 640,
        classes: Optional[List[int]] = None,
        camera_id: str = "CAM_01",
        session_id: str = "SESSION_01",
        device: Optional[Union[str, int]] = None,
        half: Optional[bool] = None,
    ):
        self.model_path = model_path
        self.tracker_type = tracker_type
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes if classes is not None else [0]  # Person class only
        self.camera_id = camera_id
        self.session_id = session_id

        # Device determination
        if device is not None:
            self.device = device
        else:
            self.device = 0 if torch.cuda.is_available() else "cpu"

        self.half = half

        logger.info(
            f"Initializing PersonTracker ({model_path}, {tracker_type}) for camera={camera_id}, session={session_id} on device={self.device}"
        )
        if self.device == 0 or (isinstance(self.device, str) and "cuda" in str(self.device).lower()):
            dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
            print(f"Using GPU: {dev_name}")
        else:
            print("CUDA not available. Running on CPU.")

        self.model = YOLO(model_path)

    def reset(self, new_session_id: Optional[str] = None) -> None:
        """
        Resets ByteTrack's internal Kalman filters, active tracks, and track ID counters.
        Must be called when switching to a new video or camera stream to prevent state leakage.
        """
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor = None

        if new_session_id is not None:
            self.session_id = new_session_id

        logger.info(f"PersonTracker reset completed for camera={self.camera_id}, session={self.session_id}")

    def track(self, frame: np.ndarray):
        """
        Executes ByteTrack on a single frame.
        Maintains 100% backwards compatibility with legacy code accessing results[0].plot() / results[0].boxes.
        """
        if frame is None or (isinstance(frame, np.ndarray) and frame.size == 0):
            logger.warning("Empty or None frame received by PersonTracker.track()")
            return []

        try:
            kwargs = {
                "source": frame,
                "persist": True,
                "tracker": self.tracker_type,
                "classes": self.classes,
                "conf": self.conf,
                "iou": self.iou,
                "imgsz": self.imgsz,
                "device": self.device,
                "verbose": False,
                "stream": False,
            }
            if self.half is not None:
                kwargs["half"] = self.half

            results = self.model.track(**kwargs)
            return results
        except Exception as exc:
            logger.error(f"Tracking inference error: {exc}")
            return []

    def track_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: str = ""
    ) -> List[TrackedPerson]:
        """
        Executes tracking and parses detections into typed, namespaced TrackedPerson objects.
        Guarantees that track IDs are scoped under stream_track_uid: f'{camera_id}_{session_id}_{track_id}'.
        """
        results = self.track(frame)
        if not results:
            return []

        tracked_persons: List[TrackedPerson] = []
        try:
            res = results[0]
            boxes = res.boxes
            if boxes is not None and boxes.id is not None and len(boxes) > 0:
                for box, track_id in zip(boxes, boxes.id):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item()) if box.cls is not None else 0
                    t_id = int(track_id.item())

                    bbox = BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=round(conf, 4),
                        class_id=cls_id
                    )

                    person = TrackedPerson(
                        camera_id=self.camera_id,
                        session_id=self.session_id,
                        track_id=t_id,
                        bbox=bbox,
                        frame_id=frame_id,
                        timestamp=timestamp
                    )
                    tracked_persons.append(person)
        except Exception as exc:
            logger.error(f"Error parsing tracked persons: {exc}")

        return tracked_persons
