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
    from ..core.types import BoundingBox
except (ImportError, ValueError):
    from src.core.types import BoundingBox

logger = logging.getLogger("trace.detector")


class PersonDetector:
    """
    Robust, headless YOLOv8 Person Detector.
    Supports CUDA GPU acceleration on RTX 3050 and gracefully handles CPU fallback.
    Maintains 100% backward compatibility with existing legacy scripts.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        conf: float = 0.40,
        iou: float = 0.45,
        imgsz: int = 640,
        classes: Optional[List[int]] = None,
        device: Optional[Union[str, int]] = None,
        half: Optional[bool] = None,
    ):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes if classes is not None else [0]  # Class 0 = Person

        # Device determination
        if device is not None:
            self.device = device
        else:
            self.device = 0 if torch.cuda.is_available() else "cpu"

        self.half = half if half is not None else (self.device != "cpu")

        logger.info(f"Loading YOLO detector: {model_path} on device={self.device}")
        self.model = YOLO(model_path)

        if self.device == 0 or (isinstance(self.device, str) and "cuda" in str(self.device).lower()):
            dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
            logger.info(f"YOLO detector initialized on GPU: {dev_name}")
            print(f"Using GPU: {dev_name}")
        else:
            logger.info("YOLO detector running on CPU")
            print("Using CPU")

    def detect(self, frame: np.ndarray):
        """
        Runs YOLO inference on a single frame.
        Maintains 100% backwards compatibility with existing Ultralytics results calls.
        """
        if frame is None or (isinstance(frame, np.ndarray) and frame.size == 0):
            logger.warning("Empty or None frame received by PersonDetector.detect()")
            return []

        try:
            kwargs = {
                "imgsz": self.imgsz,
                "classes": self.classes,
                "conf": self.conf,
                "iou": self.iou,
                "device": self.device,
                "half": self.half,
                "verbose": False
            }

            results = self.model(frame, **kwargs)
            return results
        except Exception as e:
            logger.error(f"Inference error in PersonDetector: {e}")
            return []

    def detect_boxes(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Runs inference and parses detections into typed BoundingBox objects.
        """
        results = self.detect(frame)
        if not results:
            return []

        boxes_out: List[BoundingBox] = []
        try:
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item()) if box.cls is not None else 0

                    boxes_out.append(
                        BoundingBox(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            confidence=round(conf, 4),
                            class_id=cls_id
                        )
                    )
        except Exception as exc:
            logger.error(f"Error parsing bounding boxes: {exc}")

        return boxes_out
