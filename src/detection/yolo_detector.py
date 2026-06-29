"""YOLO v11 player detection module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    """Single person detection."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int


class YOLOPlayerDetector:
    """Detect players (person class) in video frames using YOLO v11."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.4,
        iou: float = 0.5,
        person_class_id: int = 0,
        device: str | None = None,
    ) -> None:
        self.confidence = confidence
        self.iou = iou
        self.person_class_id = person_class_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run person detection on a single BGR frame."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.person_class_id],
            verbose=False,
            device=self.device,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=confidence,
                    class_id=class_id,
                )
            )
        return detections
