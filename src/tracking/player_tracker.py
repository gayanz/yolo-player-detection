"""Multi-object player tracking using YOLO v11 + ByteTrack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class TrackedPlayer:
    """A tracked player in a single frame."""

    track_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


class PlayerTracker:
    """
    Track players across frames using YOLO v11 built-in tracking (ByteTrack).

    ByteTrack maintains consistent IDs by associating detections over time.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.4,
        iou: float = 0.5,
        person_class_id: int = 0,
        tracker: str = "bytetrack.yaml",
        persist: bool = True,
        device: str | None = None,
    ) -> None:
        self.confidence = confidence
        self.iou = iou
        self.person_class_id = person_class_id
        self.tracker = tracker
        self.persist = persist
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_name)
        self._needs_reset = False

    def track(self, frame: np.ndarray) -> list[TrackedPlayer]:
        """Track players in a single BGR frame."""
        use_persist = self.persist and not self._needs_reset
        results = self.model.track(
            source=frame,
            persist=use_persist,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.person_class_id],
            tracker=self.tracker,
            verbose=False,
            device=self.device,
        )

        tracked: list[TrackedPlayer] = []
        if not results:
            return tracked

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return tracked

        for box in boxes:
            if box.id is None:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            track_id = int(box.id[0].cpu().numpy())

            tracked.append(
                TrackedPlayer(
                    track_id=track_id,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=confidence,
                )
            )
        self._needs_reset = False
        return tracked

    def reset(self) -> None:
        """Reset tracker state (call before processing a new video)."""
        self._needs_reset = True
