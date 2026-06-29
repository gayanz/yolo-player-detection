"""Video I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import cv2
import numpy as np


def read_video_frames(
    video_path: str | Path,
    max_frames: int | None = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_index, frame_bgr) from a video file.

    Args:
        video_path: Path to input video.
        max_frames: Stop after this many frames (None = entire video).
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_index = 0
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break

            success, frame = capture.read()
            if not success:
                break

            yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def get_video_properties(video_path: str | Path) -> dict[str, float | int]:
    """Read basic video metadata."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
    }


def create_video_writer(
    output_path: str | Path,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    """Create an MP4 writer for annotated output videos."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")
    return writer


def crop_with_padding(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: float = 0.1,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Crop a region from the frame with optional padding.

    Returns:
        crop: Cropped image.
        offset: (x_offset, y_offset) of crop in original frame coordinates.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    return frame[y1:y2, x1:x2].copy(), (x1, y1)
