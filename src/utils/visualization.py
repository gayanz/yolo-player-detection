"""Drawing helpers for detections, tracks, and keypoints."""

from __future__ import annotations

import cv2
import numpy as np

# OpenPose BODY-18 skeleton connections (0-indexed)
OPENPOSE_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16),
    (0, 15), (15, 17),
]


def draw_bounding_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw a labeled bounding box on the frame."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_w, text_h = text_size
    cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    keypoint_color: tuple[int, int, int],
    skeleton_color: tuple[int, int, int],
    radius: int = 4,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw OpenPose-style keypoints and skeleton.

    Args:
        keypoints: Array of shape (num_keypoints, 3) -> [x, y, confidence].
    """
    points = []
    for x, y, confidence in keypoints:
        if confidence > 0.1:
            point = (int(x), int(y))
            points.append(point)
            cv2.circle(frame, point, radius, keypoint_color, -1, cv2.LINE_AA)
        else:
            points.append(None)

    for start_idx, end_idx in OPENPOSE_SKELETON:
        if start_idx >= len(points) or end_idx >= len(points):
            continue
        if points[start_idx] is None or points[end_idx] is None:
            continue
        cv2.line(frame, points[start_idx], points[end_idx], skeleton_color, thickness, cv2.LINE_AA)

    return frame


def draw_player(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    track_id: int | None,
    keypoints: np.ndarray | None,
    vis_config: dict,
) -> np.ndarray:
    """Draw one tracked player with optional keypoints."""
    label = f"Player {track_id}" if track_id is not None else "Player"
    frame = draw_bounding_box(
        frame,
        bbox,
        label,
        tuple(vis_config["bbox_color"]),
        vis_config.get("line_thickness", 2),
    )

    if keypoints is not None and len(keypoints) > 0:
        frame = draw_keypoints(
            frame,
            keypoints,
            tuple(vis_config["keypoint_color"]),
            tuple(vis_config["skeleton_color"]),
            radius=vis_config.get("keypoint_radius", 4),
            thickness=vis_config.get("line_thickness", 2),
        )
    return frame
