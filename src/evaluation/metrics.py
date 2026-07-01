"""Detection evaluation metrics: precision, recall, F1, and accuracy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.detection.yolo_detector import YOLOPlayerDetector


def box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union for two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    pred_boxes: list[tuple[int, int, int, int]],
    gt_boxes: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int, int]:
    """
    Match predicted boxes to ground truth using greedy IoU matching.

    Returns:
        tp, fp, fn, tn counts for this frame.
    """
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0, 1

    if not gt_boxes:
        return 0, len(pred_boxes), 0, 0

    if not pred_boxes:
        return 0, 0, len(gt_boxes), 0

    matched_gt: set[int] = set()
    true_positives = 0

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives += 1
            matched_gt.add(best_gt_idx)

    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - len(matched_gt)
    return true_positives, false_positives, false_negatives, 0


def compute_metrics(tp: int, fp: int, fn: int, tn: int = 0) -> dict[str, float]:
    """Compute precision, recall, F1, and accuracy from confusion counts."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def load_annotations(annotation_path: str | Path) -> dict[int, list[tuple[int, int, int, int]]]:
    """
    Load per-frame ground truth boxes from JSON.

    JSON format:
    {
      "frames": [
        {"frame_index": 0, "boxes": [[x1, y1, x2, y2], ...]},
        ...
      ]
    }
    """
    with Path(annotation_path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    annotations: dict[int, list[tuple[int, int, int, int]]] = {}
    for frame_data in data.get("frames", []):
        frame_idx = int(frame_data["frame_index"])
        boxes = [tuple(map(int, box)) for box in frame_data.get("boxes", [])]
        annotations[frame_idx] = boxes
    return annotations


def save_annotations(
    annotations: dict[int, list[tuple[int, int, int, int]]],
    output_path: str | Path,
    video_name: str,
) -> Path:
    """Save annotations to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "video": video_name,
        "frames": [
            {"frame_index": frame_idx, "boxes": [list(box) for box in boxes]}
            for frame_idx, boxes in sorted(annotations.items())
        ],
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return output_path


def generate_proxy_annotations(
    video_path: str | Path,
    detector: YOLOPlayerDetector,
    frame_indices: list[int],
) -> dict[int, list[tuple[int, int, int, int]]]:
    """
    Create proxy ground truth using a stronger YOLO model.

    Useful for demo evaluation when manual labels are not available.
    Replace with human-annotated JSON for real evaluation.
    """
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    annotations: dict[int, list[tuple[int, int, int, int]]] = {}
    for frame_idx in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = capture.read()
        if not success:
            continue
        detections = detector.detect(frame)
        annotations[frame_idx] = [det.bbox for det in detections]

    capture.release()
    return annotations


def evaluate_video_detections(
    detector: YOLOPlayerDetector,
    video_path: str | Path,
    annotations: dict[int, list[tuple[int, int, int, int]]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a detector against frame-level ground truth annotations."""
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_tp = total_fp = total_fn = total_tn = 0
    frame_results: list[dict[str, Any]] = []

    for frame_idx, gt_boxes in sorted(annotations.items()):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = capture.read()
        if not success:
            continue

        pred_boxes = [det.bbox for det in detector.detect(frame)]
        tp, fp, fn, tn = match_boxes(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
        frame_metrics = compute_metrics(tp, fp, fn, tn)
        frame_metrics["frame_index"] = frame_idx
        frame_results.append(frame_metrics)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    capture.release()
    overall = compute_metrics(total_tp, total_fp, total_fn, total_tn)
    overall["frames_evaluated"] = len(frame_results)
    overall["frame_results"] = frame_results
    return overall


def evaluate_at_confidence_thresholds(
    detector: YOLOPlayerDetector,
    video_path: str | Path,
    annotations: dict[int, list[tuple[int, int, int, int]]],
    thresholds: list[float],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Evaluate detection metrics across multiple confidence thresholds."""
    original_confidence = detector.confidence
    results = []

    for threshold in thresholds:
        detector.confidence = threshold
        metrics = evaluate_video_detections(
            detector=detector,
            video_path=video_path,
            annotations=annotations,
            iou_threshold=iou_threshold,
        )
        metrics["confidence_threshold"] = threshold
        results.append(metrics)

    detector.confidence = original_confidence
    return results
