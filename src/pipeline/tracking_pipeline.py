"""End-to-end player tracking pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from src.config_loader import load_config
from src.detection.yolo_detector import YOLOPlayerDetector
from src.keypoints.openpose_estimator import OpenPoseEstimator
from src.tracking.player_tracker import PlayerTracker
from src.utils.video_utils import create_video_writer, get_video_properties, read_video_frames
from src.utils.visualization import draw_player


class PlayerTrackingPipeline:
    """
    Full pipeline: YOLO v11 tracking + OpenPose keypoints on sports videos.

    Steps per frame:
      1. Track players with YOLO v11 (ByteTrack)
      2. Estimate keypoints with OpenPose on each player crop
      3. Draw annotations and save results
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()

        yolo_cfg = self.config["models"]["yolo"]
        openpose_cfg = self.config["models"]["openpose"]
        track_cfg = self.config["models"]["tracking"]

        self.tracker = PlayerTracker(
            model_name=yolo_cfg["model_name"],
            confidence=yolo_cfg["confidence"],
            iou=yolo_cfg["iou"],
            person_class_id=yolo_cfg["person_class_id"],
            tracker=track_cfg["tracker"],
            persist=track_cfg["persist"],
        )
        self.pose_estimator = OpenPoseEstimator(
            device=openpose_cfg["device"],
            include_hands=openpose_cfg["include_hands"],
            include_face=openpose_cfg["include_face"],
        )
        self.vis_config = self.config["visualization"]
        self.processing = self.config["processing"]

    def process_video(self, video_path: str | Path) -> dict[str, Any]:
        """Process one video and return frame-level tracking results."""
        video_path = Path(video_path)
        output_dir = Path(self.config["paths"]["output_dir"]) / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        props = get_video_properties(video_path)
        max_frames = self.processing.get("max_frames")

        writer = None
        if self.processing.get("save_video", True):
            writer = create_video_writer(
                output_dir / f"{video_path.stem}_tracked.mp4",
                props["width"],
                props["height"],
                props["fps"],
            )

        self.tracker.reset()
        all_frames: list[dict[str, Any]] = []

        frame_iter = read_video_frames(video_path, max_frames=max_frames)
        total = min(max_frames, props["frame_count"]) if max_frames else props["frame_count"]

        for frame_idx, frame in tqdm(frame_iter, total=total, desc=video_path.name):
            tracked_players = self.tracker.track(frame)
            frame_data = {"frame": frame_idx, "players": []}

            annotated = frame.copy()
            for player in tracked_players:
                keypoints = self.pose_estimator.estimate_on_crop(frame, player.bbox)
                player_record = {
                    "track_id": player.track_id,
                    "bbox": list(player.bbox),
                    "confidence": player.confidence,
                    "keypoints": keypoints.tolist() if keypoints is not None else None,
                }
                frame_data["players"].append(player_record)

                annotated = draw_player(
                    annotated,
                    player.bbox,
                    player.track_id,
                    keypoints,
                    self.vis_config,
                )

            all_frames.append(frame_data)

            if writer is not None:
                writer.write(annotated)

            if self.processing.get("show_preview", False):
                cv2.imshow("Player Tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if writer is not None:
            writer.release()
        if self.processing.get("show_preview", False):
            cv2.destroyAllWindows()

        results = {
            "video": str(video_path),
            "fps": props["fps"],
            "frames": all_frames,
        }

        if self.processing.get("save_json", True):
            json_path = output_dir / f"{video_path.stem}_results.json"
            with json_path.open("w", encoding="utf-8") as file:
                json.dump(results, file, indent=2)

        return results

    def process_all_videos(self, video_paths: list[Path]) -> list[dict[str, Any]]:
        """Process multiple videos sequentially."""
        all_results = []
        for video_path in video_paths:
            print(f"\nProcessing: {video_path.name}")
            results = self.process_video(video_path)
            all_results.append(results)
        return all_results


def run_detection_only(frame, config: dict | None = None) -> list:
    """Utility: run YOLO detection on a single frame (for Colab Part 2)."""
    config = config or load_config()
    yolo_cfg = config["models"]["yolo"]
    detector = YOLOPlayerDetector(
        model_name=yolo_cfg["model_name"],
        confidence=yolo_cfg["confidence"],
        iou=yolo_cfg["iou"],
        person_class_id=yolo_cfg["person_class_id"],
    )
    return detector.detect(frame)


def run_keypoints_only(frame, bbox, config: dict | None = None):
    """Utility: run OpenPose on a single player crop (for Colab Part 3)."""
    config = config or load_config()
    openpose_cfg = config["models"]["openpose"]
    estimator = OpenPoseEstimator(
        device=openpose_cfg["device"],
        include_hands=openpose_cfg["include_hands"],
        include_face=openpose_cfg["include_face"],
    )
    return estimator.estimate_on_crop(frame, bbox)


def run_tracking_only(frame, config: dict | None = None) -> list:
    """Utility: run tracking on a single frame (for Colab Part 4)."""
    config = config or load_config()
    yolo_cfg = config["models"]["yolo"]
    track_cfg = config["models"]["tracking"]
    tracker = PlayerTracker(
        model_name=yolo_cfg["model_name"],
        confidence=yolo_cfg["confidence"],
        iou=yolo_cfg["iou"],
        person_class_id=yolo_cfg["person_class_id"],
        tracker=track_cfg["tracker"],
        persist=track_cfg["persist"],
    )
    return tracker.track(frame)
