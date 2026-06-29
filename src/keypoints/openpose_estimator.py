"""OpenPose keypoint estimation module (PyTorch)."""

from __future__ import annotations

import numpy as np
import torch
from controlnet_aux.open_pose import OpenposeDetector


# OpenPose BODY-18 joint names (standard CMU format)
BODY_JOINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar",
]


class OpenPoseEstimator:
    """
    Estimate human body keypoints using OpenPose (PyTorch via controlnet-aux).

    Runs on player crops for cleaner per-player pose estimation.
    """

    def __init__(
        self,
        device: str = "auto",
        include_hands: bool = False,
        include_face: bool = False,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.include_hands = include_hands
        self.include_face = include_face

        # Downloads pretrained OpenPose weights on first use
        self.detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        self.detector.to(device)

    def estimate(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        """
        Estimate keypoints for all people in an image.

        Args:
            image_bgr: Input image in BGR format (OpenCV).

        Returns:
            List of keypoint arrays, each with shape (18, 3) -> [x, y, confidence].
        """
        height, width = image_bgr.shape[:2]
        image_rgb = image_bgr[:, :, ::-1].copy()

        pose_results = self.detector.detect_poses(
            image_rgb,
            include_hand=self.include_hands,
            include_face=self.include_face,
        )
        return self._results_to_keypoints(pose_results, width, height)

    def estimate_on_crop(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        padding: float = 0.1,
    ) -> np.ndarray | None:
        """
        Estimate keypoints for a single player crop and map back to frame coords.

        Returns:
            Keypoints array (18, 3) in full-frame coordinates, or None if no pose found.
        """
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]

        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * padding)
        pad_y = int(box_h * padding)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(width, x2 + pad_x)
        crop_y2 = min(height, y2 + pad_y)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return None

        poses = self.estimate(crop)
        if not poses:
            return None

        # Use the pose with the highest total joint confidence
        best_pose = max(poses, key=lambda kp: float(kp[:, 2].sum()))
        best_pose = best_pose.copy()
        best_pose[:, 0] += crop_x1
        best_pose[:, 1] += crop_y1
        return best_pose

    @staticmethod
    def _results_to_keypoints(
        pose_results: list,
        width: int,
        height: int,
    ) -> list[np.ndarray]:
        """Convert controlnet-aux PoseResult objects to pixel keypoint arrays."""
        poses: list[np.ndarray] = []

        for result in pose_results:
            keypoints = np.zeros((18, 3), dtype=np.float32)
            for joint_idx, joint in enumerate(result.body.keypoints):
                if joint is None:
                    continue
                x_norm = float(joint.x)
                y_norm = float(joint.y)
                score = float(joint.score)

                # OpenPose marks missing joints with negative normalized coords
                if x_norm < 0 or y_norm < 0:
                    continue

                keypoints[joint_idx, 0] = x_norm * width
                keypoints[joint_idx, 1] = y_norm * height
                keypoints[joint_idx, 2] = score

            if keypoints[:, 2].sum() > 0:
                poses.append(keypoints)

        return poses
