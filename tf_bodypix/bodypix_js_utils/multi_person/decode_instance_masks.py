# based on;
# https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.5/body-pix/src/multi_person/decode_instance_masks.ts

import numpy as np

from typing import List

from ..types import Padding, PartSegmentation, PersonSegmentation, Pose
from .decode_multiple_masks_cpu import decodeMultipleMasksCPU, decodeMultiplePartMasksCPU


def toPersonKSegmentation(
        segmentation: np.ndarray, k: int
) -> np.ndarray:
    return (segmentation == k).astype(np.int32)


def toPersonKPartSegmentation(
        segmentation: np.ndarray, body_parts: np.ndarray, k: int
) -> np.ndarray:
    return np.multiply(
        (segmentation == k).astype(np.int32),
        body_parts + 1
    ) - 1


def isWebGlBackend() -> bool:
    return False


def decodePersonInstanceMask(
        segmentation: np.ndarray, long_offsets: np.ndarray, poses: List[Pose],
        height: int, width: int, stride: int,
        in_height: int, in_width: int, padding: Padding, min_pose_score: float = 0.2,
        refine_steps: int = 8, min_keypoint_score: float = 0.3,
        max_num_people: int = 10
) -> List[PersonSegmentation]:
    # Filter out poses with smaller score.
    poses_above_score = [
        pose for pose in poses if pose.score >= min_pose_score
    ]
    if isWebGlBackend():
        # min_keypoint_score and max_num_people args only used with WebGl
        raise NotImplementedError
    else:
        person_segmentation_data = decodeMultipleMasksCPU(
            segmentation, long_offsets, poses_above_score, height, width,
            stride, in_height, in_width, padding, refine_steps
        )
    return [
        PersonSegmentation(data, width, height, poses_above_score[i])
        for i, data in enumerate(person_segmentation_data)
    ]


def decodePersonInstancePartMask(
        segmentation: np.ndarray, long_offsets: np.ndarray,
        part_segmentation: np.ndarray, poses: List[Pose], height: int, width:int,
        stride: int, in_height: int, in_width: int, padding: Padding,
        min_pose_score: float = 0.2, refine_steps: int = 8, min_keypoint_score: float = 0.3,
        max_num_people: int = 10
) -> List[PartSegmentation]:
    poses_above_score = [
        pose for pose in poses if pose.score >= min_pose_score
    ]
    if isWebGlBackend():
        # min_keypoint_score and max_num_people args only used with WebGl
        raise NotImplementedError
    else:
        part_segmentations_by_person_data = decodeMultiplePartMasksCPU(
            segmentation, long_offsets, part_segmentation,
            poses_above_score, height, width, stride, in_height, in_width, padding,
            refine_steps
        )
    return [
        PartSegmentation(data, width, height, poses_above_score[k])
        for k, data in enumerate(part_segmentations_by_person_data)
    ]
