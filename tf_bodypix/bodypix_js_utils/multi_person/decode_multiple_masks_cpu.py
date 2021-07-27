# based on;
# https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.5/body-pix/src/multi_person/decode_multiple_masks_cpu.ts

import numpy as np

from typing import NamedTuple, List, Callable, Tuple

from ..keypoints import NUM_KEYPOINTS
from ..types import Padding, Pose

from .util import getScale


class Pair(NamedTuple):
    x: float
    y: float


def computeDistance(
        embedding: List[Pair], pose: Pose, min_part_score: float = 0.3
) -> float:
    distance = 0.
    num_kpt = 0
    for p, embed in enumerate(embedding):
        if pose.keypoints[p].score > min_part_score:
            num_kpt += 1
            distance += (embed.x - pose.keypoints[p].position.x) ** 2 + \
                        (embed.y - pose.keypoints[p].position.y) ** 2
    if num_kpt == 0:
        distance = np.inf
    else:
        distance /= num_kpt
    return distance


def convertToPositionInOutput(
        position: Pair, pad_t: int, pad_l: int,
        scale_x: float, scale_y: float, stride: int
) -> Pair:
    y = round(((pad_t + position.y + 1.) * scale_y - 1.) / stride)
    x = round(((pad_l + position.x + 1.) * scale_x - 1.) / stride)
    return Pair(x, y)


def getEmbedding(
        location: Pair, keypoint_index: int,
        convert_to_position: Callable[[Pair], Pair],
        output_resolution_x: float, long_offsets: np.ndarray,
        refine_steps: int, height: int, width: int
) -> Pair:
    new_location = convert_to_position(location)
    nn = new_location.y * output_resolution_x + new_location.x
    dy = long_offsets[NUM_KEYPOINTS * (2 * nn) + keypoint_index]
    dx = long_offsets[NUM_KEYPOINTS * (2 * nn + 1) + keypoint_index]
    y = location.y + dy
    x = location.x + dx
    for t in range(refine_steps):
        y = min(y, height - 1)
        x = min(x, width - 1)
        new_pos = convert_to_position(Pair(x, y))
        nn = new_pos.y * output_resolution_x + new_pos.x
        dy = long_offsets[NUM_KEYPOINTS * (2 * nn) + keypoint_index]
        dx = long_offsets[NUM_KEYPOINTS * (2 * nn + 1) + keypoint_index]
        y += dy
        x += dx
    return Pair(x, y)


def matchEmbeddingToInstance(
        location: Pair, long_offsets: np.ndarray, poses: List[Pose],
        num_kpt_for_matching: int, pad_t: int, pad_l: int,
        scale_x: float, scale_y: float, output_resolution_x: float,
        height: int, width: int, stride: int, refine_steps: int
) -> int:
    def convertToPosition(pair: Pair) -> Pair:
        return convertToPositionInOutput(
            pair, pad_t, pad_l, scale_x, scale_y, stride
        )

    embed = []
    for keypoints_index in range(num_kpt_for_matching):
        embedding = getEmbedding(
            location, keypoints_index, convertToPosition, output_resolution_x,
            long_offsets, refine_steps, height, width
        )
        embed.append(embedding)
    k_min = -1
    k_min_dist = np.inf
    for k, pose in enumerate(poses):
        dist = computeDistance(embed, pose)
        if dist < k_min_dist:
            k_min = k
            k_min_dist = dist
    return k_min


def getOutputResolution(
        input_resolution_y: int, input_resolution_x: int,
        stride: int
) -> Tuple[float, float]:
    output_resolution_x = round((input_resolution_x - 1.) / stride + 1.)
    output_resolution_y = round((input_resolution_y - 1.) / stride + 1.)
    return output_resolution_x, output_resolution_y


def decodeMultipleMasksCPU(
        segmentation: np.ndarray, long_offsets: np.ndarray,
        poses_above_score: List[Pose], height: int, width: int,
        stride: int, in_height: int, in_width: int, padding: Padding,
        refine_steps: int, num_kpt_for_matching: int = 5
) -> np.ndarray:
    data_arrays = [
        np.zeros(height * width, dtype=np.uint8)
        for _ in poses_above_score
    ]
    pad_t, pad_l = padding.top, padding.left
    scale_x, scale_y = getScale(height, width, in_height, in_width, padding)
    output_resolution_x, _ = getOutputResolution(in_height, in_width, stride)
    for i in range(height):
        for j in range(width):
            n = i * width + j
            prob = segmentation[n]
            if prob == 1:
                k_min = matchEmbeddingToInstance(
                    Pair(j, i), long_offsets, poses_above_score, num_kpt_for_matching,
                    pad_t, pad_l, scale_x, scale_y, output_resolution_x, height, width,
                    stride, refine_steps
                )
                if k_min >= 0:
                    data_arrays[k_min][n] = 1
    # Reshape data into 2D mask
    return np.stack([data.reshape(height, width) for data in data_arrays])


def decodeMultiplePartMasksCPU(
        segmentation: np.ndarray, long_offsets: np.ndarray,
        part_segmentation: np.ndarray, poses_above_score: List[Pose],
        height: int, width: int, stride: int, in_height: int, in_width: int,
        padding: Padding, refine_steps: int, num_kpt_for_matching: int = 5
) -> np.ndarray:
    data_arrays = [
        np.ones(height * width, dtype=np.int32) * -1
        for _ in poses_above_score
    ]
    pad_t, pad_l = padding.top, padding.left
    scale_x, scale_y = getScale(height, width, in_height, in_width, padding)
    output_resolution_x, _ = getOutputResolution(in_height, in_width, stride)
    for i in range(height):
        for j in range(width):
            n = i * width + j
            prob = segmentation[n]
            if prob == 1:
                k_min = matchEmbeddingToInstance(
                    Pair(j, i), long_offsets, poses_above_score, num_kpt_for_matching,
                    pad_t, pad_l, scale_x, scale_y, output_resolution_x, height, width,
                    stride, refine_steps
                )
                if k_min >= 0:
                    data_arrays[k_min][n] = part_segmentation[n]
    # Reshape data into 2D mask
    return np.stack([data.reshape(height, width) for data in data_arrays])
