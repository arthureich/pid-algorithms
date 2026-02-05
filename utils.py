from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import math


@dataclass
class EdgeResults:
    magnitude: List[List[float]]
    edges: List[List[int]]


def to_list(image) -> List[List[int]]:
    return [[int(pixel) for pixel in row] for row in image.tolist()]


def zeros(height: int, width: int, value: int = 0) -> List[List[int]]:
    return [[value for _ in range(width)] for _ in range(height)]


def convolve2d(image: List[List[int]], kernel: List[List[float]]) -> List[List[float]]:
    height = len(image)
    width = len(image[0])
    k_h = len(kernel)
    k_w = len(kernel[0])
    pad_h = k_h // 2
    pad_w = k_w // 2
    output: List[List[float]] = [[0.0 for _ in range(width)] for _ in range(height)]

    active_kernel_indices = []
    for ky in range(k_h):
        for kx in range(k_w):
            if kernel[ky][kx] != 0:
                active_kernel_indices.append((ky, kx, kernel[ky][kx]))

    for y in range(height):
        for x in range(width):
            acc = 0.0
            for ky, kx, k_val in active_kernel_indices:
                iy = y + ky - pad_h
                ix = x + kx - pad_w

                if iy < 0:
                    iy = 0
                elif iy >= height:
                    iy = height - 1

                if ix < 0:
                    ix = 0
                elif ix >= width:
                    ix = width - 1

                acc += image[iy][ix] * k_val
            output[y][x] = acc
    return output


def gaussian_kernel(size: int, sigma: float) -> List[List[float]]:
    ax = [i - size // 2 for i in range(size)]
    kernel: List[List[float]] = []
    total = 0.0
    for y in ax:
        row: List[float] = []
        for x in ax:
            value = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
            row.append(value)
            total += value
        kernel.append(row)
    for y in range(size):
        for x in range(size):
            kernel[y][x] /= total
    return kernel


def log_kernel(size: int, sigma: float) -> List[List[float]]:
    ax = [i - size // 2 for i in range(size)]
    kernel: List[List[float]] = []
    values: List[float] = []
    for y in ax:
        row: List[float] = []
        for x in ax:
            norm = (x * x + y * y - 2 * sigma * sigma) / (sigma**4)
            value = norm * math.exp(-(x * x + y * y) / (2 * sigma * sigma))
            row.append(value)
            values.append(value)
        kernel.append(row)
    mean_val = sum(values) / len(values)
    for y in range(size):
        for x in range(size):
            kernel[y][x] -= mean_val
    return kernel


def sobel_gradients(image: List[List[int]]) -> Tuple[List[List[float]], List[List[float]]]:
    kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    gx = convolve2d(image, kx)
    gy = convolve2d(image, ky)
    return gx, gy
