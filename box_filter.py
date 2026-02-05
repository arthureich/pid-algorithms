"""Box filter (mean filter) implementations."""
from __future__ import annotations

from typing import List

from utils import convolve2d, zeros


def box_filter(image: List[List[int]], size: int) -> List[List[int]]:
    kernel = [[1.0 / (size * size) for _ in range(size)] for _ in range(size)]
    filtered = convolve2d(image, kernel)
    height = len(filtered)
    width = len(filtered[0])
    output = zeros(height, width, 0)
    for y in range(height):
        for x in range(width):
            value = int(round(filtered[y][x]))
            output[y][x] = max(0, min(255, value))
    return output
