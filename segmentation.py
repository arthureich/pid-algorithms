"""Greyscale intensity segmentation."""
from __future__ import annotations

from typing import List

from utils import zeros


def intensity_segmentation(image: List[List[int]]) -> List[List[int]]:
    height = len(image)
    width = len(image[0])
    result = zeros(height, width, 0)
    for y in range(height):
        for x in range(width):
            value = image[y][x]
            if 0 <= value <= 50:
                result[y][x] = 25
            elif 51 <= value <= 100:
                result[y][x] = 75
            elif 101 <= value <= 150:
                result[y][x] = 125
            elif 151 <= value <= 200:
                result[y][x] = 175
            else:
                result[y][x] = 255
    return result
