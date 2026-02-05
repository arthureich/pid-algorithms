"""Canny edge detection."""
from __future__ import annotations

from typing import List

import math

from utils import EdgeResults, convolve2d, gaussian_kernel, sobel_gradients, zeros


def non_maximum_suppression(
    magnitude: List[List[float]],
    angle: List[List[float]],
) -> List[List[float]]:
    height = len(magnitude)
    width = len(magnitude[0])
    suppressed = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            direction = angle[y][x] % 180
            q = 0.0
            r = 0.0
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                q = magnitude[y][x + 1]
                r = magnitude[y][x - 1]
            elif 22.5 <= direction < 67.5:
                q = magnitude[y + 1][x - 1]
                r = magnitude[y - 1][x + 1]
            elif 67.5 <= direction < 112.5:
                q = magnitude[y + 1][x]
                r = magnitude[y - 1][x]
            elif 112.5 <= direction < 157.5:
                q = magnitude[y - 1][x - 1]
                r = magnitude[y + 1][x + 1]
            if magnitude[y][x] >= q and magnitude[y][x] >= r:
                suppressed[y][x] = magnitude[y][x]
    return suppressed


def hysteresis_threshold(image: List[List[float]], low: float, high: float) -> List[List[int]]:
    height = len(image)
    width = len(image[0])
    strong = 255
    weak = 75
    result = zeros(height, width, 0)
    stack = []
    for y in range(height):
        for x in range(width):
            value = image[y][x]
            if value >= high:
                result[y][x] = strong
                stack.append((y, x))
            elif value >= low:
                result[y][x] = weak
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if result[ny][nx] == weak:
                        result[ny][nx] = strong
                        stack.append((ny, nx))
    for y in range(height):
        for x in range(width):
            if result[y][x] != strong:
                result[y][x] = 0
    return result


def canny(
    image: List[List[int]],
    low_ratio: float = 0.05,
    high_ratio: float = 0.15,
) -> EdgeResults:
    blurred = convolve2d(image, gaussian_kernel(5, 1.0))
    gx, gy = sobel_gradients([[int(round(value)) for value in row] for row in blurred])
    height = len(gx)
    width = len(gx[0])
    magnitude = [[0.0 for _ in range(width)] for _ in range(height)]
    angle = [[0.0 for _ in range(width)] for _ in range(height)]
    max_mag = 0.0
    for y in range(height):
        for x in range(width):
            mag = math.hypot(gx[y][x], gy[y][x])
            magnitude[y][x] = mag
            angle[y][x] = math.degrees(math.atan2(gy[y][x], gx[y][x]))
            if mag > max_mag:
                max_mag = mag
    if max_mag == 0:
        max_mag = 1.0
    for y in range(height):
        for x in range(width):
            magnitude[y][x] = magnitude[y][x] / max_mag * 255
    suppressed = non_maximum_suppression(magnitude, angle)
    high = max(max(row) for row in suppressed) * high_ratio
    low = high * low_ratio
    edges = hysteresis_threshold(suppressed, low, high)
    return EdgeResults(magnitude=magnitude, edges=edges)
