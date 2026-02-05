"""Watershed segmentation."""
from __future__ import annotations

from typing import List, Tuple

import math

from otsu import connected_components, otsu_threshold
from utils import sobel_gradients


def watershed_segment(image: List[List[int]]) -> List[List[Tuple[int, int, int]]]:
    _, binary = otsu_threshold(image)
    labels, count = connected_components(binary)
    if count == 0:
        height = len(image)
        width = len(image[0])
        return [[(pixel, pixel, pixel) for pixel in row] for row in image]
    gx, gy = sobel_gradients(image)
    height = len(image)
    width = len(image[0])
    gradient = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            gradient[y][x] = math.hypot(gx[y][x], gy[y][x])
    import heapq

    markers = [[labels[y][x] for x in range(width)] for y in range(height)]
    heap: List[Tuple[float, int, int]] = []
    for y in range(height):
        for x in range(width):
            if markers[y][x] > 0:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if markers[ny][nx] == 0:
                                heapq.heappush(heap, (gradient[ny][nx], ny, nx))
    while heap:
        _, y, x = heapq.heappop(heap)
        if markers[y][x] != 0:
            continue
        neighbor_labels = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if markers[ny][nx] > 0:
                        neighbor_labels.add(markers[ny][nx])
        if len(neighbor_labels) == 1:
            markers[y][x] = neighbor_labels.pop()
        elif len(neighbor_labels) > 1:
            markers[y][x] = -1
        if markers[y][x] >= 0:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if markers[ny][nx] == 0:
                            heapq.heappush(heap, (gradient[ny][nx], ny, nx))
    result: List[List[Tuple[int, int, int]]] = []
    for y in range(height):
        row: List[Tuple[int, int, int]] = []
        for x in range(width):
            if markers[y][x] == -1:
                row.append((255, 0, 0))
            else:
                value = image[y][x]
                row.append((value, value, value))
        result.append(row)
    return result
