"""Otsu thresholding and connected components."""
from __future__ import annotations

from typing import List, Tuple

from utils import zeros


def otsu_threshold(image: List[List[int]]) -> Tuple[int, List[List[int]]]:
    hist = [0 for _ in range(256)]
    total = 0
    for row in image:
        for value in row:
            hist[value] += 1
            total += 1
    sum_total = sum(i * hist[i] for i in range(256))
    sum_b = 0.0
    weight_b = 0.0
    max_var = 0.0
    threshold = 0
    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        between_var = weight_b * weight_f * (mean_b - mean_f) ** 2
        if between_var > max_var:
            max_var = between_var
            threshold = t
    height = len(image)
    width = len(image[0])
    binary = zeros(height, width, 0)
    for y in range(height):
        for x in range(width):
            if image[y][x] >= threshold:
                binary[y][x] = 255
    return threshold, binary


def count_objects(binary: List[List[int]]) -> int:
    height = len(binary)
    width = len(binary[0])
    visited = [[False for _ in range(width)] for _ in range(height)]
    count = 0
    for y in range(height):
        for x in range(width):
            if binary[y][x] == 0 or visited[y][x]:
                continue
            count += 1
            stack = [(y, x)]
            visited[y][x] = True
            while stack:
                cy, cx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if binary[ny][nx] != 0 and not visited[ny][nx]:
                                visited[ny][nx] = True
                                stack.append((ny, nx))
    return count


def connected_components(binary: List[List[int]]) -> Tuple[List[List[int]], int]:
    height = len(binary)
    width = len(binary[0])
    labels = [[0 for _ in range(width)] for _ in range(height)]
    current = 0
    for y in range(height):
        for x in range(width):
            if binary[y][x] == 0 or labels[y][x] != 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y][x] = current
            while stack:
                cy, cx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if binary[ny][nx] != 0 and labels[ny][nx] == 0:
                                labels[ny][nx] = current
                                stack.append((ny, nx))
    return labels, current
