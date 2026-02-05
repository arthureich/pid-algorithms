"""Image processing algorithms for the PID assignment (pixel-by-pixel)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import math


@dataclass
class EdgeResults:
    magnitude: List[List[float]]
    edges: List[List[int]]


# References used in this file (for study and algorithm notes):
# - Marr-Hildreth: https://en.wikipedia.org/wiki/Marr%E2%80%93Hildreth_algorithm
# - Canny: https://en.wikipedia.org/wiki/Canny_edge_detector
# - Otsu: https://en.wikipedia.org/wiki/Otsu%27s_method
# - Watershed: https://en.wikipedia.org/wiki/Watershed_(image_processing)
# - Freeman chain code: https://en.wikipedia.org/wiki/Chain_code


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
                
                if iy < 0: iy = 0
                elif iy >= height: iy = height - 1
                
                if ix < 0: ix = 0
                elif ix >= width: ix = width - 1
                
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


def marr_hildreth(image: List[List[int]], sigma: float = 1.4, threshold: float = 0.02) -> EdgeResults:
    log = log_kernel(9, sigma)
    log_response = convolve2d(image, log)
    edges = zero_crossing(log_response, threshold)
    magnitude = [[abs(value) for value in row] for row in log_response]
    return EdgeResults(magnitude=magnitude, edges=edges)


def zero_crossing(response: List[List[float]], threshold: float) -> List[List[int]]:
    height = len(response)
    width = len(response[0])
    edges = zeros(height, width, 0)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            patch = [
                response[y + dy][x + dx]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ]
            min_val = min(patch)
            max_val = max(patch)
            if min_val < 0 < max_val and (max_val - min_val) > threshold:
                edges[y][x] = 255
    return edges


def sobel_gradients(image: List[List[int]]) -> Tuple[List[List[float]], List[List[float]]]:
    kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    gx = convolve2d(image, kx)
    gy = convolve2d(image, ky)
    return gx, gy


def non_maximum_suppression(magnitude: List[List[float]], angle: List[List[float]]) -> List[List[float]]:
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
    stack: List[Tuple[int, int]] = []
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


def canny(image: List[List[int]], low_ratio: float = 0.05, high_ratio: float = 0.15) -> EdgeResults:
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


def freeman_chain_code(binary: List[List[int]]) -> List[int]:
    directions = [
        (0, 1),  # 0
        (-1, 1),  # 1
        (-1, 0),  # 2
        (-1, -1),  # 3
        (0, -1),  # 4
        (1, -1),  # 5
        (1, 0),  # 6
        (1, 1),  # 7
    ]
    start = None
    height = len(binary)
    width = len(binary[0])
    for y in range(height):
        for x in range(width):
            if binary[y][x] != 0:
                start = (y, x)
                break
        if start:
            break
    if start is None:
        return []
    chain = []
    current = start
    prev_dir = 0
    visited_once = False
    while True:
        found = False
        for i in range(8):
            direction = (prev_dir + i) % 8
            dy, dx = directions[direction]
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < height and 0 <= nx < width:
                if binary[ny][nx] != 0:
                    chain.append(direction)
                    current = (ny, nx)
                    prev_dir = (direction + 5) % 8
                    found = True
                    break
        if not found:
            break
        if current == start:
            if visited_once:
                break
            visited_once = True
    return chain


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


def comparison_text() -> str:
    # Comentário solicitado: este texto é usado na interface para comparar os algoritmos.
    return (
        "Marr-Hildreth aplica um filtro LoG (Laplacian of Gaussian) para realçar bordas e, "
        "em seguida, detecta mudanças de sinal (zero-crossing) para marcar contornos. "
        "Canny suaviza a imagem, calcula gradientes, realiza supressão de não-máximos e "
        "usa dupla limiarização com histerese para obter bordas mais finas e conectadas.\n\n"
        "Na prática, Marr-Hildreth tende a gerar bordas mais grossas e sensíveis ao ruído, "
        "enquanto o Canny geralmente produz contornos mais limpos, finos e contínuos, "
        "com melhor controle de falsos positivos via limiares."
    )
