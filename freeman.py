"""Freeman chain code extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from otsu import connected_components
from utils import zeros


@dataclass
class FreemanChainResult:
    chain: List[int]
    normalized_chain: List[int]
    first_difference: List[int]
    circular_first_difference: List[int]
    boundary: List[Tuple[int, int]]
    start_point: Tuple[int, int] | None
    component_mask: List[List[int]] | None



def freeman_chain_code(binary: List[List[int]]) -> FreemanChainResult:
    height = len(binary)
    width = len(binary[0])

    freeman_directions = [
        (0, 1),   # 0: Leste
        (-1, 1),  # 1: Nordeste
        (-1, 0),  # 2: Norte
        (-1, -1), # 3: Noroeste
        (0, -1),  # 4: Oeste
        (1, -1),  # 5: Sudoeste
        (1, 0),   # 6: Sul
        (1, 1),   # 7: Sudeste
    ]
    offset_to_code = {offset: idx for idx, offset in enumerate(freeman_directions)}
    clockwise_offsets = [
        (0, -1),   # Oeste
        (-1, -1),  # Noroeste
        (-1, 0),   # Norte
        (-1, 1),   # Nordeste
        (0, 1),    # Leste
        (1, 1),    # Sudeste
        (1, 0),    # Sul
        (1, -1),   # Sudoeste
    ]

    def is_foreground(y: int, x: int) -> bool:
        return 0 <= y < height and 0 <= x < width and binary[y][x] != 0

    def is_border(y: int, x: int) -> bool:
        if not is_foreground(y, x):
            return False
        for dy, dx in clockwise_offsets:
            ny, nx = y + dy, x + dx
            if not is_foreground(ny, nx):
                return True
        return False

    def next_boundary_point(b: Tuple[int, int], c: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]] | None:
        by, bx = b
        cy, cx = c
        offset = (cy - by, cx - bx)
        try:
            start_index = clockwise_offsets.index(offset)
        except ValueError:
            start_index = 0
        previous = c
        for i in range(1, 9):
            dy, dx = clockwise_offsets[(start_index + i) % 8]
            ny, nx = by + dy, bx + dx
            if is_foreground(ny, nx):
                return (ny, nx), previous
            previous = (ny, nx)
        return None

    start: Tuple[int, int] | None = None
    for y in range(height):
        for x in range(width):
            if is_border(y, x):
                start = (y, x)
                break
        if start is not None:
            break

    if start is None:
        return FreemanChainResult(
            chain=[],
            normalized_chain=[],
            first_difference=[],
            circular_first_difference=[],
            boundary=[],
            start_point=None,
            component_mask=None,
        )

    c0 = (start[0], start[1] - 1)
    first_step = next_boundary_point(start, c0)
    if first_step is None:
        return FreemanChainResult(
            chain=[],
            normalized_chain=[],
            first_difference=[],
            circular_first_difference=[],
            boundary=[start],
            start_point=start,
            component_mask=None,
        )
    b1, c1 = first_step
    boundary = [start, b1]
    b = b1
    c = c1
    max_steps = height * width * 4
    steps = 0
    while steps < max_steps:
        steps += 1
        next_step = next_boundary_point(b, c)
        if next_step is None:
            break
        b_next, c_next = next_step
        boundary.append(b_next)
        if b == start and b_next == b1:
            break
        b, c = b_next, c_next

    chain: List[int] = []
    for i in range(1, len(boundary)):
        dy = boundary[i][0] - boundary[i - 1][0]
        dx = boundary[i][1] - boundary[i - 1][1]
        code = offset_to_code.get((dy, dx))
        if code is None:
            continue
        chain.append(code)

    def minimal_rotation(sequence: List[int]) -> List[int]:
        if not sequence:
            return []
        doubled = sequence * 2
        n = len(sequence)
        i = 0
        j = 1
        k = 0
        while i < n and j < n and k < n:
            a = doubled[i + k]
            b_val = doubled[j + k]
            if a == b_val:
                k += 1
                continue
            if a > b_val:
                i = i + k + 1
                if i <= j:
                    i = j + 1
            else:
                j = j + k + 1
                if j <= i:
                    j = i + 1
            k = 0
        start_idx = min(i, j)
        return doubled[start_idx:start_idx + n]
    
    normalized_chain = minimal_rotation(chain)

    def first_difference(sequence: List[int]) -> List[int]:
        if len(sequence) < 2:
            return []
        diffs = []
        for i in range(1, len(sequence)):
            diffs.append((sequence[i] - sequence[i - 1]) % 8)
        return diffs

    first_diff = first_difference(normalized_chain)
    circular_first_diff = first_diff[:]
    if normalized_chain:
        circular_first_diff.append((normalized_chain[0] - normalized_chain[-1]) % 8)

    return FreemanChainResult(
        chain=chain,
        normalized_chain=normalized_chain,
        first_difference=first_diff,
        circular_first_difference=circular_first_diff,
        boundary=boundary,
        start_point=start,
        component_mask=None,
    )

def largest_component_mask(binary: List[List[int]]) -> List[List[int]]:
    labels, count = connected_components(binary)
    if count == 0:
        return zeros(len(binary), len(binary[0]), 0)
    height = len(binary)
    width = len(binary[0])
    sizes = [0 for _ in range(count + 1)]
    for y in range(height):
        for x in range(width):
            label = labels[y][x]
            if label > 0:
                sizes[label] += 1
    largest_label = max(range(1, count + 1), key=lambda idx: sizes[idx])
    mask = zeros(height, width, 0)
    for y in range(height):
        for x in range(width):
            if labels[y][x] == largest_label:
                mask[y][x] = 255
    return mask


def boundary_to_image(boundary: List[Tuple[int, int]], height: int, width: int, value: int = 255) -> List[List[int]]:
    image = zeros(height, width, 0)
    for y, x in boundary:
        if 0 <= y < height and 0 <= x < width:
            image[y][x] = value
    return image


def subsample_boundary(boundary: List[Tuple[int, int]], step: int) -> List[Tuple[int, int]]:
    if step <= 1:
        return boundary[:]
    return boundary[::step]


def _draw_line(image: List[List[int]], start: Tuple[int, int], end: Tuple[int, int], value: int) -> None:
    y0, x0 = start
    y1, x1 = end
    
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    
    y, x = y0, x0
    height = len(image)
    width = len(image[0])

    while True:
        # 1. Desenha o pixel atual se estiver dentro dos limites
        if 0 <= y < height and 0 <= x < width:
            image[y][x] = value
            
        # 2. CONDIÇÃO DE PARADA: Verificação imediata após desenhar
        if y == y1 and x == x1:
            break
            
        # 3. Cálculo do erro e incremento
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
            
        # 4. Segurança extra: se sair dos limites da imagem, para o traçado
        if x < -width or x > 2*width or y < -height or y > 2*height:
            break


def connect_points_image(points: List[Tuple[int, int]], height: int, width: int, value: int = 255) -> List[List[int]]:
    image = zeros(height, width, 0)
    if len(points) < 2:
        return image
    for idx in range(1, len(points)):
        _draw_line(image, points[idx - 1], points[idx], value)
    return image