"""Freeman chain code extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FreemanChainResult:
    chain: List[int]
    normalized_chain: List[int]
    first_difference: List[int]
    circular_first_difference: List[int]
    boundary: List[Tuple[int, int]]
    start_point: Tuple[int, int] | None



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
    )