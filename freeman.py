"""
REFERENCIAL TEÓRICO:
[1] Gonzalez, R. C., & Woods, R. E. "Digital Image Processing". 
    (Capítulo 11: Representation and Description).

RESUMO:
O Código de Cadeia de Freeman representa um contorno como uma sequência de números inteiros,
onde cada número denota a direção do próximo pixel da borda em relação ao atual.
Para conectividade-8, usam-se códigos de 0 a 7.
Esta técnica é fundamental para compressão de dados de forma e análise de descritores (área, perímetro, curvatura).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    """
    Extrai o contorno usando o algoritmo de Moore-Neighbor Tracing e gera a Cadeia de Freeman.

    Para gerar a cadeia, primeiro precisamos ordenar os pixels da borda.
    O algoritmo "Moore-Neighbor Tracing" funciona assim:
    1. Encontre um pixel inicial de borda (start).
    2. Defina uma direção de entrada (backtrack).
    3. Varra os 8 vizinhos em sentido horário até encontrar o próximo pixel de borda (foreground).
    4. Mova-se para esse pixel e repita até voltar ao início.
    
    A sequência de movimentos (direções) forma a Cadeia de Freeman bruta.
    """
    height = len(binary)
    width = len(binary[0])

    # Direções para conectividade-8 
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
    # Mapeia deslocamento (dy, dx) -> Código (0-7)
    offset_to_code = {offset: idx for idx, offset in enumerate(freeman_directions)}
    
    # Offsets para o Moore Tracing
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
        """Verifica se um pixel de foreground tem pelo menos um vizinho de fundo (4-vizinhos)."""
        if not is_foreground(y, x):
            return False
        # Checagem simplificada de borda
        for dy, dx in clockwise_offsets:
            ny, nx = y + dy, x + dx
            if not is_foreground(ny, nx):
                return True
        return False

    def next_boundary_point(b: Tuple[int, int], c: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]] | None:
        """Encontra o próximo pixel da borda usando varredura horária (Moore)."""
        by, bx = b
        cy, cx = c 
        
        # Começa a busca a partir da direção de onde veio
        offset = (cy - by, cx - bx)
        try:
            start_index = clockwise_offsets.index(offset)
        except ValueError:
            start_index = 0
            
        previous = c
        # Tenta os 8 vizinhos em sentido horário
        for i in range(1, 9):
            dy, dx = clockwise_offsets[(start_index + i) % 8]
            ny, nx = by + dy, bx + dx
            if is_foreground(ny, nx):
                return (ny, nx), previous
            previous = (ny, nx)
        return None

    # 1. Encontra ponto inicial
    start: Tuple[int, int] | None = None
    for y in range(height):
        for x in range(width):
            if is_border(y, x):
                start = (y, x)
                break
        if start is not None:
            break

    if start is None:
        return FreemanChainResult([], [], [], [], [], None, None)

    # 2. Tracing
    c0 = (start[0], start[1] - 1) 
    first_step = next_boundary_point(start, c0)
    if first_step is None:
        return FreemanChainResult([], [], [], [], [start], start, None)
        
    b1, c1 = first_step
    boundary = [start, b1]
    b = b1
    c = c1
    
    # Limite para evitar loops infinitos 
    max_steps = height * width * 4 
    steps = 0
    
    while steps < max_steps:
        steps += 1
        next_step = next_boundary_point(b, c)
        if next_step is None:
            break
        b_next, c_next = next_step
        boundary.append(b_next)
        
        # Critério de parada de Jacob (revisitar o início da mesma maneira)
        if b == start and b_next == b1:
            break
        b, c = b_next, c_next

    # 3. Conversão para Códigos de Cadeia
    chain: List[int] = []
    for i in range(1, len(boundary)):
        dy = boundary[i][0] - boundary[i - 1][0]
        dx = boundary[i][1] - boundary[i - 1][1]
        code = offset_to_code.get((dy, dx))
        if code is None:
            continue
        chain.append(code)

    # 4. Normalização - Torna o código invariante ao ponto de partida
    def minimal_rotation(sequence: List[int]) -> List[int]:
        if not sequence: return []
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
                if i <= j: i = j + 1
            else:
                j = j + k + 1
                if j <= i: j = i + 1
            k = 0
        start_idx = min(i, j)
        return doubled[start_idx:start_idx + n]
    
    normalized_chain = minimal_rotation(chain)

    # 5. Primeira Diferença - Torna o código invariante à rotação
    def first_difference(sequence: List[int]) -> List[int]:
        if len(sequence) < 2: return []
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
    """Retorna uma máscara contendo apenas o maior objeto conexo."""
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
    """Plota os pontos da fronteira em uma imagem preta."""
    image = zeros(height, width, 0)
    for y, x in boundary:
        if 0 <= y < height and 0 <= x < width:
            image[y][x] = value
    return image


def subsample_boundary(boundary: List[Tuple[int, int]], step: int) -> List[Tuple[int, int]]:
    """Subamostragem simples (pula 'step' pixels)."""
    if step <= 1:
        return boundary[:]
    return boundary[::step]

# --- ALGORITMO DE SIMPLIFICAÇÃO ---

def _perpendicular_distance(point: Tuple[int, int], start: Tuple[int, int], end: Tuple[int, int]) -> float:
    """Calcula a distância perpendicular de um ponto a uma reta."""
    if start == end:
        return math.sqrt((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2)
    y0, x0 = point
    y1, x1 = start
    y2, x2 = end
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    if denominator == 0: return 0.0
    return numerator / denominator

def _rdp(points: List[Tuple[int, int]], epsilon: float) -> List[Tuple[int, int]]:
    """
    Algoritmo Ramer-Douglas-Peucker (RDP).
    
    REFERENCIAL TEÓRICO:
    [1] Douglas, D. H., & Peucker, T. K. (1973). "Algorithms for the reduction of the number of 
        points required to represent a digitized line or its caricature".

    EXPLICAÇÃO:
    Reduz o número de pontos de uma curva aproximando-a por uma série de segmentos de reta,
    mantendo a distorção abaixo de um limite (epsilon).
    É uma abordagem recursiva que encontra o ponto mais distante da corda e divide a curva ali.
    """
    if len(points) < 3: return points[:]
    start = points[0]
    end = points[-1]
    max_distance = -1.0
    index = 0
    for i in range(1, len(points) - 1):
        distance = _perpendicular_distance(points[i], start, end)
        if distance > max_distance:
            index = i
            max_distance = distance
    if max_distance <= epsilon:
        return [start, end]
    left = _rdp(points[: index + 1], epsilon)
    right = _rdp(points[index:], epsilon)
    return left[:-1] + right

def subsample_boundary_grid(boundary: List[Tuple[int, int]], step: int) -> List[Tuple[int, int]]:
    """
    Quantização por Grade (Grid Quantization).

    REFERENCIAL TEÓRICO:
    [1] Pavlidis, T. (1982). "Algorithms for Graphics and Image Processing".
    
    EXPLICAÇÃO:
    Reduz a resolução espacial da curva sobrepondo uma grade grossa (tamanho 'step') sobre a imagem.
    Sempre que a fronteira cruza de uma célula da grade para outra, registra-se um vértice.
    Isso gera uma versão "pixelada" em baixa resolução do contorno, ideal para gerar 
    Cadeias de Freeman curtas e representativas.
    """
    if not boundary or step <= 0:
        return []

    simplified_points = []
    last_grid_pos = None
    start_y, start_x = boundary[0]
    offset = step // 2 
    
    for y, x in boundary:
        # Coordenada na Macro Grade
        gy = y // step
        gx = x // step
        current_grid_pos = (gy, gx)

        # Detecta transição de célula
        if current_grid_pos != last_grid_pos:
            pixel_y = (gy * step) + offset
            pixel_x = (gx * step) + offset
            simplified_points.append((pixel_y, pixel_x))
            last_grid_pos = current_grid_pos

    # Garante fechamento do polígono
    if len(simplified_points) > 2:
        if simplified_points[0] != simplified_points[-1]:
             simplified_points.append(simplified_points[0])

    return simplified_points


def get_freeman_from_points(points: List[Tuple[int, int]]) -> List[int]:
    """
    Calcula a Cadeia de Freeman a partir de vértices esparsos.
    As direções são calculadas trigonometricamente (atan2) e quantizadas em 8 setores (45 graus).
    """
    chain = []
    if len(points) < 2: return []

    def get_code(dy, dx):
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0: angle += 360
        # Divide por 45 e arredonda para pegar o setor (0 a 7)
        return int((angle + 22.5) // 45) % 8

    for i in range(1, len(points)):
        p_prev = points[i-1]
        p_curr = points[i]
        dy = p_curr[0] - p_prev[0]
        dx = p_curr[1] - p_prev[1]
        if dy == 0 and dx == 0: continue
        code = get_code(dy, dx)
        chain.append(code)
        
    return chain

# --- UTILITÁRIOS DE DESENHO (Rasterização de Linha) ---

def _draw_line(image: List[List[int]], start: Tuple[int, int], end: Tuple[int, int], value: int) -> None:
    """
    Algoritmo incremental para rasterização de linhas que utiliza apenas aritmética inteira 
    (somas e subtrações), usado para conectar os pontos da fronteira simplificada.
    """
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
        if 0 <= y < height and 0 <= x < width:
            image[y][x] = value
            
        if y == y1 and x == x1: break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
            
        if x < -width or x > 2*width or y < -height or y > 2*height: break

def connect_points_image(
    points: List[Tuple[int, int]],
    height: int,
    width: int,
    value: int = 255,
    close: bool = True,
) -> List[List[int]]:
    """Conecta uma lista de pontos com linhas retas (Polígono)."""
    image = zeros(height, width, 0)
    if len(points) < 2:
        return image
    for idx in range(1, len(points)):
        _draw_line(image, points[idx - 1], points[idx], value)
    if close and len(points) > 2:
        _draw_line(image, points[-1], points[0], value)
    return image