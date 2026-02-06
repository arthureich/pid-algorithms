"""
REFERENCIAL TEÓRICO:
[1] Canny, J. (1986). "A Computational Approach to Edge Detection". 
    IEEE Transactions on Pattern Analysis and Machine Intelligence.

Objetivos:
1. Baixa taxa de erro: Todas as bordas devem ser encontradas e não deve haver respostas falsas.
2. Boa localização: A distância entre os pontos de borda encontrados e a borda real deve ser mínima.
3. Resposta única: Uma borda real não deve resultar em múltiplos pixels de borda (bordas finas).
"""
from __future__ import annotations

from typing import List
import math
from utils import EdgeResults, convolve2d, gaussian_kernel, sobel_gradients, zeros


def non_maximum_suppression(
    magnitude: List[List[float]],
    angle: List[List[float]],
) -> List[List[float]]:
    """
    Aplica a Supressão de Não-Máximos (Non-Maximum Suppression - NMS).

    Este passo é responsável por "afinar" as bordas. O operador Sobel gera bordas grossas e borradas.
    O NMS verifica, para cada pixel, se a sua magnitude é um máximo local na direção do gradiente.
    
    Se o pixel for o "pico" da montanha na direção da borda, ele é mantido.
    Se houver um vizinho maior na direção do gradiente, o pixel atual é suprimido (zerado).
    As direções são quantizadas em 4 setores: 0° (horizontal), 45° (diagonal /), 90° (vertical), 135° (diagonal \).
    """
    height = len(magnitude)
    width = len(magnitude[0])
    suppressed = [[0.0 for _ in range(width)] for _ in range(height)]
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            direction = angle[y][x] % 180
            q = 0.0
            r = 0.0
            
            # 1. Direção Horizontal (0°) - Verifica vizinhos Esquerda e Direita
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                q = magnitude[y][x + 1]
                r = magnitude[y][x - 1]
            # 2. Direção Diagonal / (45°) - Verifica Nordeste e Sudoeste
            elif 22.5 <= direction < 67.5:
                q = magnitude[y + 1][x - 1]
                r = magnitude[y - 1][x + 1]
            # 3. Direção Vertical (90°) - Verifica Cima e Baixo
            elif 67.5 <= direction < 112.5:
                q = magnitude[y + 1][x]
                r = magnitude[y - 1][x]
            # 4. Direção Diagonal \ (135°) - Verifica Noroeste e Sudeste
            elif 112.5 <= direction < 157.5:
                q = magnitude[y - 1][x - 1]
                r = magnitude[y + 1][x + 1]

            # Só mantém o pixel se ele for maior ou igual aos seus vizinhos na direção do gradiente
            if magnitude[y][x] >= q and magnitude[y][x] >= r:
                suppressed[y][x] = magnitude[y][x]
            else:
                suppressed[y][x] = 0.0
                
    return suppressed


def hysteresis_threshold(image: List[List[float]], low: float, high: float) -> List[List[int]]:
    """
    Resolve o problema de bordas quebradas (streaking) comum em limiarização simples.
    Utiliza dois limiares (T_high e T_low):
    
    1. Bordas Fortes: Pixels > T_high. São aceitos imediatamente como borda final.
    2. Bordas Fracas: Pixels entre T_low e T_high. São candidatos.
    3. Ruído: Pixels < T_low são descartados.
    
    Conectividade: Uma "Borda Fraca" só é promovida a "Borda Forte" se estiver conectada 
    (por vizinhança-8) a uma borda forte. Isso preserva linhas contínuas fracas que estão 
    ligadas a linhas fortes, descartando ruídos isolados.
    """
    height = len(image)
    width = len(image[0])
    strong = 255
    weak = 75
    result = zeros(height, width, 0)
    stack = []
    
    # Fase 1: Identificar bordas fortes e fracas
    for y in range(height):
        for x in range(width):
            value = image[y][x]
            if value >= high:
                result[y][x] = strong
                stack.append((y, x)) 
            elif value >= low:
                result[y][x] = weak
    
    # Fase 2: Rastreamento de bordas 
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if result[ny][nx] == weak:
                        result[ny][nx] = strong
                        stack.append((ny, nx))
    
    # Fase 3: Limpeza (Descarta bordas fracas que não foram conectadas)
    for y in range(height):
        for x in range(width):
            if result[y][x] != strong:
                result[y][x] = 0
                
    return result


def canny(
    image: List[List[int]],
    sigma: float = 1.0,
    kernel_size: int = 5,
    low_ratio: float = 0.05,
    high_ratio: float = 0.15,
) -> EdgeResults:
    """
    Pipeline completo do algoritmo Canny.
    
    Passos:
    1. Suavização Gaussiana (Redução de Ruído).
    2. Cálculo do Gradiente (Sobel) para obter Magnitude e Direção.
    3. Supressão de Não-Máximos (Afinamento).
    4. Limiarização.
    """
    
    # 1. Suavização Gaussiana 
    blurred = convolve2d(image, gaussian_kernel(kernel_size, sigma))
    
    # 2. Gradientes Sobel
    # Aproximação da derivada parcial primeira em X e Y
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
        
    # Normalização da magnitude para 0-255 
    for y in range(height):
        for x in range(width):
            magnitude[y][x] = magnitude[y][x] / max_mag * 255
            
    # 3. Supressão de Não-Máximos
    suppressed = non_maximum_suppression(magnitude, angle)
    
    # Recalcula max após supressão para definir thresholds relativos
    max_suppressed = max(max(row) for row in suppressed) if suppressed else 1.0
    
    # low_ratio e high_ratio são porcentagens do valor máximo encontrado
    high = max_suppressed * high_ratio
    low = max_suppressed * low_ratio
    
    # 4. Histerese
    edges = hysteresis_threshold(suppressed, low, high)
    
    return EdgeResults(magnitude=magnitude, edges=edges)