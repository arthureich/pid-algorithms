"""Marr-Hildreth edge detection."""
from __future__ import annotations

from typing import List

from utils import EdgeResults, convolve2d, log_kernel, zeros


def zero_crossing(response: List[List[float]], threshold: float) -> List[List[int]]:
    # Verifica pares opostos e trata a 'zona morta' (valores próximos de zero)
    height = len(response)
    width = len(response[0])
    # Cria matriz de zeros para as bordas
    edges = [[0 for _ in range(width)] for _ in range(height)]
    
    # Pares de vizinhos opostos para verificar: (dy, dx)
    # Vertical, Horizontal, Diagonal 1 (\), Diagonal 2 (/)
    pairs = [
        ((-1, 0), (1, 0)),   # Cima vs Baixo
        ((0, -1), (0, 1)),   # Esquerda vs Direita
        ((-1, -1), (1, 1)),  # Noroeste vs Sudeste
        ((-1, 1), (1, -1))   # Nordeste vs Sudoeste
    ]

    # Começa de 1 e vai até height-1 para evitar verificar bordas da imagem 
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            val = response[y][x]
            is_edge = False
            
            # O pixel atual é um pico positivo forte
            if val > threshold:
                for (dy1, dx1), (dy2, dx2) in pairs:
                    # Verifica se algum vizinho oposto é negativo forte
                    if response[y + dy1][x + dx1] < -threshold or \
                       response[y + dy2][x + dx2] < -threshold:
                        is_edge = True
                        break
            
            # O pixel atual é um pico negativo forte
            elif val < -threshold:
                for (dy1, dx1), (dy2, dx2) in pairs:
                    # Verifica se algum vizinho oposto é positivo forte
                    if response[y + dy1][x + dx1] > threshold or \
                       response[y + dy2][x + dx2] > threshold:
                        is_edge = True
                        break

            # O pixel está na zona morta (abs(val) <= threshold)
            else:
                for (dy1, dx1), (dy2, dx2) in pairs:
                    p1 = response[y + dy1][x + dx1]
                    p2 = response[y + dy2][x + dx2]
                    
                    if (p1 > threshold and p2 < -threshold) or \
                       (p1 < -threshold and p2 > threshold):
                        is_edge = True
                        break
            
            if is_edge:
                edges[y][x] = 255

    return edges

def marr_hildreth(
    image: List[List[int]],
    sigma: float = 1.4,
    threshold: float = 0.5 
) -> EdgeResults:
    # 1. Gera o Kernel 
    kernel_size = int(sigma * 3) * 2 + 1 
    if kernel_size % 2 == 0: kernel_size += 1 
    log = log_kernel(kernel_size, sigma)
    
    # 2. Aplica Convolução
    log_response = convolve2d(image, log)
    
    # 3. Aplica a detecção 
    edges = zero_crossing(log_response, threshold)
    
    # 4. Gera Magnitude 
    magnitude = [[abs(val) for val in row] for row in log_response]
    return EdgeResults(magnitude=magnitude, edges=edges)