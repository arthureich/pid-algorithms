"""
REFERENCIAL TEÓRICO:
[1] Marr, D., & Hildreth, E. (1980). "Theory of edge detection". 
    Proceedings of the Royal Society of London. Series B. Biological Sciences.

RESUMO:
Este algoritmo propõe que as bordas de intensidade em uma imagem são melhor detectadas procurando
por cruzamentos de zero (zero-crossings) na segunda derivada da imagem suavizada.
A combinação da suavização Gaussiana com o operador Laplaciano resulta no filtro LoG (Laplacian of Gaussian),
que serve como um detector de bordas passa-banda pelo Sigma.
"""
from __future__ import annotations

from typing import List

from utils import EdgeResults, convolve2d, log_kernel, zeros


def zero_crossing(response: List[List[float]], threshold: float) -> List[List[int]]:
    """
    Detecta os cruzamentos por zero na imagem de resposta do Laplaciano.

    ALGORITMO:
    Uma borda (mudança abrupta de intensidade) corresponde a um pico na primeira derivada
    (gradiente) e a um cruzamento por zero na segunda derivada (Laplaciano).
    
    Esta função varre a imagem procurando por mudanças de sinal entre pixels
    vizinhos opostos (horizontal, vertical e diagonais). 
    
    O parâmetro 'threshold' é usado para rejeitar "falsos cruzamentos" causados por ruído em regiões
    uniformes da imagem. Apenas cruzamentos com uma magnitude (diferença de valor) significativa são aceitos.
    """
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
            
            # CASO 1: O pixel atual é um pico POSITIVO forte (> threshold)
            # Procuramos por um vizinho oposto que seja NEGATIVO forte
            if val > threshold:
                for (dy1, dx1), (dy2, dx2) in pairs:
                    if response[y + dy1][x + dx1] < -threshold or \
                       response[y + dy2][x + dx2] < -threshold:
                        is_edge = True
                        break
            
            # CASO 2: O pixel atual é um vale NEGATIVO forte (< -threshold)
            # Procuramos por um vizinho oposto que seja POSITIVO forte
            elif val < -threshold:
                for (dy1, dx1), (dy2, dx2) in pairs:
                    if response[y + dy1][x + dx1] > threshold or \
                       response[y + dy2][x + dx2] > threshold:
                        is_edge = True
                        break

            # CASO 3: O pixel está na "ZONA MORTA" (próximo de zero)
            # Isso acontece quando o cruzamento exato cai entre pixels. 
            # Verificamos se os vizinhos opostos trocam de sinal drasticamente entre si.
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
    threshold: float = 0.5,
    kernel_size: int = 0 
) -> EdgeResults:
    """
    Executa o pipeline completo do Detector de Bordas Marr-Hildreth.

    PARÂMETROS:
    - sigma: Desvio padrão da Gaussiana. Controla a escala.
             Sigma maior = Detecta bordas maiores/grosseiras, ignora detalhes finos.
             Sigma menor = Detecta detalhes finos, mas é mais sensível a ruído.
    - threshold: Limiar para validar o cruzamento por zero.

    FUNCIONAMENTO:
    1. Gera um kernel LoG (Laplacian of Gaussian). O LoG é a segunda derivada de uma Gaussiana.
       Isso combina suavização (para reduzir ruído) e diferenciação (para achar bordas) em um único passo.
       f(x,y) * LoG = (f * G)'' 
    2. Convolução da imagem com este kernel.
    3. Detecção de Zero-Crossing no resultado.
    """
    
    # 1. Definição do Tamanho do Kernel 
    if kernel_size > 0:
        # Se o usuário definiu um tamanho, forçamos que seja ímpar para ter centro exato
        final_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    else:
        # Cálculo automático baseado no Sigma (padrão 3*sigma para cada lado = 6*sigma total)
        final_size = int(sigma * 3) * 2 + 1 
        if final_size % 2 == 0: final_size += 1 

    log = log_kernel(final_size, sigma)
    
    # 2. Aplica Convolução (LoG)
    log_response = convolve2d(image, log)
    
    # 3. Aplica a detecção de bordas (Cruzamento por Zero)
    edges = zero_crossing(log_response, threshold)
    
    # 4. Gera Magnitude 
    magnitude = [[abs(val) for val in row] for row in log_response]
    
    return EdgeResults(magnitude=magnitude, edges=edges)