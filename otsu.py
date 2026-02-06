"""
Otsu Thresholding and Connected Components Labeling.

REFERENCIAL TEÓRICO GERAL:
[1] Otsu, N. (1979). "A threshold selection method from gray-level histograms". 
    IEEE Transactions on Systems, Man, and Cybernetics.
[2] Rosenfeld, A., & Pfaltz, J. L. (1966). "Sequential operations in digital picture processing". 
    Journal of the ACM.
"""
from __future__ import annotations

from typing import List, Tuple

from utils import zeros


def otsu_threshold(image: List[List[int]], manual_threshold: int = 0) -> Tuple[int, List[List[int]]]:
    """
    Executa a Binarização (Limiarização) de Otsu.

    REFERENCIAL TEÓRICO:
    [1] Otsu, N. (1979). "A threshold selection method from gray-level histograms".
        IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.

    EXPLICAÇÃO:
    O método de Otsu é um algoritmo não-supervisionado e não-paramétrico que determina o limiar (threshold)
    ótimo global para separar o fundo (background) do objeto (foreground).
    
    Ele assume que a imagem contém duas classes de pixels seguindo uma distribuição bi-modal no histograma.
    O algoritmo busca exaustivamente o valor de limiar 't' (de 0 a 255) que MAXIMIZA a variância 
    entre-classes (between-class variance) ou, equivalentemente, MINIMIZA a variância intra-classe.
    
    Matematicamente, buscamos maximizar: sigma_b^2(t) = w_0(t) * w_1(t) * [mu_0(t) - mu_1(t)]^2
    Onde w são as probabilidades (pesos) das classes e mu são as médias das classes.
    """
    height = len(image)
    width = len(image[0])
    
    # Se o usuário não definiu um threshold manual, calculamos via Otsu
    if manual_threshold <= 0:
        # 1. Cálculo do Histograma (PDF empírica)
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
        
        # 2. Busca exaustiva pelo limiar que maximiza a variância entre classes
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
            
            # Variância Entre-Classes (Between Class Variance)
            between_var = weight_b * weight_f * (mean_b - mean_f) ** 2
            
            if between_var > max_var:
                max_var = between_var
                threshold = t
    else:
        threshold = manual_threshold

    # 3. Aplica a limiarização
    binary = zeros(height, width, 0)
    for y in range(height):
        for x in range(width):
            if image[y][x] >= threshold:
                binary[y][x] = 255
                
    return threshold, binary


def count_objects(binary: List[List[int]], min_area: int = 20) -> int:
    """
    Conta o número de objetos conexos na imagem binária.
    
    REFERENCIAL TEÓRICO:
    [1] Rosenfeld, A., & Pfaltz, J. L. (1966). "Sequential operations in digital picture processing".
        Journal of the ACM, 13(4), 471-494.
    
    EXPLICAÇÃO:
    Utiliza o algoritmo de Rotulação de Componentes Conexos (Connected Component Labeling - CCL)
    baseado em busca em grafo (DFS/BFS).
    
    O algoritmo varre a imagem pixel a pixel. Ao encontrar um pixel de objeto (255) não visitado,
    inicia uma busca (neste caso, usando uma Pilha/Stack para DFS) para encontrar todos os pixels
    vizinhos conectados (usando 8-conectividade). O grupo inteiro é marcado como visitado e 
    contado como 1 objeto.
    
    Filtro de Ruído: O parâmetro 'min_area' descarta componentes com poucos pixels, assumindo
    que são ruídos de sensor ou artefatos de binarização.
    """
    height = len(binary)
    width = len(binary[0])
    visited = [[False for _ in range(width)] for _ in range(height)]
    count = 0
    
    for y in range(height):
        for x in range(width):
            # Procura pixels brancos (255) não visitados
            if binary[y][x] == 0 or visited[y][x]:
                continue
            
            # Novo objeto encontrado, vamos medir o tamanho dele
            pixel_count = 0
            stack = [(y, x)]
            visited[y][x] = True
            
            while stack:
                cy, cx = stack.pop()
                pixel_count += 1
                
                # Conectividade-8 (vizinhos horizontais, verticais e diagonais)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if binary[ny][nx] != 0 and not visited[ny][nx]:
                                visited[ny][nx] = True
                                stack.append((ny, nx))
            
            # SÓ CONTA SE FOR MAIOR QUE O TAMANHO MÍNIMO (filtra ruídos)
            if pixel_count >= min_area:
                count += 1
                
    return count


def connected_components(binary: List[List[int]]) -> Tuple[List[List[int]], int]:
    """
    Rotula os componentes conexos, atribuindo um ID único para cada objeto.

    REFERENCIAL TEÓRICO:
    [1] Rosenfeld, A., & Pfaltz, J. L. (1966). "Sequential operations in digital picture processing".
    [2] Gonzalez, R. C., & Woods, R. E. "Digital Image Processing".
    
    EXPLICAÇÃO:
    Similar à função de contagem, mas gera uma matriz de saída onde cada pixel pertencente
    a um objeto recebe um número inteiro (Label ID: 1, 2, 3...) em vez de apenas contar.
    
    A matriz 'labels' resultante é fundamental para a segmentação Watershed e para a extração
    de características individuais (como a Cadeia de Freeman de um único objeto).
    """
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