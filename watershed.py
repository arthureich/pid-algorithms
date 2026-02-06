"""
Watershed Controlado por Marcadores (Baseado em Transformada de Distância).

REFERÊNCIA GERAL DO ALGORITMO:
Esta implementação segue a abordagem híbrida de segmentação morfológica descrita por:
[1] Beucher, S. "The Watershed Transformation applied to image segmentation", Scanning Microscopy International, 1992.
    - Introduziu o conceito de "Watershed Controlado por Marcadores" para evitar a super-segmentação.
[2] Gonzalez, R. C., & Woods, R. E. "Digital Image Processing".
    - Capítulo sobre Segmentação Morfológica: descreve o uso da Transformada de Distância como
      superfície topográfica para separar objetos convexos sobrepostos.
"""
from __future__ import annotations

from typing import List, Tuple
from collections import deque
import heapq
import random

from otsu import otsu_threshold, connected_components
from utils import zeros

def manual_distance_transform(binary: List[List[int]]) -> List[List[int]]:
    """
    Calcula a Transformada de Distância (Distance Transform) via propagação (BFS).

    REFERENCIAL TEÓRICO:
    [1] Rosenfeld, A., & Pfaltz, J. L. (1968). "Distance functions on digital pictures". 
        Pattern Recognition, 1(1), 33-61.
    
    EXPLICAÇÃO:
    A Transformada de Distância converte uma imagem binária em uma imagem em tons de cinza, 
    onde o valor de cada pixel representa a distância Euclidiana (ou Manhattan/Chessboard, 
    dependendo da métrica) até o pixel de fundo (zero) mais próximo.
    
    Neste algoritmo, ela é fundamental pois cria o "relevo topográfico". O centro dos objetos
    terá os valores mais altos (picos), servindo como as "bacias profundas" onde a água (labels)
    começará a subir.
    """
    height = len(binary)
    width = len(binary[0])
    dist = [[float('inf') for _ in range(width)] for _ in range(height)]
    queue = deque()

    # Inicializa a fila com pixels de fundo (0) - Pontos de partida da propagação
    for y in range(height):
        for x in range(width):
            if binary[y][x] == 0:
                dist[y][x] = 0
                queue.append((y, x))
    
    # Métrica: Manhattan (4-conectado) para eficiência em Python puro
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Algoritmo de Grassfire / Breadth-First Search (BFS)
    while queue:
        y, x = queue.popleft()
        current_dist = dist[y][x]
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if dist[ny][nx] == float('inf'):
                    dist[ny][nx] = current_dist + 1
                    queue.append((ny, nx))
    return dist

def find_markers(dist_map: List[List[int]], threshold: int = 3) -> Tuple[List[List[int]], int]:
    """
    Extrai os Marcadores (Sementes) baseando-se nos máximos regionais da distância.

    REFERENCIAL TEÓRICO:
    [1] Beucher, S., & Meyer, F. (1993). "The morphological approach to segmentation: 
        The watershed transformation". Mathematical Morphology in Image Processing.
    [2] Soille, P. (2013). "Morphological image analysis: principles and applications". Springer.

    EXPLICAÇÃO:
    O algoritmo Watershed puro tende a "super-segmentar" (oversegmentation) devido a ruídos locais.
    Para corrigir isso, Beucher introduziu o conceito de "Marcadores".
    
    Esta função identifica os "picos" da transformada de distância (os centros dos objetos).
    Utilizamos a técnica de Componentes Conexos (Rosenfeld et al., 1966) nos picos para garantir 
    que um topo plano (plateau) seja considerado apenas UM marcador, e não vários pixels isolados,
    garantindo que cada objeto receba apenas uma cor/rótulo final.
    """
    height = len(dist_map)
    width = len(dist_map[0])
    
    # 1. Limiarização Regional (H-maxima transform simplificada)
    # Identifica regiões que são picos locais significativos
    peaks_binary = zeros(height, width, 0)
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            val = dist_map[y][x]
            if val < threshold: continue
            
            is_max = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dist_map[y + dy][x + dx] > val:
                        is_max = False
                        break
                if not is_max: break
            
            if is_max:
                peaks_binary[y][x] = 255

    # 2. Rotulação de Componentes Conexos
    # Agrupa pixels vizinhos de pico em um único marcador (Label único)
    markers, count = connected_components(peaks_binary)
    
    return markers, count

def watershed_segment(image: List[List[int]]) -> Tuple[
    List[List[Tuple[int, int, int]]], 
    List[List[int]],                  
    List[List[Tuple[int, int, int]]]  
]:
    """
    Executa a Segmentação Watershed por Imersão Controlada.

    REFERENCIAL TEÓRICO:
    [1] Vincent, L., & Soille, P. (1991). "Watersheds in digital spaces: an efficient algorithm 
        based on immersion simulations". IEEE Transactions on Pattern Analysis and Machine Intelligence.
    [2] Otsu, N. (1979). "A threshold selection method from gray-level histograms".

    EXPLICAÇÃO:
    Esta função orquestra o pipeline completo:
    1. Otsu (1979): Separa o Foreground do Background.
    2. Distance Transform (1968): Cria a topografia onde o centro do objeto é o ponto mais fundo.
    3. Marker Extraction (1992): Define onde a "água" começa a subir.
    4. Immersion Simulation (Vincent & Soille, 1991): Simula a inundação usando uma 
       Fila de Prioridade (Heap). A água sobe dos marcadores (distância máxima) para as bordas.
       Onde águas de marcadores diferentes se encontram, constrói-se uma linha de divisão (Dam).
    """
    
    # 1. Binarização (Otsu - 1979)
    _, binary = otsu_threshold(image)
    h, w = len(binary), len(binary[0])
    
    # Heurística para garantir fundo preto e objeto branco
    corners = [binary[0][0], binary[0][w-1], binary[h-1][0], binary[h-1][w-1]]
    if corners.count(255) >= 3:
         binary = [[255 - val for val in row] for row in binary]

    # 2. Transformada de Distância (Rosenfeld & Pfaltz - 1968)
    dist_map = manual_distance_transform(binary)

    # Visualização auxiliar da Distância
    max_dist = 0
    for row in dist_map:
        max_dist = max(max_dist, max(row))
    dist_visual = []
    scale = 255.0 / max_dist if max_dist > 0 else 0
    for y in range(h):
        row_vis = []
        for x in range(w):
            val = int(dist_map[y][x] * scale)
            row_vis.append(val)
        dist_visual.append(row_vis)

    # 3. Extração de Marcadores (Beucher - 1992)
    # Threshold=5 ajustado empiricamente para evitar ruídos pequenos
    markers, num_markers = find_markers(dist_map, threshold=5) 

    # Visualização auxiliar dos Marcadores
    markers_visual = []
    for y in range(h):
        row_vis = []
        for x in range(w):
            if markers[y][x] > 0:
                row_vis.append((255, 0, 0))
            else:
                row_vis.append((0, 0, 0))
        markers_visual.append(row_vis)
    
    # 4. Simulação de Imersão (Vincent & Soille - 1991)
    labels = [[0 for _ in range(w)] for _ in range(h)]
    pq = []
    
    # Inicializa a Fila de Prioridade com os Marcadores
    # Usamos -dist_map para simular Max-Heap com a biblioteca heapq (que é Min-Heap)
    # Isso garante que processamos do centro do objeto (maior distância) para fora.
    for y in range(h):
        for x in range(w):
            if markers[y][x] > 0:
                labels[y][x] = markers[y][x]
                # Adiciona vizinhos à fila para iniciar a expansão
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and labels[ny][nx] == 0:
                        heapq.heappush(pq, (-dist_map[ny][nx], ny, nx))
            
            # Marca o fundo como visitado (-1) para conter a inundação dentro do objeto
            if binary[y][x] == 0:
                labels[y][x] = -1

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # Loop de Inundação
    while pq:
        d, y, x = heapq.heappop(pq)
        
        if labels[y][x] != 0: continue
            
        # Votação de vizinhos: Quem está inundando este pixel?
        adj_labels = []
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                lbl = labels[ny][nx]
                if lbl > 0: adj_labels.append(lbl)
        
        if not adj_labels: continue
            
        # Se todos os vizinhos pertencem à mesma bacia, expande a região
        if all(L == adj_labels[0] for L in adj_labels):
            labels[y][x] = adj_labels[0]
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and labels[ny][nx] == 0:
                    heapq.heappush(pq, (-dist_map[ny][nx], ny, nx))
        else:
            # Se há vizinhos de bacias diferentes, temos um encontro de águas.
            # Define-se como 0 (Watershed Line / Dique)
            labels[y][x] = 0

    # 5. Composição da Imagem Final
    colors = {}
    for i in range(1, num_markers + 1):
        # Cores aleatórias para distinguir as regiões segmentadas
        colors[i] = (random.randint(50, 255), random.randint(50, 200), random.randint(100, 255))
    
    result_img = []
    for y in range(h):
        row = []
        for x in range(w):
            lbl = labels[y][x]
            if lbl > 0:
                row.append(colors[lbl]) # Objeto segmentado
            elif lbl == 0 and binary[y][x] != 0: 
                row.append((255, 0, 0)) # Linha de Divisão (Vermelho)
            else:
                row.append((0, 0, 0))   # Fundo
        result_img.append(row)
        
    return result_img, dist_visual, markers_visual