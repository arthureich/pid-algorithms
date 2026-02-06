"""
REFERENCIAL TEÓRICO:
Esta implementação segue a abordagem híbrida de segmentação morfológica descrita por:
[1] PAZOTI, M. A., BRUNO, O. M. . "Aplicação da transformada watershed no processo de separação de objetos". 2004.
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
    Calcula a Transformada de Distância via propagação (BFS).
    
    ALGORITMO:
    A Transformada de Distância converte uma imagem binária em uma imagem em tons de cinza, 
    onde o valor de cada pixel representa a distância Euclidiana até o pixel de fundo mais próximo.
    
    O centro dos objetos terá os valores mais altos, servindo como as "bacias profundas" onde os labels começarão a subir.
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
    
    # Métrica: Manhattan (4-conectado) 
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

    Esta função identifica os "picos" da transformada de distância.
    """
    height = len(dist_map)
    width = len(dist_map[0])
    
    # 1. Limiarização Regional. Identifica regiões que são picos locais significativos
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

    # 2. Rotulação de Componentes Conexos. Agrupa pixels vizinhos de pico em um único label
    markers, count = connected_components(peaks_binary)
    
    return markers, count

def watershed_segment(image: List[List[int]]) -> Tuple[
    List[List[Tuple[int, int, int]]], 
    List[List[int]],                  
    List[List[Tuple[int, int, int]]]  
]:
    """
    Executa a Segmentação Watershed 

    Esta função orquestra o pipeline completo:
    1. Otsu: Separa o Foreground do Background.
    2. Distance Transform: Cria a topografia onde o centro do objeto é o ponto mais fundo.
    3. Marker Extraction: Define onde a "água" começa a subir.
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

    # 2. Transformada de Distância 
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

    # 3. Extração de Marcadores 
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
    
    # 4. Simulação de Imersão 
    labels = [[0 for _ in range(w)] for _ in range(h)]
    pq = []
    
    # Inicializa a Fila de Prioridade com os Marcadores
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
                row.append(colors[lbl]) 
            elif lbl == 0 and binary[y][x] != 0: 
                row.append((255, 0, 0)) 
            else:
                row.append((0, 0, 0))   
        result_img.append(row)
        
    return result_img, dist_visual, markers_visual