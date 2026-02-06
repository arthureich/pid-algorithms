"""
REFERENCIAL TEÓRICO:
[1] Gonzalez, R. C., & Woods, R. E. "Digital Image Processing".
    (Capítulo 3: Intensity Transformations and Spatial Filtering).

Técnica que destaca intervalos específicos de níveis de cinza na imagem. 
Neste caso, faixas de brilho são mapeadas para um único valor.
Isso reduz a quantidade de informação visual, agrupando pixels de intensidades similares.
"""

from __future__ import annotations

from typing import List
from utils import zeros


def intensity_segmentation(image: List[List[int]]) -> List[List[int]]:
    """
    Realiza a Segmentação.

    ALGORITMO:
    A função aplica uma transformação T(r) em cada pixel 'r' da imagem original:
    
    T(r) = {
       25,  se   0 <= r <= 50
       75,  se  51 <= r <= 100
       125, se 101 <= r <= 150
       175, se 151 <= r <= 200
       255, se r > 200
    }
    
    APLICAÇÃO:
    É frequentemente usada para realçar características específicas que residem em uma
    faixa de cinza conhecida, separando-as do fundo ou de outros tecidos.
    """
    height = len(image)
    width = len(image[0])
    result = zeros(height, width, 0)
    
    for y in range(height):
        for x in range(width):
            value = image[y][x]
            
            # Mapeamento de intervalos 
            if 0 <= value <= 50:
                result[y][x] = 25
            elif 51 <= value <= 100:
                result[y][x] = 75
            elif 101 <= value <= 150:
                result[y][x] = 125
            elif 151 <= value <= 200:
                result[y][x] = 175
            else:
                # Valores > 200 
                result[y][x] = 255
                
    return result