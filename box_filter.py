"""
Box Filter (Mean Filter / Average Filter).

REFERENCIAL TEÓRICO GERAL:
[1] Gonzalez, R. C., & Woods, R. E. (2002). "Digital Image Processing". 
    Prentice Hall. (Capítulo 3: Intensity Transformations and Spatial Filtering).
[2] McDonnell, M. J. (1981). "Box-filtering techniques". 
    Computer Graphics and Image Processing, 17(1), 65-70.

RESUMO:
O filtro de média (Box Filter) é o mais simples dos filtros lineares de suavização (passa-baixa).
Ele substitui o valor de cada pixel pela média aritmética dos valores de intensidade na vizinhança 
definida pela máscara (kernel). É primariamente usado para redução de ruído, embora tenha o 
efeito colateral indesejado de borrar bordas (blurring) mais do que filtros não-lineares ou Gaussianos.
"""
from __future__ import annotations

from typing import List

from utils import convolve2d, zeros


def box_filter(image: List[List[int]], size: int) -> List[List[int]]:
    """
    Aplica um Filtro de Média (Box Filter) na imagem.

    REFERENCIAL TEÓRICO:
    [1] Gonzalez, R. C., & Woods, R. E. "Digital Image Processing".
        Seção: Smoothing Spatial Filters (Linear Filters).

    EXPLICAÇÃO MATEMÁTICA:
    A operação é uma convolução discreta da imagem I com um kernel K de tamanho (m x m).
    
    O kernel K é definido como uma matriz uniforme onde cada coeficiente w_ij é:
        w_ij = 1 / (m * m)
    
    A normalização (divisão pela soma dos pesos, que é a área do kernel) é crucial para:
    1. Manter a energia da imagem constante (o brilho médio da imagem não se altera).
    2. Garantir que regiões de intensidade constante na entrada permaneçam constantes na saída.

    COMPORTAMENTO:
    - Redução de Ruído: Eficaz contra ruído gaussiano ou uniforme, pois a média tende a eliminar variações aleatórias.
    - Suavização: Remove detalhes finos da imagem (altas frequências espaciais).
    """
    
    # 1. Construção do Kernel Normalizado
    # Exemplo para size=3 (3x3): 
    # [[1/9, 1/9, 1/9], 
    #  [1/9, 1/9, 1/9], 
    #  [1/9, 1/9, 1/9]]
    kernel = [[1.0 / (size * size) for _ in range(size)] for _ in range(size)]
    
    # 2. Convolução Espacial
    # Aplica a máscara deslizante sobre a imagem usando a função utilitária
    filtered = convolve2d(image, kernel)
    
    height = len(filtered)
    width = len(filtered[0])
    output = zeros(height, width, 0)
    
    # 3. Quantização e Clipping
    # O resultado da convolução é float (média). Convertemos de volta para uint8.
    for y in range(height):
        for x in range(width):
            value = int(round(filtered[y][x]))
            # Garante que o pixel fique entre 0 e 255 (clipping de segurança)
            output[y][x] = max(0, min(255, value))
            
    return output