from __future__ import annotations

# Texto da interface para comparar os algoritmos
def comparison_text() -> str:
    return (
        "Marr-Hildreth aplica um filtro LoG (Laplacian of Gaussian) para realçar bordas e, "
        "em seguida, detecta mudanças de sinal (zero-crossing) para marcar contornos. "
        "Canny suaviza a imagem, calcula gradientes, realiza supressão de não-máximos e "
        "usa dupla limiarização com histerese para obter bordas mais finas e conectadas.\n\n"
        "Na prática, Marr-Hildreth tende a gerar bordas mais grossas e sensíveis ao ruído, "
        "enquanto o Canny geralmente produz contornos mais limpos, finos e contínuos, "
        "com melhor controle de falsos positivos via limiares."
    )
