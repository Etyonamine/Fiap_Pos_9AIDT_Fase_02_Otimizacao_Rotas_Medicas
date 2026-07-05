import math
from typing import List, Optional, Tuple
from data.delivery_points import PontoEntrega

# ---------------------------------------------------------------------------
# Distância
# ---------------------------------------------------------------------------

def calcular_distancia(p1: PontoEntrega, p2: PontoEntrega) -> float:
    """Calcula a distância Euclidiana entre dois pontos de entrega."""
    return math.sqrt((p1.coords[0] - p2.coords[0]) ** 2 +
                     (p1.coords[1] - p2.coords[1]) ** 2)

def calcular_distancia_rota(
    rota: List[PontoEntrega],
    hospital_base: PontoEntrega,
) -> float:
    """
    Calcula somente a distância Euclidiana total do ciclo fechado (sem penalidades).

    Útil para verificar restrição de autonomia separadamente do custo fitness.

    Parâmetros:
    - rota: sequência de pontos de entrega (sem o hospital base)
    - hospital_base: ponto de origem e retorno

    Retorno:
    - distância total do ciclo em pixels
    """
    rota_completa = [hospital_base] + list(rota) + [hospital_base]
    return sum(
        calcular_distancia(rota_completa[i], rota_completa[i + 1])
        for i in range(len(rota_completa) - 1)
    )
