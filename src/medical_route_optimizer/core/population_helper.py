from random import random
from typing import List, Optional, Tuple
from data.delivery_points import PontoEntrega
import random

# ---------------------------------------------------------------------------
# Geração de população inicial
# ---------------------------------------------------------------------------

def gerar_populacao_aleatoria(
    locais_entrega: List[PontoEntrega],
    tamanho_populacao: int
) -> List[List[PontoEntrega]]:
    """
    Gera uma população de rotas aleatórias por permutação dos locais de entrega.

    O hospital base NÃO é incluído nas rotas — ele é tratado como origem fixa.

    Parâmetros:
    - locais_entrega: pontos de entrega (sem o hospital base)
    - tamanho_populacao: número de rotas a gerar

    Retorno:
    - lista de rotas (cada rota é uma permutação dos locais_entrega)
    """
    return [random.sample(locais_entrega, len(locais_entrega))
            for _ in range(tamanho_populacao)]

# ---------------------------------------------------------------------------
# Ordenação da população
# ---------------------------------------------------------------------------

def ordenar_populacao(
    populacao_rotas: List[List[PontoEntrega]],
    custos: List[float]
) -> Tuple[List[List[PontoEntrega]], List[float]]:
    """
    Ordena a população de rotas do menor custo (melhor) para o maior (pior).

    Parâmetros:
    - populacao_rotas: lista de rotas
    - custos: lista de custos correspondentes a cada rota

    Retorno:
    - tupla (populacao_ordenada, custos_ordenados)
    """
    combinado = sorted(zip(populacao_rotas, custos), key=lambda x: x[1])
    populacao_ordenada, custos_ordenados = zip(*combinado)
    return list(populacao_ordenada), list(custos_ordenados)
