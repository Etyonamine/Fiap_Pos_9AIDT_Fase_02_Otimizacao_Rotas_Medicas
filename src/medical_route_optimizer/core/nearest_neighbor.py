"""
Heurística Construtiva: Vizinho Mais Próximo (Nearest Neighbor) para TSP.

Algoritmo:
1. Inicia no hospital base (ponto de origem).
2. A cada passo, visita o ponto de entrega ainda não visitado mais próximo do atual.
3. Ao esgotar todos os pontos, retorna ao hospital base (ciclo fechado).

Complexidade: O(n²)
Uso no pipeline:
    - Gera soluções iniciais de qualidade superior às puramente aleatórias.
    - Inicializa parte da população do GA para acelerar a convergência.
    - Serve como baseline de comparação com a solução final do GA + Two Opt.
"""

from typing import List, Tuple

from data.delivery_points import PontoEntrega
from core.route_calculator import calcular_distancia
from core.fitness_calculator import calcular_custo_rota

def nearest_neighbor(
    locais_entrega: List[PontoEntrega],
    hospital_base: PontoEntrega
) -> List[PontoEntrega]:
    """
    Constrói uma rota pelo método do Vizinho Mais Próximo.

    Parâmetros:
    - locais_entrega: pontos de entrega a visitar (sem o hospital base)
    - hospital_base: ponto de partida e retorno

    Retorno:
    - rota ordenada de pontos de entrega (sem o hospital base nos extremos)
    """
    nao_visitados = list(locais_entrega)
    rota: List[PontoEntrega] = []
    ponto_atual: PontoEntrega = hospital_base

    while nao_visitados:
        mais_proximo = min(nao_visitados,
                           key=lambda p: calcular_distancia(ponto_atual, p))
        rota.append(mais_proximo)
        nao_visitados.remove(mais_proximo)
        ponto_atual = mais_proximo

    return rota


def gerar_populacao_nearest_neighbor(
    locais_entrega: List[PontoEntrega],
    hospital_base: PontoEntrega,
    n_solucoes: int = 1
) -> List[List[PontoEntrega]]:
    """
    Gera múltiplas soluções via Nearest Neighbor com pontos de partida variados.

    Para n_solucoes > 1, utiliza pontos de entrega aleatórios como ponto de
    partida interno (após sair do hospital base), gerando variações do NN.

    Parâmetros:
    - locais_entrega: pontos de entrega (sem o hospital base)
    - hospital_base: ponto de origem e retorno
    - n_solucoes: número de soluções NN a gerar

    Retorno:
    - lista de rotas geradas pelo método NN
    """
    import random

    solucoes: List[List[PontoEntrega]] = []

    # Primeira solução: NN padrão iniciando do hospital base
    solucoes.append(nearest_neighbor(locais_entrega, hospital_base))

    # Demais soluções: NN iniciando de pontos de partida internos aleatórios
    for _ in range(n_solucoes - 1):
        ponto_partida = random.choice(locais_entrega)
        nao_visitados = [p for p in locais_entrega if p != ponto_partida]
        rota = [ponto_partida]
        ponto_atual = ponto_partida

        while nao_visitados:
            mais_proximo = min(nao_visitados,
                               key=lambda p: calcular_distancia(ponto_atual, p))
            rota.append(mais_proximo)
            nao_visitados.remove(mais_proximo)
            ponto_atual = mais_proximo

        solucoes.append(rota)

    return solucoes


def avaliar_baseline_nn(
    locais_entrega: List[PontoEntrega],
    hospital_base: PontoEntrega
) -> Tuple[List[PontoEntrega], float]:
    """
    Retorna a rota NN padrão e seu custo, para uso como baseline de comparação.

    Parâmetros:
    - locais_entrega: pontos de entrega (sem o hospital base)
    - hospital_base: ponto de origem e retorno

    Retorno:
    - (rota_nn, custo_nn)
    """
    rota_nn = nearest_neighbor(locais_entrega, hospital_base)
    custo_nn = calcular_custo_rota(rota_nn, hospital_base)
    return rota_nn, custo_nn
