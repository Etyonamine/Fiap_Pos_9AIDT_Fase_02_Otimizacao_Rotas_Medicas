"""
Algoritmo Genético para Otimização de Rotas Médicas (TSP).

Nomenclatura alinhada ao domínio de negócio:
    - population     → populacao_rotas
    - individual     → rota
    - cities         → locais_entrega
    - fitness        → custo_rota

O hospital base (ponto de origem) é fixo: não entra na permutação.
Ele é adicionado implicitamente no início e no fim ao calcular o custo da rota,
garantindo que toda rota seja um ciclo fechado com origem e retorno ao hospital.

Função de custo (fitness):
    custo = distancia_total + penalidade_prioridade

    penalidade_prioridade: pacientes de alta prioridade atendidos tarde na rota
    recebem penalidade proporcional à sua posição × fator de prioridade.
    Prioridade 1 (Alta) penaliza mais do que prioridade 2 (Média).
"""

import random
import math
import copy
from typing import List, Tuple

from medical_route_optimizer.data.delivery_points import PontoEntrega


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FATOR_PENALIDADE_PRIORIDADE = 10.0  # peso da penalidade de prioridade no custo


# ---------------------------------------------------------------------------
# Distância
# ---------------------------------------------------------------------------

def calcular_distancia(p1: PontoEntrega, p2: PontoEntrega) -> float:
    """Calcula a distância Euclidiana entre dois pontos de entrega."""
    return math.sqrt((p1.coords[0] - p2.coords[0]) ** 2 +
                     (p1.coords[1] - p2.coords[1]) ** 2)


# ---------------------------------------------------------------------------
# Função de custo (fitness)
# ---------------------------------------------------------------------------

def calcular_custo_rota(
    rota: List[PontoEntrega],
    hospital_base: PontoEntrega,
    fator_penalidade: float = FATOR_PENALIDADE_PRIORIDADE
) -> float:
    """
    Calcula o custo total de uma rota incluindo distância e penalidade por prioridade.

    A rota recebida NÃO inclui o hospital base — ele é adicionado implicitamente
    no início e no fim para formar o ciclo fechado.

    Parâmetros:
    - rota: sequência de pontos de entrega (sem o hospital base)
    - hospital_base: ponto de origem e retorno
    - fator_penalidade: peso aplicado à penalidade de prioridade

    Retorno:
    - custo total (distância + penalidade de prioridade)
    """
    rota_completa = [hospital_base] + list(rota) + [hospital_base]
    n = len(rota_completa)

    # Distância total do ciclo
    distancia_total = sum(
        calcular_distancia(rota_completa[i], rota_completa[i + 1])
        for i in range(n - 1)
    )

    # Penalidade por prioridade:
    # pacientes de alta prioridade (menor número) atendidos em posições tardias
    # na rota recebem penalidade proporcional à (posição × fator / prioridade).
    penalidade_prioridade = 0.0
    for posicao, ponto in enumerate(rota, start=1):
        if ponto.prioridade in (1, 2):  # só penaliza prioridades altas e médias
            penalidade_prioridade += (posicao / ponto.prioridade) * fator_penalidade

    return distancia_total + penalidade_prioridade


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
# Operadores genéticos
# ---------------------------------------------------------------------------

def order_crossover(
    parent1: List[PontoEntrega],
    parent2: List[PontoEntrega]
) -> List[PontoEntrega]:
    """
    Operador de crossover por ordem (OX — Order Crossover) para permutações.

    Preserva a estrutura de permutação válida, evitando duplicidade de cidades.

    Parâmetros:
    - parent1: rota do primeiro pai
    - parent2: rota do segundo pai

    Retorno:
    - rota filho resultante da recombinação
    """
    length = len(parent1)
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    child = parent1[start_index:end_index]
    remaining_positions = [i for i in range(length) if i < start_index or i >= end_index]
    remaining_genes = [gene for gene in parent2 if gene not in child]

    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)

    return child


def mutate(
    rota: List[PontoEntrega],
    probabilidade_mutacao: float
) -> List[PontoEntrega]:
    """
    Operador de mutação: troca dois pontos adjacentes da rota com dada probabilidade.

    Preserva a validade da permutação.

    Parâmetros:
    - rota: sequência de pontos de entrega
    - probabilidade_mutacao: probabilidade de ocorrer mutação [0.0, 1.0]

    Retorno:
    - rota mutada (ou cópia da original se mutação não ocorrer)
    """
    rota_mutada = copy.deepcopy(rota)

    if random.random() < probabilidade_mutacao:
        if len(rota) < 2:
            return rota_mutada
        index = random.randint(0, len(rota) - 2)
        rota_mutada[index], rota_mutada[index + 1] = rota[index + 1], rota[index]

    return rota_mutada


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


# ---------------------------------------------------------------------------
# Loop evolutivo principal
# ---------------------------------------------------------------------------

def executar_algoritmo_genetico(
    locais_entrega: List[PontoEntrega],
    hospital_base: PontoEntrega,
    populacao_inicial: List[List[PontoEntrega]],
    n_geracoes: int = 200,
    probabilidade_mutacao: float = 0.3,
    tamanho_elite: int = 10,
    verbose: bool = True
) -> Tuple[List[PontoEntrega], float, List[float]]:
    """
    Executa o Algoritmo Genético para otimização da rota de entregas médicas.

    Aplica elitismo, seleção por roleta ponderada, crossover OX e mutação.

    Parâmetros:
    - locais_entrega: pontos de entrega (sem o hospital base)
    - hospital_base: ponto de origem e retorno fixo
    - populacao_inicial: população de rotas para iniciar a evolução
    - n_geracoes: número de gerações
    - probabilidade_mutacao: probabilidade de mutação por indivíduo
    - tamanho_elite: número dos melhores usados na seleção parental
    - verbose: se True, imprime progresso por geração

    Retorno:
    - melhor_rota: melhor sequência de pontos de entrega encontrada
    - melhor_custo: custo total da melhor rota
    - historico_custos: lista com o melhor custo por geração
    """
    import numpy as np

    populacao_rotas = list(populacao_inicial)
    historico_custos: List[float] = []

    for geracao in range(1, n_geracoes + 1):

        custos = [calcular_custo_rota(rota, hospital_base) for rota in populacao_rotas]
        populacao_rotas, custos = ordenar_populacao(populacao_rotas, custos)

        melhor_custo = custos[0]
        melhor_rota = populacao_rotas[0]
        historico_custos.append(melhor_custo)

        if verbose:
            print(f"Geração {geracao:>4}: melhor custo = {melhor_custo:.2f}")

        # Seleção por roleta ponderada (probabilidade inversa ao custo)
        pesos = 1.0 / np.array(custos)

        nova_populacao = [melhor_rota]  # Elitismo: preserva o melhor

        while len(nova_populacao) < len(populacao_rotas):
            parent1, parent2 = random.choices(populacao_rotas, weights=pesos, k=2)
            filho = order_crossover(parent1, parent2)
            filho = mutate(filho, probabilidade_mutacao)
            nova_populacao.append(filho)

        populacao_rotas = nova_populacao

    return melhor_rota, melhor_custo, historico_custos
