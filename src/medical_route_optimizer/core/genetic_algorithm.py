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
from typing import List, Optional, Tuple

from medical_route_optimizer.data.delivery_points import PontoEntrega


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FATOR_PENALIDADE_PRIORIDADE = 10.0   # peso da penalidade de prioridade no custo
FATOR_PENALIDADE_CAPACIDADE = 50.0   # penalidade por unidade de carga acima da capacidade
FATOR_PENALIDADE_AUTONOMIA  = 3.0    # penalidade por pixel percorrido além da autonomia


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


# ---------------------------------------------------------------------------
# Função de custo (fitness)
# ---------------------------------------------------------------------------

def calcular_custo_rota(
    rota: List[PontoEntrega],
    hospital_base: PontoEntrega,
    fator_penalidade: float = FATOR_PENALIDADE_PRIORIDADE,
    capacidade_veiculo: Optional[float] = None,
    autonomia_veiculo: Optional[float] = None,
    fator_penalidade_capacidade: float = FATOR_PENALIDADE_CAPACIDADE,
    fator_penalidade_autonomia: float = FATOR_PENALIDADE_AUTONOMIA,
) -> float:
    """
    Calcula o custo total de uma rota incluindo distância e penalidades de restrições.

    A rota recebida NÃO inclui o hospital base — ele é adicionado implicitamente
    no início e no fim para formar o ciclo fechado.

    Penalidades aplicadas:
    - Prioridade   : pacientes críticos atendidos tarde recebem penalidade proporcional
                     à (posição / prioridade) × fator_penalidade.
    - Capacidade   : excesso de carga além de ``capacidade_veiculo`` é penalizado por
                     (excesso_kg × fator_penalidade_capacidade). None = sem restrição.
    - Autonomia    : excesso de distância além de ``autonomia_veiculo`` é penalizado por
                     (excesso_pixels × fator_penalidade_autonomia). None = sem restrição.

    Parâmetros:
    - rota: sequência de pontos de entrega (sem o hospital base)
    - hospital_base: ponto de origem e retorno
    - fator_penalidade: peso da penalidade de prioridade
    - capacidade_veiculo: carga máxima permitida por veículo (None = irrestrito)
    - autonomia_veiculo: distância máxima por ciclo em pixels (None = irrestrito)
    - fator_penalidade_capacidade: penalidade por unidade acima da capacidade
    - fator_penalidade_autonomia: penalidade por pixel além da autonomia

    Retorno:
    - custo total (distância + penalidades)
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

    # Penalidade por capacidade:
    # aplica somente se ``capacidade_veiculo`` for definido.
    penalidade_capacidade = 0.0
    if capacidade_veiculo is not None:
        peso_total = sum(p.peso for p in rota)
        excesso = max(0.0, peso_total - capacidade_veiculo)
        penalidade_capacidade = excesso * fator_penalidade_capacidade

    # Penalidade por autonomia:
    # aplica somente se ``autonomia_veiculo`` for definido.
    penalidade_autonomia = 0.0
    if autonomia_veiculo is not None:
        excesso = max(0.0, distancia_total - autonomia_veiculo)
        penalidade_autonomia = excesso * fator_penalidade_autonomia

    return distancia_total + penalidade_prioridade + penalidade_capacidade + penalidade_autonomia


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
    paciencia: int = 50,
    tolerancia: float = 1e-6,
    capacidade_veiculo: Optional[float] = None,
    autonomia_veiculo: Optional[float] = None,
    verbose: bool = True
) -> Tuple[List[PontoEntrega], float, List[float]]:
    """
    Executa o Algoritmo Genético para otimização da rota de entregas médicas.

    Aplica elitismo, seleção por roleta ponderada, crossover OX e mutação.
    A evolução para antecipadamente quando não há melhora no melhor custo por
    ``paciencia`` gerações consecutivas (convergência), ou ao atingir ``n_geracoes``.

    Quando ``capacidade_veiculo`` ou ``autonomia_veiculo`` são fornecidos, o GA
    opera em modo VRP penalizado: a função de custo inclui penalidades por
    violação das restrições, guiando o algoritmo a organizar o "giant tour" de
    forma que os particionamentos respeitem as restrições do veículo.

    Parâmetros:
    - locais_entrega: pontos de entrega (sem o hospital base)
    - hospital_base: ponto de origem e retorno fixo
    - populacao_inicial: população de rotas para iniciar a evolução
    - n_geracoes: número máximo de gerações
    - probabilidade_mutacao: probabilidade de mutação por indivíduo
    - tamanho_elite: número dos melhores usados na seleção parental
    - paciencia: gerações consecutivas sem melhora para acionar parada antecipada
    - tolerancia: melhoria mínima absoluta para ser considerada progresso
    - capacidade_veiculo: carga máxima por veículo (None = TSP sem restrição)
    - autonomia_veiculo: distância máxima por ciclo em pixels (None = sem restrição)
    - verbose: se True, imprime progresso por geração

    Retorno:
    - melhor_rota: melhor sequência de pontos de entrega encontrada
    - melhor_custo: custo total da melhor rota
    - historico_custos: lista com o melhor custo por geração
    """
    import numpy as np

    def _custo(rota):
        return calcular_custo_rota(
            rota, hospital_base,
            capacidade_veiculo=capacidade_veiculo,
            autonomia_veiculo=autonomia_veiculo,
        )

    populacao_rotas = list(populacao_inicial)
    historico_custos: List[float] = []

    # Inicializa o melhor global com a avaliação da população inicial
    custos_iniciais = [_custo(rota) for rota in populacao_rotas]
    idx_melhor = custos_iniciais.index(min(custos_iniciais))
    melhor_custo_global: float = custos_iniciais[idx_melhor]
    melhor_rota_global: List[PontoEntrega] = populacao_rotas[idx_melhor]
    geracoes_sem_melhora = 0

    for geracao in range(1, n_geracoes + 1):

        custos = [_custo(rota) for rota in populacao_rotas]
        populacao_rotas, custos = ordenar_populacao(populacao_rotas, custos)

        melhor_custo = custos[0]
        melhor_rota = populacao_rotas[0]
        historico_custos.append(melhor_custo)

        # Rastreia o melhor global e contabiliza estagnação
        if melhor_custo_global - melhor_custo > tolerancia:
            melhor_custo_global = melhor_custo
            melhor_rota_global = melhor_rota
            geracoes_sem_melhora = 0
        else:
            geracoes_sem_melhora += 1

        if verbose:
            print(
                f"Geração {geracao:>4}: melhor custo = {melhor_custo:.2f}"
                f"  (sem melhora: {geracoes_sem_melhora}/{paciencia})"
            )

        # Parada antecipada por convergência
        if geracoes_sem_melhora >= paciencia:
            if verbose:
                print(
                    f"\n⏹  Parada antecipada na geração {geracao}: "
                    f"sem melhora por {paciencia} gerações consecutivas."
                )
            break

        # Seleção por roleta ponderada (probabilidade inversa ao custo)
        pesos = 1.0 / np.array(custos)

        nova_populacao = [melhor_rota]  # Elitismo: preserva o melhor

        while len(nova_populacao) < len(populacao_rotas):
            parent1, parent2 = random.choices(populacao_rotas, weights=pesos, k=2)
            # Usa comparação de conteúdo (==) para detectar pais com rotas idênticas
            # e garantir diversidade genética no crossover. random.choices sempre
            # retorna novos objetos, então `is` nunca detectaria duplicatas de conteúdo.
            if parent1 == parent2:
                parent2 = random.choices(populacao_rotas, weights=pesos, k=1)[0]
            filho = order_crossover(parent1, parent2)
            filho = mutate(filho, probabilidade_mutacao)
            nova_populacao.append(filho)

        populacao_rotas = nova_populacao

    return melhor_rota_global, melhor_custo_global, historico_custos
