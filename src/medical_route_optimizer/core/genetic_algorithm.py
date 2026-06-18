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

from data.delivery_points import PontoEntrega
from core.genetic_operator import order_crossover, mutate, mutate_segment_inversion, pmx_crossover

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FATOR_PENALIDADE_PRIORIDADE = 5.0    # peso da penalidade de prioridade no custo
FATOR_PENALIDADE_CAPACIDADE = 17.0  # penalidade por unidade de carga acima da capacidade
FATOR_PENALIDADE_AUTONOMIA  = 1.0    # penalidade por pixel percorrido além da autonomia

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

def calcular_custo_giant_tour_vrp(
    giant_tour: List[PontoEntrega],
    hospital_base: PontoEntrega,
    n_veiculos: int,
    capacidade_veiculo: Optional[float] = None,
    autonomia_veiculo: Optional[float] = None,
    fator_penalidade: float = FATOR_PENALIDADE_PRIORIDADE,
    fator_penalidade_capacidade: float = FATOR_PENALIDADE_CAPACIDADE,
    fator_penalidade_autonomia: float = FATOR_PENALIDADE_AUTONOMIA,
) -> float:
    """
    Fitness VRP-aware: simula o particionamento guloso (greedy split) do giant
    tour em sub-rotas por veículo e avalia o custo total considerando penalidades
    **por veículo**.

    Isso garante que a penalidade de capacidade varie entre indivíduos (ao
    contrário de avaliar o giant tour inteiro como uma única rota, onde a
    penalidade seria constante para toda a população e não guiaria a evolução).

    O GA aprende a **ordenar o giant tour** de modo que o greedy split produza
    sub-rotas balanceadas e factíveis.

    Parâmetros:
    - giant_tour: permutação completa de todos os pontos de entrega
    - hospital_base: ponto de origem e retorno (depósito)
    - n_veiculos: número máximo de veículos disponíveis
    - capacidade_veiculo: carga máxima por veículo (None = irrestrito)
    - autonomia_veiculo: distância máxima por ciclo em pixels (None = irrestrito)
    - fator_penalidade: peso da penalidade de prioridade
    - fator_penalidade_capacidade: penalidade por unidade de excesso de carga
    - fator_penalidade_autonomia: penalidade por pixel além da autonomia

    Retorno:
    - custo total VRP (soma dos custos de todas as sub-rotas com penalidades)
    """
    # Greedy split inline (replica vrp_split.dividir_rotas_vrp sem importar o módulo,
    # evitando importação circular)
    rotas: List[List[PontoEntrega]] = []
    rota_atual: List[PontoEntrega] = []
    peso_atual: float = 0.0
    dist_atual: float = 0.0
    ponto_anterior: PontoEntrega = hospital_base

    for ponto in giant_tour:
        dist_passo = calcular_distancia(ponto_anterior, ponto)
        dist_retorno = calcular_distancia(ponto, hospital_base)

        capacidade_violada = (
            capacidade_veiculo is not None
            and peso_atual + ponto.peso > capacidade_veiculo
        )
        autonomia_violada = (
            autonomia_veiculo is not None
            and dist_atual + dist_passo + dist_retorno > autonomia_veiculo
        )

        veiculos_disponiveis = len(rotas) < n_veiculos - 1
        if (capacidade_violada or autonomia_violada) and rota_atual and veiculos_disponiveis:
            rotas.append(rota_atual)
            rota_atual = []
            peso_atual = 0.0
            dist_atual = 0.0
            ponto_anterior = hospital_base
            dist_passo = calcular_distancia(hospital_base, ponto)

        rota_atual.append(ponto)
        peso_atual += ponto.peso
        dist_atual += dist_passo
        ponto_anterior = ponto

    if rota_atual:
        rotas.append(rota_atual)

    # Custo total: soma do custo de cada sub-rota com penalidades por veículo
    return sum(
        calcular_custo_rota(
            rota, hospital_base,
            fator_penalidade=fator_penalidade,
            capacidade_veiculo=capacidade_veiculo,
            autonomia_veiculo=autonomia_veiculo,
            fator_penalidade_capacidade=fator_penalidade_capacidade,
            fator_penalidade_autonomia=fator_penalidade_autonomia,
        )
        for rota in rotas
    )

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

# ---------------------------------------------------------------------------
# Loop evolutivo principal
# --------------------------------------------------------------------------- 
def executar_algoritmo_genetico(
    locais_entrega: List[PontoEntrega],
    hospital_base: PontoEntrega,
    populacao_inicial: List[List[PontoEntrega]],
    n_geracoes: int = 200,
    probabilidade_mutacao: float = 0.4,
    probabilidade_crossover: float = 1.0,
    tamanho_elite: int = 10,
    paciencia: int = 150,
    tolerancia: float = 1e-6,
    n_veiculos: Optional[int] = None,
    capacidade_veiculo: Optional[float] = None,
    autonomia_veiculo: Optional[float] = None,
    fator_penalidade: float = FATOR_PENALIDADE_PRIORIDADE,
    fator_penalidade_capacidade: float = FATOR_PENALIDADE_CAPACIDADE,
    fator_penalidade_autonomia: float = FATOR_PENALIDADE_AUTONOMIA,
    verbose: bool = True
) -> Tuple[List[PontoEntrega], float, List[float], List[float]]:
    import numpy as np

    # Fitness adaptado para VRP
    _vrp_mode = n_veiculos is not None and (
        capacidade_veiculo is not None or autonomia_veiculo is not None
    )

    def _custo(rota):
        if _vrp_mode:
            return calcular_custo_giant_tour_vrp(
                rota, hospital_base,
                n_veiculos=n_veiculos,
                capacidade_veiculo=capacidade_veiculo,
                autonomia_veiculo=autonomia_veiculo,
                fator_penalidade=fator_penalidade,
                fator_penalidade_capacidade=fator_penalidade_capacidade,
                fator_penalidade_autonomia=fator_penalidade_autonomia,
            )
        return calcular_custo_rota(
            rota, hospital_base,
            capacidade_veiculo=capacidade_veiculo,
            autonomia_veiculo=autonomia_veiculo,
            fator_penalidade=fator_penalidade,
            fator_penalidade_capacidade=fator_penalidade_capacidade,
            fator_penalidade_autonomia=fator_penalidade_autonomia,
        )

    populacao_rotas = list(populacao_inicial)
    historico_custos, historico_media = [], []

    custos_iniciais = [_custo(rota) for rota in populacao_rotas]
    idx_melhor = custos_iniciais.index(min(custos_iniciais))
    melhor_custo_global = custos_iniciais[idx_melhor]
    melhor_rota_global = populacao_rotas[idx_melhor]
    geracoes_sem_melhora = 0

    for geracao in range(1, n_geracoes + 1):
        custos = [_custo(rota) for rota in populacao_rotas]
        populacao_rotas, custos = ordenar_populacao(populacao_rotas, custos)

        melhor_custo, melhor_rota = custos[0], populacao_rotas[0]
        historico_custos.append(melhor_custo)
        historico_media.append(float(np.mean(custos)))

        if melhor_custo_global - melhor_custo > tolerancia:
            melhor_custo_global, melhor_rota_global = melhor_custo, melhor_rota
            geracoes_sem_melhora = 0
        else:
            geracoes_sem_melhora += 1

        if verbose:
            print(f"Geração {geracao:>4}: melhor custo = {melhor_custo:.2f} "
                  f"(sem melhora: {geracoes_sem_melhora}/{paciencia})")

        if geracoes_sem_melhora >= paciencia:
            if verbose:
                print(f"\n⏹  Parada antecipada na geração {geracao}: "
                      f"sem melhora por {paciencia} gerações consecutivas.")
            break

        pesos_selecao = 1.0 / np.array(custos)
        nova_populacao = [melhor_rota]  # elitismo

        while len(nova_populacao) < len(populacao_rotas):
            parent1, parent2 = random.choices(populacao_rotas, weights=pesos_selecao, k=2)
            if parent1 == parent2:
                parent2 = random.choices(populacao_rotas, weights=pesos_selecao, k=1)[0]

            # Crossover híbrido
            if random.random() < probabilidade_crossover:
                if random.random() < 0.5:
                    filho = order_crossover(parent1, parent2)
                else:
                    filho = pmx_crossover(parent1, parent2)
            else:
                filho = list(parent1)

            # Mutação híbrida
            if random.random() < 0.5:
                filho = mutate(filho, probabilidade_mutacao)
            else:
                filho = mutate_segment_inversion(filho, probabilidade_mutacao)

            nova_populacao.append(filho)

        populacao_rotas = nova_populacao

    return melhor_rota_global, melhor_custo_global, historico_custos, historico_media