
# ---------------------------------------------------------------------------
# Função de custo (fitness)
# ---------------------------------------------------------------------------

from typing import List, Optional
from data.delivery_points import PontoEntrega
from medical_route_optimizer.core.route_calculator import calcular_distancia


def calcular_distancia_rota(
    rota: List[PontoEntrega],
    hospital_base: PontoEntrega,
) -> float:
    """
    Retorna apenas a distância total do ciclo (depósito → rota → depósito),
    sem nenhuma penalidade.

    Útil para comparativos finais entre abordagens, onde se deseja medir
    exclusivamente o custo geográfico, independente dos fatores de penalidade
    usados internamente pelo GA.
    """
    rota_completa = [hospital_base] + list(rota) + [hospital_base]
    return sum(
        calcular_distancia(rota_completa[i], rota_completa[i + 1])
        for i in range(len(rota_completa) - 1)
    )


def calcular_custo_rota(
    rota: List[PontoEntrega],
    hospital_base: PontoEntrega,
    fator_penalidade: float = 5.0,
    capacidade_veiculo: Optional[float] = None,
    autonomia_veiculo: Optional[float] = None,
    fator_penalidade_capacidade: float = 17.0,
    fator_penalidade_autonomia: float = 1.0,
) -> float:
    """
    Calcula o custo total de uma rota incluindo distância e penalidades de restrições.

    Penalidades aplicadas:
    - Prioridade   : pacientes críticos atendidos tarde recebem penalidade proporcional
                     à (posição / prioridade) × fator_penalidade.
    - Capacidade   : excesso de carga além de ``capacidade_veiculo`` é penalizado de forma
                     proporcional ao excesso relativo (excesso/capacidade × fator).
    - Autonomia    : excesso de distância além de ``autonomia_veiculo`` é penalizado por
                     (excesso_pixels × fator_penalidade_autonomia).
    """
    rota_completa = [hospital_base] + list(rota) + [hospital_base]
    n = len(rota_completa)

    # Distância total do ciclo
    distancia_total = sum(
        calcular_distancia(rota_completa[i], rota_completa[i + 1])
        for i in range(n - 1)
    )

    # Penalidade por prioridade
    penalidade_prioridade = 0.0
    for posicao, ponto in enumerate(rota, start=1):
        if ponto.prioridade in (1, 2):
            penalidade_prioridade += (posicao / ponto.prioridade) * fator_penalidade

    # Penalidade proporcional por capacidade
    penalidade_capacidade = 0.0
    if capacidade_veiculo is not None:
        peso_total = sum(p.peso for p in rota)
        excesso = max(0.0, peso_total - capacidade_veiculo)
        if excesso > 0:
            excesso_relativo = excesso / capacidade_veiculo
            penalidade_capacidade = excesso_relativo * fator_penalidade_capacidade

    # Penalidade por autonomia
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
    fator_penalidade: float = 5.0,
    fator_penalidade_capacidade: float = 17.0,
    fator_penalidade_autonomia: float = 1.0,
) -> float:
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

    # Soma dos custos de cada sub-rota
    custo_total = 0.0
    for rota in rotas:
        custo_total += calcular_custo_rota(
            rota, hospital_base,
            fator_penalidade=fator_penalidade,
            capacidade_veiculo=capacidade_veiculo,
            autonomia_veiculo=autonomia_veiculo,
            fator_penalidade_capacidade=fator_penalidade_capacidade,
            fator_penalidade_autonomia=fator_penalidade_autonomia,
        )
    return custo_total

