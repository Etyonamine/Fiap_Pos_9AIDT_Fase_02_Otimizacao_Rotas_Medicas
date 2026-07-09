"""
VRP Split: particionamento de giant tour em sub-rotas por veículo.

Abordagem "Penalized GA + Greedy Split":
    1. O GA otimiza um "giant tour" (permutação de todos os pontos) com
       penalidades por violação de capacidade e autonomia, resultando numa
       ordenação de pontos que minimiza o custo total considerando as restrições.
    2. Este módulo particiona o giant tour em sub-rotas válidas para cada
       veículo usando o algoritmo de particionamento guloso (greedy split):
       - Adiciona pontos à rota do veículo atual enquanto as restrições forem
         respeitadas.
       - Quando adicionar o próximo ponto violaria capacidade ou autonomia,
         fecha a rota do veículo atual e inicia uma nova rota.

Complexidade: O(n) — percorre o giant tour uma única vez.

Referências:
    - Prins, C. (2004). A simple and effective evolutionary algorithm for the
      vehicle routing problem. Computers & Operations Research, 31(12).
"""

from typing import List, Optional, Tuple

from data.delivery_points import PontoEntrega
from core.fitness_calculator import calcular_custo_rota
from core.route_calculator import calcular_distancia


def dividir_rotas_vrp(
    giant_tour: List[PontoEntrega],
    hospital_base: PontoEntrega,
    capacidade_veiculo: Optional[float] = None,
    autonomia_veiculo: Optional[float] = None,
    n_veiculos: Optional[int] = None,
) -> List[List[PontoEntrega]]:
    """
    Divide um giant tour em sub-rotas válidas para múltiplos veículos.

    Utiliza particionamento guloso: percorre o giant tour sequencialmente e
    abre uma nova rota de veículo sempre que adicionar o próximo ponto violaria
    a capacidade ou a autonomia do veículo atual.

    Parâmetros:
    - giant_tour: permutação completa de todos os pontos de entrega
    - hospital_base: ponto de origem e retorno (depósito)
    - capacidade_veiculo: carga máxima por veículo em unidades (None = irrestrito)
    - autonomia_veiculo: distância máxima por ciclo em pixels (None = irrestrito)
    - n_veiculos: número máximo de veículos disponíveis (None = irrestrito).
                  Se as restrições exigirem mais veículos do que este limite,
                  o excedente é adicionado ao último veículo (infeasível, mas
                  mantém todos os pontos cobertos).

    Retorno:
    - lista de sub-rotas, uma por veículo; cada sub-rota é uma lista de
      pontos de entrega (sem o hospital base nos extremos).
    """
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

        # Se há violação E a rota atual já tem pelo menos um ponto,
        # e ainda há veículos disponíveis → fecha a rota e abre nova.
        # O limite é `n_veiculos - 1` porque o último veículo é reservado
        # para absorver todos os pontos restantes (incluindo os infeasíveis):
        # len(rotas) < n_veiculos - 1  →  "posso abrir mais um veículo"
        # len(rotas) == n_veiculos - 1 →  "já tenho n_veiculos-1 rotas fechadas;
        #                                   o veículo atual É o último disponível"
        veiculos_disponiveis = n_veiculos is None or len(rotas) < n_veiculos - 1
        if (capacidade_violada or autonomia_violada) and rota_atual and veiculos_disponiveis:
            rotas.append(rota_atual)
            rota_atual = []
            peso_atual = 0.0
            dist_atual = 0.0
            ponto_anterior = hospital_base
            # Recalcula distância do depósito até este ponto
            dist_passo = calcular_distancia(hospital_base, ponto)

        rota_atual.append(ponto)
        peso_atual += ponto.peso
        dist_atual += dist_passo
        ponto_anterior = ponto

    if rota_atual:
        rotas.append(rota_atual)

    return rotas


def calcular_custo_vrp(
    rotas_vrp: List[List[PontoEntrega]],
    hospital_base: PontoEntrega,
    capacidade_veiculo: Optional[float] = None,
    autonomia_veiculo: Optional[float] = None,
) -> Tuple[float, List[float]]:
    """
    Calcula o custo total da solução VRP e o custo individual de cada veículo.

    Parâmetros:
    - rotas_vrp: lista de sub-rotas (uma por veículo)
    - hospital_base: ponto de origem e retorno
    - capacidade_veiculo: limite de carga por veículo (None = irrestrito)
    - autonomia_veiculo: limite de distância por veículo (None = irrestrito)

    Retorno:
    - (custo_total, custos_por_veiculo)
    """
    custos = [
        calcular_custo_rota(
            rota, hospital_base,
            capacidade_veiculo=capacidade_veiculo,
            autonomia_veiculo=autonomia_veiculo,
        )
        for rota in rotas_vrp
    ]
    return sum(custos), custos


def resumo_restricoes_vrp(
    rotas_vrp: List[List[PontoEntrega]],
    hospital_base: PontoEntrega,
    capacidade_veiculo: Optional[float],
    autonomia_veiculo: Optional[float],
) -> List[dict]:
    """
    Gera um resumo das métricas de restrição por veículo.

    Retorno:
    - lista de dicionários com métricas de cada veículo
    """
    resumo = []
    for idx, rota in enumerate(rotas_vrp, start=1):
        peso = sum(p.peso for p in rota)
        rota_completa = [hospital_base] + rota + [hospital_base]
        distancia = sum(
            calcular_distancia(rota_completa[i], rota_completa[i + 1])
            for i in range(len(rota_completa) - 1)
        )
        capacidade_ok = capacidade_veiculo is None or peso <= capacidade_veiculo
        autonomia_ok = autonomia_veiculo is None or distancia <= autonomia_veiculo

        resumo.append({
            "veiculo": idx,
            "n_pontos": len(rota),
            "peso_total": round(peso, 2),
            "capacidade_veiculo": capacidade_veiculo,
            "capacidade_ok": capacidade_ok,
            "distancia_pixels": round(distancia, 2),
            "autonomia_veiculo": autonomia_veiculo,
            "autonomia_ok": autonomia_ok,
            "pontos": [p.nome for p in rota],
        })
    return resumo
