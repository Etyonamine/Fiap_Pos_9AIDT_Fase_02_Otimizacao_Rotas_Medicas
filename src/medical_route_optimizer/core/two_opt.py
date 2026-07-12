"""
Heurística de Melhoria: Two Opt Inversion para TSP.

Algoritmo:
    Para cada par de arestas (i, k) na rota:
        - Inverte o segmento entre os índices i+1 e k (inclusive).
        - Se o novo custo for menor, aceita a melhoria.
        - Repete até não encontrar mais melhorias (convergência local).

Complexidade por iteração: O(n²)
Uso no pipeline:
    - Aplicado UMA VEZ sobre a melhor rota encontrada pelo GA após convergência.
    - Refinamento local que elimina cruzamentos de arestas e reduz custo final.
    - Não modifica o GA — é uma etapa de pós-processamento.
"""

from typing import List, Optional, Tuple
from core.fitness_calculator import calcular_custo_rota
from data.delivery_points import PontoEntrega


def two_opt_inversion(
    rota: List[PontoEntrega],
    hospital_base: PontoEntrega,
    fator_penalidade: float = 5.0,
    capacidade_veiculo: Optional[float] = None,
    autonomia_veiculo: Optional[float] = None,
    fator_penalidade_capacidade: float = 17.0,
    fator_penalidade_autonomia: float = 1.0,
    max_iteracoes: int = 1000,
    verbose: bool = False
) -> Tuple[List[PontoEntrega], float]:
    """
    Aplica o Two Opt Inversion sobre uma rota para refinamento local.

    Para cada par de arestas (i, k), testa se inverter o segmento [i+1 .. k]
    reduz o custo total. Aceita imediatamente a primeira melhoria encontrada
    (estratégia first improvement) e reinicia a busca.

    Os parâmetros de custo devem ser os mesmos usados pelo GA e pelo baseline NN
    para garantir que todas as métricas estejam na mesma escala.

    Parâmetros:
    - rota: sequência de pontos de entrega (sem o hospital base nos extremos)
    - hospital_base: ponto de origem e retorno (usado no cálculo do custo)
    - fator_penalidade: peso da penalidade de prioridade (deve coincidir com o GA)
    - capacidade_veiculo: limite de carga — None = irrestrito
    - autonomia_veiculo: limite de distância — None = irrestrito
    - fator_penalidade_capacidade: peso por excesso de carga
    - fator_penalidade_autonomia: peso por excesso de distância
    - max_iteracoes: limite máximo de iterações para evitar loop infinito
    - verbose: se True, imprime cada melhoria encontrada

    Retorno:
    - (rota_otimizada, custo_otimizado)
    """
    def _custo(r):
        return calcular_custo_rota(
            r, hospital_base,
            fator_penalidade=fator_penalidade,
            capacidade_veiculo=capacidade_veiculo,
            autonomia_veiculo=autonomia_veiculo,
            fator_penalidade_capacidade=fator_penalidade_capacidade,
            fator_penalidade_autonomia=fator_penalidade_autonomia,
        )

    melhor_rota = list(rota)
    melhor_custo = _custo(melhor_rota)
    n = len(melhor_rota)
    iteracao = 0
    melhoria_encontrada = True

    while melhoria_encontrada and iteracao < max_iteracoes:
        melhoria_encontrada = False
        iteracao += 1

        for i in range(n - 1):
            for k in range(i + 1, n):
                # Gera nova rota invertendo o segmento entre i+1 e k
                nova_rota = melhor_rota[:i + 1] + \
                            melhor_rota[i + 1:k + 1][::-1] + \
                            melhor_rota[k + 1:]

                novo_custo = _custo(nova_rota)

                if novo_custo < melhor_custo:
                    melhor_rota = nova_rota
                    melhor_custo = novo_custo
                    melhoria_encontrada = True

                    if verbose:
                        print(f"  Two Opt iter {iteracao}: inversão [{i+1}:{k+1}] "
                              f"→ custo = {melhor_custo:.2f}")
                    break  # first improvement: reinicia

            if melhoria_encontrada:
                break

    return melhor_rota, melhor_custo


def two_opt_vrp(
    rotas_vrp: List[List[PontoEntrega]],
    hospital_base: PontoEntrega,
    fator_penalidade: float = 5.0,
    capacidade_veiculo: Optional[float] = None,
    autonomia_veiculo: Optional[float] = None,
    fator_penalidade_capacidade: float = 17.0,
    fator_penalidade_autonomia: float = 1.0,
    max_iteracoes: int = 1000,
    verbose: bool = False,
) -> Tuple[List[List[PontoEntrega]], float]:
    """
    Aplica Two-Opt Inversion individualmente em cada sub-rota de uma solução VRP.

    Ao contrário de aplicar Two-Opt sobre o giant tour, esta abordagem preserva
    a divisão de veículos estabelecida pelo VRP Split e melhora cada sub-rota
    de forma independente, evitando que a reordenação global desfaça uma
    partição válida das restrições de capacidade/autonomia.

    Parâmetros:
    - rotas_vrp: lista de sub-rotas (uma por veículo), sem o hospital base nos extremos
    - hospital_base: ponto de origem e retorno de cada veículo
    - demais parâmetros: idênticos a two_opt_inversion, devem ser os mesmos
      usados pelo GA e pelo baseline NN para comparativos coerentes

    Retorno:
    - (rotas_otimizadas, custo_total) onde custo_total é a soma dos custos
      de cada sub-rota após o refinamento
    """
    rotas_otimizadas: List[List[PontoEntrega]] = []
    custo_total = 0.0

    for rota in rotas_vrp:
        rota_otim, custo = two_opt_inversion(
            rota, hospital_base,
            fator_penalidade=fator_penalidade,
            capacidade_veiculo=capacidade_veiculo,
            autonomia_veiculo=autonomia_veiculo,
            fator_penalidade_capacidade=fator_penalidade_capacidade,
            fator_penalidade_autonomia=fator_penalidade_autonomia,
            max_iteracoes=max_iteracoes,
            verbose=verbose,
        )
        rotas_otimizadas.append(rota_otim)
        custo_total += custo

    return rotas_otimizadas, custo_total

