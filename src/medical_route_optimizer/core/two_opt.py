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

from typing import List, Tuple
from core.genetic_algorithm import calcular_custo_rota
from data.delivery_points import PontoEntrega


def two_opt_inversion(
    rota: List[PontoEntrega],
    hospital_base: PontoEntrega,
    max_iteracoes: int = 1000,
    verbose: bool = False
) -> Tuple[List[PontoEntrega], float]:
    """
    Aplica o Two Opt Inversion sobre uma rota para refinamento local.

    Para cada par de arestas (i, k), testa se inverter o segmento [i+1 .. k]
    reduz o custo total. Aceita imediatamente a primeira melhoria encontrada
    (estratégia first improvement) e reinicia a busca.

    Parâmetros:
    - rota: sequência de pontos de entrega (sem o hospital base nos extremos)
    - hospital_base: ponto de origem e retorno (usado no cálculo do custo)
    - max_iteracoes: limite máximo de iterações para evitar loop infinito
    - verbose: se True, imprime cada melhoria encontrada

    Retorno:
    - (rota_otimizada, custo_otimizado)
    """
    melhor_rota = list(rota)
    melhor_custo = calcular_custo_rota(melhor_rota, hospital_base)
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

                novo_custo = calcular_custo_rota(nova_rota, hospital_base)

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
