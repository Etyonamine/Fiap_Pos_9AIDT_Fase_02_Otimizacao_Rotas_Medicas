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
from typing import List, Optional, Tuple

from data.delivery_points import PontoEntrega
from core.genetic_operator import order_crossover, mutate, mutate_segment_inversion, pmx_crossover
from core.fitness_calculator import calcular_custo_rota, calcular_custo_giant_tour_vrp
from core.population_helper import ordenar_populacao
from visualizacao.animacao_vrp import AnimacaoVRP



# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FATOR_PENALIDADE_PRIORIDADE = 2.0    # peso da penalidade de prioridade no custo
FATOR_PENALIDADE_CAPACIDADE = 5.0  # penalidade por unidade de carga acima da capacidade
FATOR_PENALIDADE_AUTONOMIA  = 1.5    # penalidade por pixel percorrido além da autonomia


# ---------------------------------------------------------------------------
# Loop evolutivo principal
# --------------------------------------------------------------------------- 
def executar_algoritmo_genetico(
    locais_entrega: List[PontoEntrega],
    hospital_base: PontoEntrega,
    populacao_inicial: List[List[PontoEntrega]],
    probabilidade_mutacao: float = 0.4,
    probabilidade_crossover: float = 1.0,
    tamanho_elite: int = 10,
    paciencia: int = 150,
    tolerancia: float = 1e-4,
    n_veiculos: Optional[int] = None,
    capacidade_veiculo: Optional[float] = None,
    autonomia_veiculo: Optional[float] = None,
    fator_penalidade: float = FATOR_PENALIDADE_PRIORIDADE,
    fator_penalidade_capacidade: float = FATOR_PENALIDADE_CAPACIDADE,
    fator_penalidade_autonomia: float = FATOR_PENALIDADE_AUTONOMIA,
    verbose: bool = True,
    limite_tempo: Optional[int] = 120,  # tempo máximo em segundos
    animacao: Optional[AnimacaoVRP] = None
) -> Tuple[List[PontoEntrega], float, List[float], List[float]]:
    import numpy as np
    import time

    inicio = time.time()

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
    geracao = 0

    

    while True:  # loop infinito até critério híbrido parar
        geracao += 1
        custos = [_custo(rota) for rota in populacao_rotas]
        populacao_rotas, custos = ordenar_populacao(populacao_rotas, custos)

        melhor_custo, melhor_rota = custos[0], populacao_rotas[0]
        historico_custos.append(melhor_custo)
        historico_media.append(float(np.mean(custos)))

        media_custos = float(np.mean(custos))
        if animacao:
            animacao.registrar(geracao, melhor_custo, media_custos, melhor_rota, populacao_inicial[0])

        if melhor_custo_global - melhor_custo > tolerancia:
            melhoria = melhor_custo_global - melhor_custo
            melhor_custo_global, melhor_rota_global = melhor_custo, melhor_rota
            geracoes_sem_melhora = 0
            if verbose:
                print(f"Geração {geracao:>4}: melhor custo = {melhor_custo:.2f} | melhoria detectada: {melhoria:.2f}")
        else:
            geracoes_sem_melhora += 1
            if verbose:
                print(f"Geração {geracao:>4}: melhor custo = {melhor_custo:.2f} (sem melhoria: {geracoes_sem_melhora}/{paciencia})")

        # Critérios híbridos de parada
        if geracoes_sem_melhora >= paciencia:
            print(f"\n⏹  Parada antecipada na geração {geracao}: sem melhoria por {paciencia} gerações consecutivas.")
            break
        if np.std(custos) < 1e-3:
            print(f"\n⏹  Parada antecipada na geração {geracao}: população convergiu (baixa diversidade).")
            break
        if limite_tempo is not None and (time.time() - inicio > limite_tempo):
            print(f"\n⏹  Parada antecipada na geração {geracao}: limite de tempo atingido ({limite_tempo}s).")
            break

        # Seleção com elitismo
        pesos_selecao = 1.0 / np.array(custos)
        nova_populacao = populacao_rotas[:tamanho_elite]

        while len(nova_populacao) < len(populacao_rotas):
            parent1, parent2 = random.choices(populacao_rotas, weights=pesos_selecao, k=2)
            if parent1 == parent2:
                parent2 = random.choices(populacao_rotas, weights=pesos_selecao, k=1)[0]

            # Crossover híbrido
            if random.random() < probabilidade_crossover:
                filho = order_crossover(parent1, parent2) if random.random() < 0.5 else pmx_crossover(parent1, parent2)
            else:
                filho = list(parent1)

            # Mutação adaptativa
            taxa_mutacao_atual = probabilidade_mutacao
            if geracoes_sem_melhora > 20:
                taxa_mutacao_atual = min(1.0, probabilidade_mutacao * 1.5)
                if verbose:
                    print(f"⚡ Mutação adaptativa ativada na geração {geracao}: taxa = {taxa_mutacao_atual:.2f}")

            filho = mutate(filho, taxa_mutacao_atual) if random.random() < 0.5 else mutate_segment_inversion(filho, taxa_mutacao_atual)
            nova_populacao.append(filho)

        # Reinicialização parcial
        if geracoes_sem_melhora > paciencia // 2:
            if verbose:
                print(f"🔄 Reinicialização parcial na geração {geracao}: adicionando novos indivíduos.")
            for _ in range(len(populacao_rotas) // 4):
                nova_populacao.append(random.sample(locais_entrega, len(locais_entrega)))

        populacao_rotas = nova_populacao

    if animacao:
        animacao.finalizar()

    return melhor_rota_global, melhor_custo_global, historico_custos, historico_media

