from typing import List
from data.delivery_points import get_hospital_base, get_pontos_entrega_sem_origem
from core.genetic_algorithm import executar_algoritmo_genetico
from core.population_helper import gerar_populacao_aleatoria
from core.nearest_neighbor import gerar_populacao_nearest_neighbor, avaliar_baseline_nn
from core.two_opt import two_opt_vrp
from core.vrp_split import dividir_rotas_vrp, resumo_restricoes_vrp
from reports.route_report import gerar_relatorio_rota
from models.resultado_ga import ResultadoGA



PROPORCAO_NN = 0.15
DEFAULT_PACIENCIA = 150

def run_ga(tamanho_pop: int, taxa_crossover: float,
           taxa_mutacao: float, pesos: List[float], n_veiculos: int,
           capacidade: float, autonomia: float) -> ResultadoGA:
    """Executa o pipeline completo de otimização e retorna um objeto ResultadoGA.

    Mapeamento de pesos: [prioridade, capacidade, autonomia].
    Valores excedentes são ignorados e faltantes recebem 1.0.

    Pipeline (Opção 1 + Opção 3):
    1. GA otimiza o giant tour com os mesmos parâmetros de custo.
    2. VRP Split divide o giant tour em sub-rotas válidas por veículo.
    3. Two-Opt VRP refina cada sub-rota individualmente, preservando a divisão
       de veículos e usando os mesmos parâmetros de custo do GA e do baseline NN.

    Desta forma todos os valores reportados (custo_nn, custo do GA, custo_final)
    estão na mesma escala de avaliação, tornando o comparativo coerente.
    """
    hospital_base = get_hospital_base()
    locais = get_pontos_entrega_sem_origem()
    pesos_efetivos = (list(pesos) + [1.0, 1.0, 1.0])[:3]

    # Parâmetros de custo compartilhados entre todas as etapas
    params_custo = dict(
        fator_penalidade=pesos_efetivos[0],
        fator_penalidade_capacidade=pesos_efetivos[1],
        fator_penalidade_autonomia=pesos_efetivos[2],
        capacidade_veiculo=capacidade,
        autonomia_veiculo=autonomia,
    )

    # População inicial híbrida
    n_nn = max(1, int(tamanho_pop * PROPORCAO_NN))
    pop_nn = gerar_populacao_nearest_neighbor(locais, hospital_base, n_nn)
    pop_aleat = gerar_populacao_aleatoria(locais, tamanho_pop - n_nn)
    pop_inicial = pop_nn + pop_aleat

    # Baseline NN com os mesmos parâmetros de custo do GA
    rota_nn, custo_nn = avaliar_baseline_nn(locais, hospital_base, **params_custo)

    # GA
    melhor_rota, custo_ga, hist_best, hist_mean = executar_algoritmo_genetico(
        locais_entrega=locais,
        hospital_base=hospital_base,
        populacao_inicial=pop_inicial,
        probabilidade_mutacao=taxa_mutacao,
        probabilidade_crossover=taxa_crossover,
        paciencia=DEFAULT_PACIENCIA,
        n_veiculos=n_veiculos,
        **params_custo,
    )

    # VRP Split primeiro (Opção 3: Two-Opt por sub-rota, não sobre o giant tour)
    rotas_vrp_raw = dividir_rotas_vrp(melhor_rota, hospital_base, capacidade, autonomia, n_veiculos)

    # Two-Opt por sub-rota com os mesmos parâmetros de custo
    rotas_vrp, custo_final = two_opt_vrp(rotas_vrp_raw, hospital_base, **params_custo)

    # Giant tour final = concatenação das sub-rotas otimizadas
    melhor_rota_final = [p for sub in rotas_vrp for p in sub]

    resumo_vrp = resumo_restricoes_vrp(rotas_vrp, hospital_base, capacidade, autonomia)

    # Relatório
    relatorio = gerar_relatorio_rota(
        rota_otimizada=melhor_rota_final,
        hospital_base=hospital_base,
        custo_otimizado=custo_final,
        historico_custos=hist_best,
        rota_baseline_nn=rota_nn,
        custo_baseline_nn=custo_nn,
        rotas_vrp=rotas_vrp,
        resumo_vrp=resumo_vrp,
    )

    # Retorna objeto ResultadoGA
    return ResultadoGA(
        hist_best=hist_best,
        hist_mean=hist_mean,
        melhor_rota=melhor_rota_final,
        custo_final=custo_final,
        custo_nn=custo_nn,
        hospital_base=hospital_base,
        locais=locais,
        rotas_vrp=rotas_vrp,
        resumo_vrp=resumo_vrp,
        relatorio=relatorio,
    )
