from typing import List
from data.delivery_points import get_hospital_base, get_pontos_entrega_sem_origem
from core.genetic_algorithm import executar_algoritmo_genetico, gerar_populacao_aleatoria
from core.nearest_neighbor import gerar_populacao_nearest_neighbor, avaliar_baseline_nn
from core.two_opt import two_opt_inversion
from core.vrp_split import dividir_rotas_vrp, resumo_restricoes_vrp
from reports.route_report import gerar_relatorio_rota
from models.resultado_ga import ResultadoGA
from core.genetic_algorithm import calcular_custo_giant_tour_vrp
from models.resultado_ga import ResultadoGA



PROPORCAO_NN = 0.15

def run_ga(tamanho_pop: int, taxa_crossover: float,
           taxa_mutacao: float, pesos: List[float], n_veiculos: int,
           capacidade: float, autonomia: float) -> ResultadoGA:
    """Executa o pipeline completo de otimização e retorna um objeto ResultadoGA."""
    hospital_base = get_hospital_base()
    locais = get_pontos_entrega_sem_origem()
    pesos_efetivos = (list(pesos) + [1.0, 1.0, 1.0])[:3]

    # População inicial híbrida
    n_nn = max(1, int(tamanho_pop * PROPORCAO_NN))
    pop_nn = gerar_populacao_nearest_neighbor(locais, hospital_base, n_nn)
    pop_aleat = gerar_populacao_aleatoria(locais, tamanho_pop - n_nn)
    pop_inicial = pop_nn + pop_aleat

    rota_nn, custo_nn = avaliar_baseline_nn(locais, hospital_base)

    # GA
    melhor_rota, melhor_custo, hist_best, hist_mean = executar_algoritmo_genetico(
        locais_entrega=locais,
        hospital_base=hospital_base,
        populacao_inicial=pop_inicial,
        probabilidade_mutacao=taxa_mutacao,
        probabilidade_crossover=taxa_crossover,
        paciencia=150,
        n_veiculos=n_veiculos,
        capacidade_veiculo=capacidade,
        autonomia_veiculo=autonomia,
        fator_penalidade=pesos_efetivos[0],
        fator_penalidade_capacidade=pesos_efetivos[1],
        fator_penalidade_autonomia=pesos_efetivos[2],
    )

    # Two Opt
    melhor_rota_final, custo_final = two_opt_inversion(melhor_rota, hospital_base)

    # VRP Split
    rotas_vrp = dividir_rotas_vrp(melhor_rota_final, hospital_base, capacidade, autonomia, n_veiculos)
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
