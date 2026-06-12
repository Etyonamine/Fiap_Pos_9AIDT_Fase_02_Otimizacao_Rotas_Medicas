"""
Geração de relatório estruturado da rota otimizada.

Transforma o resultado do pipeline de otimização em um dicionário
padronizado, pronto para consumo pela camada de integração com LLM.
"""

from typing import List, Dict, Any, Optional

from medical_route_optimizer.data.delivery_points import PontoEntrega
from medical_route_optimizer.core.genetic_algorithm import calcular_distancia


# Velocidade média estimada para cálculo de tempo de deslocamento (km/h → pixels/min)
# Em ambiente de demonstração, tratamos 1 unidade de pixel ≈ 0.1 km
VELOCIDADE_MEDIA_PIXELS_POR_MINUTO = 5.0


def _tempo_deslocamento(distancia: float) -> float:
    """Converte distância em pixels para tempo estimado de deslocamento em minutos."""
    return distancia / VELOCIDADE_MEDIA_PIXELS_POR_MINUTO


def gerar_relatorio_rota(
    rota_otimizada: List[PontoEntrega],
    hospital_base: PontoEntrega,
    custo_otimizado: float,
    historico_custos: List[float],
    rota_baseline_nn: Optional[List[PontoEntrega]] = None,
    custo_baseline_nn: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Gera um relatório estruturado com os dados da rota otimizada.

    Parâmetros:
    - rota_otimizada: melhor rota encontrada pelo GA + Two Opt
    - hospital_base: ponto de origem e retorno
    - custo_otimizado: custo total da rota otimizada
    - historico_custos: custo por geração do GA
    - rota_baseline_nn: rota gerada pelo Nearest Neighbor (baseline)
    - custo_baseline_nn: custo da rota baseline NN

    Retorno:
    - dicionário estruturado com todas as métricas da rota
    """
    rota_completa = [hospital_base] + list(rota_otimizada) + [hospital_base]
    n = len(rota_completa)

    # Distância pura (sem penalidade de prioridade)
    distancia_total = sum(
        calcular_distancia(rota_completa[i], rota_completa[i + 1])
        for i in range(n - 1)
    )
    tempo_total_estimado = _tempo_deslocamento(distancia_total) + sum(
        p.tempo_atendimento for p in rota_otimizada
    )

    # Sequência detalhada de atendimentos
    sequencia_atendimentos = []
    tempo_acumulado = 0.0
    ponto_anterior = hospital_base
    for posicao, ponto in enumerate(rota_otimizada, start=1):
        deslocamento = calcular_distancia(ponto_anterior, ponto)
        tempo_desloc = _tempo_deslocamento(deslocamento)
        tempo_acumulado += tempo_desloc + ponto.tempo_atendimento
        sequencia_atendimentos.append({
            "posicao": posicao,
            "nome": ponto.nome,
            "coords": ponto.coords,
            "prioridade": ponto.prioridade,
            "prioridade_label": {1: "Alta", 2: "Média", 3: "Baixa"}.get(ponto.prioridade, "-"),
            "tempo_atendimento_min": ponto.tempo_atendimento,
            "tempo_deslocamento_min": round(tempo_desloc, 1),
            "tempo_acumulado_min": round(tempo_acumulado, 1),
        })
        ponto_anterior = ponto

    # Métricas de comparação com baseline NN
    economia_percentual = None
    if custo_baseline_nn and custo_baseline_nn > 0:
        economia_percentual = round(
            (custo_baseline_nn - custo_otimizado) / custo_baseline_nn * 100, 2
        )

    # Contagem de pacientes por prioridade
    alta_prioridade = [p for p in rota_otimizada if p.prioridade == 1]
    media_prioridade = [p for p in rota_otimizada if p.prioridade == 2]
    baixa_prioridade = [p for p in rota_otimizada if p.prioridade == 3]

    relatorio = {
        "resumo": {
            "total_pontos_entrega": len(rota_otimizada),
            "distancia_total_pixels": round(distancia_total, 2),
            "tempo_total_estimado_min": round(tempo_total_estimado, 1),
            "custo_otimizado": round(custo_otimizado, 2),
            "n_geracoes_ga": len(historico_custos),
            "custo_inicial_ga": round(historico_custos[0], 2) if historico_custos else None,
            "custo_final_ga": round(historico_custos[-1], 2) if historico_custos else None,
        },
        "origem_retorno": hospital_base.nome,
        "sequencia_atendimentos": sequencia_atendimentos,
        "prioridades": {
            "alta": [p.nome for p in alta_prioridade],
            "media": [p.nome for p in media_prioridade],
            "baixa": [p.nome for p in baixa_prioridade],
        },
        "comparacao_baseline_nn": {
            "custo_baseline_nn": round(custo_baseline_nn, 2) if custo_baseline_nn else None,
            "custo_otimizado": round(custo_otimizado, 2),
            "economia_percentual": economia_percentual,
            "ga_superou_nn": (custo_otimizado < custo_baseline_nn) if custo_baseline_nn else None,
        },
        "historico_evolucao": {
            "melhor_custo_por_geracao": [round(c, 2) for c in historico_custos],
        },
    }

    return relatorio
