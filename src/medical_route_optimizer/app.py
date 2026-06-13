"""
Streamlit App — Sistema de Otimização de Rotas Médicas

Interface visual para o motor de Algoritmo Genético (TSP/VRP) aplicado à
distribuição hospitalar de medicamentos e insumos.

Layout:
    Barra lateral  → parâmetros do GA, fitness, restrições VRP, execução
    Aba 1          → Execução ao vivo  (gráfico best/mean, indicadores, mapa de rotas)
    Aba 2          → Valores por geração (tabela + gráfico detalhado)
    Aba 3          → Assistente LLM  (perfil: motorista / operador / gerente)

Execução:
    cd src
    streamlit run medical_route_optimizer/app.py
"""

import os
import sys

# Garante que o diretório src está no path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List

from medical_route_optimizer.data.delivery_points import (
    get_hospital_base,
    get_pontos_entrega_sem_origem,
    PRIORIDADE_LABEL,
    PontoEntrega,
)
from medical_route_optimizer.core.genetic_algorithm import (
    gerar_populacao_aleatoria,
    executar_algoritmo_genetico,
)
from medical_route_optimizer.core.nearest_neighbor import (
    gerar_populacao_nearest_neighbor,
    avaliar_baseline_nn,
)
from medical_route_optimizer.core.two_opt import two_opt_inversion
from medical_route_optimizer.core.vrp_split import (
    dividir_rotas_vrp,
    resumo_restricoes_vrp,
)
from medical_route_optimizer.reports.route_report import gerar_relatorio_rota
from medical_route_optimizer.llm.prompts_perfil import PERFIS, prompt_por_perfil


# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Otimização de Rotas Médicas — GA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cores e constantes visuais
# ---------------------------------------------------------------------------
_VEHICLE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f"]
_PRIO_COLORS   = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4"}
_PRIO_SYMBOLS  = {1: "circle",  2: "square",  3: "diamond"}

PROPORCAO_NN = 0.15   # 15% da população inicial gerada por Nearest Neighbor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pesos(raw: str) -> List[float]:
    """Converte string de pesos 'w1,w2,w3,w4' em lista de floats."""
    try:
        vals = [float(x.strip()) for x in raw.split(",")]
        while len(vals) < 4:
            vals.append(1.0)
        return vals[:4]
    except Exception:
        return [1.0, 10.0, 50.0, 3.0]


def _run_ga(
    tamanho_pop: int,
    n_geracoes: int,
    taxa_crossover: float,
    taxa_mutacao: float,
    pesos: List[float],
    n_veiculos: int,
    capacidade: float,
    autonomia: float,
) -> Dict[str, Any]:
    """
    Executa o pipeline completo de otimização e retorna todos os resultados.

    Pipeline:
        1. Gera população inicial híbrida (NN + aleatória)
        2. Executa Algoritmo Genético com penalidades VRP
        3. Refinamento local: Two Opt Inversion
        4. Particionamento VRP: greedy split
        5. Geração de relatório estruturado
    """
    hospital_base = get_hospital_base()
    locais = get_pontos_entrega_sem_origem()

    # 1. População inicial híbrida
    n_nn = max(1, int(tamanho_pop * PROPORCAO_NN))
    pop_nn = gerar_populacao_nearest_neighbor(locais, hospital_base, n_nn)
    pop_aleat = gerar_populacao_aleatoria(locais, tamanho_pop - n_nn)
    pop_inicial = pop_nn + pop_aleat

    rota_nn, custo_nn = avaliar_baseline_nn(locais, hospital_base)

    # 2. GA
    melhor_rota, melhor_custo, hist_best, hist_mean = executar_algoritmo_genetico(
        locais_entrega=locais,
        hospital_base=hospital_base,
        populacao_inicial=pop_inicial,
        n_geracoes=n_geracoes,
        probabilidade_mutacao=taxa_mutacao,
        probabilidade_crossover=taxa_crossover,
        paciencia=max(20, int(n_geracoes * 0.25)),
        capacidade_veiculo=capacidade,
        autonomia_veiculo=autonomia,
        fator_penalidade=pesos[1],
        fator_penalidade_capacidade=pesos[2],
        fator_penalidade_autonomia=pesos[3],
        verbose=False,
    )

    # 3. Two Opt
    melhor_rota_final, custo_final = two_opt_inversion(melhor_rota, hospital_base)

    # 4. VRP Split
    rotas_vrp = dividir_rotas_vrp(
        giant_tour=melhor_rota_final,
        hospital_base=hospital_base,
        capacidade_veiculo=capacidade,
        autonomia_veiculo=autonomia,
        n_veiculos=n_veiculos,
    )
    resumo_vrp = resumo_restricoes_vrp(
        rotas_vrp, hospital_base, capacidade, autonomia
    )

    # 5. Relatório
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

    return {
        "hist_best": hist_best,
        "hist_mean": hist_mean,
        "melhor_rota": melhor_rota_final,
        "custo_final": custo_final,
        "custo_nn": custo_nn,
        "hospital_base": hospital_base,
        "locais": locais,
        "rotas_vrp": rotas_vrp,
        "resumo_vrp": resumo_vrp,
        "relatorio": relatorio,
    }


def _build_fitness_chart(
    hist_best: List[float],
    hist_mean: List[float],
    normalizar: bool,
) -> go.Figure:
    """Constrói o gráfico de evolução do fitness (best e mean) por geração."""
    geracoes = list(range(len(hist_best)))
    fator = max(hist_best) if normalizar and max(hist_best) > 0 else 1.0
    y_best = [v / fator for v in hist_best]
    y_mean = [v / fator for v in hist_mean]
    y_label = "Fitness (normalizado)" if normalizar else "Fitness (custo)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=geracoes, y=y_best,
        name="best", mode="lines",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=geracoes, y=y_mean,
        name="mean", mode="lines",
        line=dict(color="#aec7e8", width=1.5),
    ))
    fig.update_layout(
        xaxis_title="Geração",
        yaxis_title=y_label,
        legend=dict(orientation="h", y=-0.22),
        margin=dict(t=10, b=50, l=50, r=10),
        height=330,
    )
    return fig


def _build_route_map(
    hospital_base: PontoEntrega,
    locais: List[PontoEntrega],
    rotas_vrp: List[List[PontoEntrega]],
) -> go.Figure:
    """
    Constrói o mapa de rotas em espaço de coordenadas (x, y).

    - Eixo Y invertido para que coordenadas crescentes para baixo fiquem
      visualmente corretas em relação à origem convencional de tela.
    - Cada rota de veículo é desenhada em cor distinta.
    - Pontos coloridos por prioridade (Vermelho=Alta, Laranja=Média, Azul=Baixa).
    - Hospital Base marcado com estrela verde.
    """
    fig = go.Figure()

    # Rotas por veículo
    for idx, rota in enumerate(rotas_vrp):
        cor = _VEHICLE_COLORS[idx % len(_VEHICLE_COLORS)]
        pontos_ciclo = [hospital_base] + rota + [hospital_base]
        xs = [p.coords[0] for p in pontos_ciclo]
        ys = [-p.coords[1] for p in pontos_ciclo]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name=f"Veículo {idx + 1}",
            line=dict(color=cor, width=2.5),
            hoverinfo="skip",
        ))

    # Pontos de entrega (coloridos por prioridade)
    for ponto in locais:
        cor = _PRIO_COLORS.get(ponto.prioridade, "#7f7f7f")
        sym = _PRIO_SYMBOLS.get(ponto.prioridade, "circle")
        label = PRIORIDADE_LABEL.get(ponto.prioridade, "?")
        fig.add_trace(go.Scatter(
            x=[ponto.coords[0]], y=[-ponto.coords[1]],
            mode="markers+text",
            marker=dict(size=12, color=cor, symbol=sym, line=dict(width=1, color="white")),
            text=[ponto.nome],
            textposition="top center",
            textfont=dict(size=9),
            name=f"[{label}] {ponto.nome}",
            hovertemplate=(
                f"<b>{ponto.nome}</b><br>"
                f"Prioridade: {label}<br>"
                f"Peso: {ponto.peso:.1f} un.<br>"
                f"Atendimento: {ponto.tempo_atendimento} min<extra></extra>"
            ),
            showlegend=False,
        ))

    # Hospital Base
    fig.add_trace(go.Scatter(
        x=[hospital_base.coords[0]], y=[-hospital_base.coords[1]],
        mode="markers+text",
        marker=dict(size=18, color="#2ca02c", symbol="star",
                    line=dict(width=1, color="white")),
        text=[hospital_base.nome],
        textposition="bottom center",
        textfont=dict(size=10, color="#2ca02c"),
        name="Hospital Base",
        hovertemplate=f"<b>{hospital_base.nome}</b><br>Origem / Retorno<extra></extra>",
    ))

    # Legendas de prioridade (entradas fantasma)
    for prio, label in PRIORIDADE_LABEL.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=_PRIO_COLORS[prio], symbol=_PRIO_SYMBOLS[prio]),
            name=f"Prioridade {label}",
            showlegend=True,
        ))

    fig.update_layout(
        height=480,
        xaxis=dict(showgrid=True, title="X (coordenada)"),
        yaxis=dict(showgrid=True, title="Y (coordenada)", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="v", x=1.01, y=1),
        margin=dict(t=10, b=20, l=20, r=160),
    )
    return fig


def _render_vrp_cards(resumo_vrp: List[dict]) -> None:
    """Renderiza cartões de resumo de cada veículo abaixo do mapa."""
    cols = st.columns(len(resumo_vrp))
    for i, v in enumerate(resumo_vrp):
        status = "✅" if (v["capacidade_ok"] and v["autonomia_ok"]) else "❌"
        with cols[i]:
            st.markdown(f"**Veículo {v['veiculo']} {status}**")
            cap_icon = "✅" if v["capacidade_ok"] else "❌"
            aut_icon = "✅" if v["autonomia_ok"] else "❌"
            st.caption(
                f"{v['n_pontos']} paradas | "
                f"Carga: {v['peso_total']}/{v['capacidade_veiculo']} {cap_icon} | "
                f"Dist: {v['distancia_pixels']:.0f}/{v['autonomia_veiculo']} px {aut_icon}"
            )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "resultado" not in st.session_state:
    st.session_state["resultado"] = None


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configurações do Algoritmo Genético")

    tamanho_pop = st.number_input(
        "Tamanho da população", min_value=10, max_value=500, value=100, step=10
    )
    n_geracoes = st.number_input(
        "Número de gerações", min_value=10, max_value=1000, value=100, step=10
    )
    taxa_crossover = st.slider(
        "Taxa de crossover", min_value=0.0, max_value=1.0, value=0.80, step=0.01,
        help="Probabilidade de aplicar OX crossover; caso contrário, clona o pai 1."
    )
    taxa_mutacao = st.slider(
        "Taxa de mutação", min_value=0.0, max_value=1.0, value=0.10, step=0.01,
        help="Probabilidade de troca de posições adjacentes por indivíduo."
    )

    st.markdown("---")
    st.header("Configurações do Fitness")

    st.selectbox(
        "Modo do fitness",
        ["Minimizar custo total (distância + penalidades)"],
        help="O GA minimiza sempre; as penalidades moldam a pressão sobre cada restrição.",
    )
    pesos_raw = st.text_input(
        "Pesos do fitness",
        value="1.0, 10.0, 50.0, 3.0",
        help="Quatro pesos separados por vírgula: distância, prioridade, capacidade, autonomia",
        placeholder="w_dist, w_prior, w_cap, w_aut",
    )
    normalizar = st.checkbox(
        "Normalizar fitness no gráfico", value=True,
        help="Divide os valores pelo máximo para exibir no intervalo [0,1]."
    )

    st.markdown("---")
    st.header("Restrições VRP")

    n_veiculos = st.number_input(
        "Nº veículos disponíveis", min_value=1, max_value=10, value=2
    )
    capacidade = st.number_input(
        "Capacidade/veículo (un.)", min_value=1, max_value=200, value=16,
        help="Carga máxima que cada veículo pode transportar."
    )
    autonomia = st.number_input(
        "Autonomia/veículo (px)", min_value=100, max_value=5000, value=1400,
        help="Distância máxima (em pixels) que cada veículo pode percorrer por ciclo."
    )

    st.markdown("---")
    st.header("Execução")

    auto_executar = st.checkbox("Executar automaticamente ao carregar", value=False)
    executar = st.button("Executar GA", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption(
        "**Dicas**: teste diferentes taxas de mutação; monitore diversidade; "
        "peça ao LLM análises sobre overfitting ou estagnação."
    )

# ---------------------------------------------------------------------------
# Disparo do GA
# ---------------------------------------------------------------------------
pesos = _parse_pesos(pesos_raw)

if executar or (auto_executar and st.session_state["resultado"] is None):
    with st.spinner("⚙️ Executando Algoritmo Genético…"):
        try:
            st.session_state["resultado"] = _run_ga(
                tamanho_pop=int(tamanho_pop),
                n_geracoes=int(n_geracoes),
                taxa_crossover=taxa_crossover,
                taxa_mutacao=taxa_mutacao,
                pesos=pesos,
                n_veiculos=int(n_veiculos),
                capacidade=float(capacidade),
                autonomia=float(autonomia),
            )
        except Exception as exc:
            st.error(f"Erro durante a execução do GA: {exc}")
            st.stop()

# ---------------------------------------------------------------------------
# Abas principais
# ---------------------------------------------------------------------------
tab_exec, tab_valores, tab_llm = st.tabs(
    ["Execução ao vivo", "Valores por geração", "Assistente LLM"]
)

# ============================================================================
# ABA 1 — Execução ao vivo
# ============================================================================
with tab_exec:
    st.header("Execução ao vivo")
    res = st.session_state["resultado"]

    if res is None:
        st.info(
            "Configure os parâmetros na barra lateral e clique em **Executar GA** para iniciar."
        )
    else:
        hist_best = res["hist_best"]
        hist_mean = res["hist_mean"]

        # ── Gráfico + Indicadores ──────────────────────────────────────────
        col_chart, col_ind = st.columns([3, 1])

        with col_chart:
            st.subheader("Gráficos")
            fig_fitness = _build_fitness_chart(hist_best, hist_mean, normalizar)
            st.plotly_chart(fig_fitness, use_container_width=True)

        with col_ind:
            st.subheader("Indicadores")
            fator = max(hist_best) if normalizar and max(hist_best) > 0 else 1.0
            melhor = min(hist_best) / fator
            media  = float(np.mean(hist_mean)) / fator
            desvio = float(np.std(hist_best)) / fator
            st.metric("Melhor fitness", f"{melhor:.4f}")
            st.metric("Fitness médio",  f"{media:.4f}")
            st.metric("Desvio padrão",  f"{desvio:.4f}")

        # ── Status ────────────────────────────────────────────────────────
        vrp_ok = all(v["capacidade_ok"] and v["autonomia_ok"] for v in res["resumo_vrp"])
        comp = res["relatorio"]["comparacao_baseline_nn"]
        economia = comp.get("economia_percentual")
        msg = (
            f"Execução finalizada | Custo GA+2-Opt: **{res['custo_final']:.2f}** | "
            f"Veículos: {len(res['rotas_vrp'])} | Gerações: {len(hist_best)}"
            + (f" | Economia vs NN: **{economia}%**" if economia is not None else "")
        )
        if vrp_ok:
            st.success(msg)
        else:
            st.warning(msg + " | ⚠️ Alguma restrição VRP foi violada.")

        # ── Mapa de Rotas ─────────────────────────────────────────────────
        st.subheader("Mapa das Rotas")
        fig_map = _build_route_map(
            res["hospital_base"], res["locais"], res["rotas_vrp"]
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Cartões de resumo por veículo
        _render_vrp_cards(res["resumo_vrp"])

# ============================================================================
# ABA 2 — Valores por geração
# ============================================================================
with tab_valores:
    st.header("Valores por geração")
    res = st.session_state["resultado"]

    if res is None:
        st.info("Execute o GA para ver os valores por geração.")
    else:
        hist_best = res["hist_best"]
        hist_mean = res["hist_mean"]

        # Tabela detalhada
        melhoria_acum = [
            round((hist_best[0] - v) / hist_best[0] * 100, 3) if hist_best[0] > 0 else 0.0
            for v in hist_best
        ]
        df = pd.DataFrame({
            "Geração":         range(1, len(hist_best) + 1),
            "Melhor custo":    [round(v, 4) for v in hist_best],
            "Custo médio":     [round(v, 4) for v in hist_mean],
            "Melhoria (%)":    melhoria_acum,
        })
        st.dataframe(df, use_container_width=True, height=350)

        # Gráfico detalhado (raw, sem normalização)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["Geração"], y=df["Melhor custo"],
            name="Melhor custo", mode="lines+markers",
            marker=dict(size=4),
            line=dict(color="#1f77b4"),
        ))
        fig2.add_trace(go.Scatter(
            x=df["Geração"], y=df["Custo médio"],
            name="Custo médio", mode="lines",
            line=dict(color="#aec7e8"),
        ))

        # Linha de referência: custo NN baseline
        if res.get("custo_nn"):
            fig2.add_hline(
                y=res["custo_nn"], line_dash="dash",
                line_color="orange",
                annotation_text="Baseline NN",
                annotation_position="bottom right",
            )

        fig2.update_layout(
            xaxis_title="Geração",
            yaxis_title="Custo (raw)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=20, b=50),
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Resumo textual da convergência
        conv_gen = len(hist_best)
        reducao = round((hist_best[0] - hist_best[-1]) / hist_best[0] * 100, 2) if hist_best[0] > 0 else 0
        st.info(
            f"O GA convergiu em **{conv_gen}** gerações. "
            f"Redução de custo durante evolução: **{reducao}%** "
            f"(de {hist_best[0]:.2f} para {hist_best[-1]:.2f})."
        )

# ============================================================================
# ABA 3 — Assistente LLM
# ============================================================================
with tab_llm:
    st.header("Assistente LLM")
    res = st.session_state["resultado"]

    if res is None:
        st.info("Execute o GA primeiro para habilitar o assistente.")
    else:
        col_cfg, col_out = st.columns([1, 2])

        with col_cfg:
            st.subheader("Configuração")
            perfil_sel = st.selectbox(
                "Perfil do usuário",
                list(PERFIS.keys()),
                help="Cada perfil recebe uma análise especializada da rota otimizada.",
            )
            provider = st.selectbox(
                "Provedor LLM",
                ["groq", "openai"],
                help="Groq oferece acesso gratuito com llama-3.3-70b.",
            )
            api_key = st.text_input(
                f"Chave de API ({provider.upper()})",
                type="password",
                placeholder="Cole sua chave aqui…",
                help="A chave é usada apenas nesta sessão e não é armazenada.",
            )
            modelo_custom = st.text_input(
                "Modelo (opcional)",
                placeholder="Ex.: llama-3.3-70b-versatile",
                help="Deixe em branco para usar o modelo padrão do provedor.",
            )
            gerar = st.button("Gerar análise", type="primary", use_container_width=True)

            # Dica por perfil
            descricoes = {
                "motorista": "📋 Roteiro operacional de entregas do dia.",
                "operador":  "🖥️ Painel de monitoramento de restrições VRP.",
                "gerente":   "📊 Relatório executivo com KPIs e recomendações.",
            }
            chave_interna = PERFIS.get(perfil_sel, "motorista")
            st.caption(descricoes.get(chave_interna, ""))

        with col_out:
            st.subheader("Análise gerada")

            if gerar:
                if not api_key.strip():
                    st.error("⚠️ Informe a chave de API para gerar a análise.")
                else:
                    # Configura variáveis de ambiente para o cliente LLM
                    env_key = "OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY"
                    os.environ[env_key] = api_key.strip()
                    os.environ["LLM_PROVIDER"] = provider
                    if modelo_custom.strip():
                        os.environ["LLM_MODEL"] = modelo_custom.strip()
                    elif "LLM_MODEL" in os.environ:
                        del os.environ["LLM_MODEL"]

                    prompt = prompt_por_perfil(perfil_sel, res["relatorio"])

                    with st.spinner(f"Consultando {provider.upper()}…"):
                        try:
                            from medical_route_optimizer.llm.llm_client import chamar_llm
                            resposta = chamar_llm(prompt, max_tokens=2048)
                            st.markdown(resposta)
                        except (EnvironmentError, ValueError) as exc:
                            st.error(f"Erro de configuração: {exc}")
                        except Exception as exc:
                            st.error(f"Erro ao consultar LLM: {exc}")

                    # Expander com o prompt enviado (transparência/auditoria)
                    with st.expander("🔍 Ver prompt enviado ao modelo"):
                        st.text(prompt)
            else:
                st.caption(
                    "Selecione um perfil, configure o provedor e clique em "
                    "**Gerar análise**."
                )
