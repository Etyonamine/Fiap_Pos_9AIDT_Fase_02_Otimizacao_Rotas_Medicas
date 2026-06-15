# Streamlit App — Sistema de Otimização de Rotas Médicas

import os
import sys
import numpy as np
import streamlit as st
import pandas as pd

# Garante que o diretório src está no path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Importa serviços e visualização
from services.ga_runner import run_ga
from visualizacao.plots import build_fitness_chart
from visualizacao.maps import build_route_map, render_vrp_cards

# Prompts e perfis para LLM
from llm.prompts_perfil import PERFIS, prompt_por_perfil

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
# Helpers
# ---------------------------------------------------------------------------
def _parse_pesos(raw: str):
    """Converte string de pesos 'w1,w2,w3,w4' em lista de floats."""
    try:
        vals = [float(x.strip()) for x in raw.split(",")]
        while len(vals) < 4:
            vals.append(1.0)
        return vals[:4]
    except Exception:
        return [1.0, 10.0, 500.0, 3.0]

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

    tamanho_pop = st.number_input("Tamanho da população", min_value=10, max_value=500, value=100, step=10)
    n_geracoes = st.number_input("Número de gerações", min_value=10, max_value=1000, value=100, step=10)
    taxa_crossover = st.slider("Taxa de crossover", min_value=0.0, max_value=1.0, value=0.80, step=0.01)
    taxa_mutacao = st.slider("Taxa de mutação", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

    st.markdown("---")
    st.header("Configurações do Fitness")

    st.selectbox("Modo do fitness", ["Minimizar custo total (distância + penalidades)"])
    pesos_raw = st.text_input("Pesos do fitness", value="1.0, 10.0, 500.0, 3.0")
    normalizar = st.checkbox("Normalizar fitness no gráfico", value=True)

    st.markdown("---")
    st.header("Restrições VRP")

    n_veiculos = st.number_input("Nº veículos disponíveis", min_value=1, max_value=10, value=2)
    capacidade = st.number_input("Capacidade/veículo (un.)", min_value=1, max_value=200, value=16)
    autonomia = st.number_input("Autonomia/veículo (px)", min_value=100, max_value=5000, value=1400)

    st.markdown("---")
    st.header("Execução")

    auto_executar = st.checkbox("Executar automaticamente ao carregar", value=False)
    executar = st.button("Executar GA", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Disparo do GA
# ---------------------------------------------------------------------------
pesos = _parse_pesos(pesos_raw)

if executar or (auto_executar and st.session_state["resultado"] is None):
    with st.spinner("⚙️ Executando Algoritmo Genético…"):
        try:
            st.session_state["resultado"] = run_ga(
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
tab_exec, tab_valores, tab_llm = st.tabs(["Execução ao vivo", "Valores por geração", "Assistente LLM"])

# ============================================================================ #
# ABA 1 — Execução ao vivo
# ============================================================================ #
with tab_exec:
    st.header("Execução ao vivo")
    res = st.session_state["resultado"]

    if res is None:
        st.info("Configure os parâmetros na barra lateral e clique em **Executar GA** para iniciar.")
    else:
        col_chart, col_ind = st.columns([3, 1])
        with col_chart:
            st.subheader("Gráficos")
            st.plotly_chart(res.to_plot(normalizar), use_container_width=True)

        with col_ind:
            st.subheader("Indicadores")
            fator = max(res.hist_best) if normalizar and max(res.hist_best) > 0 else 1.0
            melhor = min(res.hist_best) / fator
            media  = float(np.mean(res.hist_mean)) / fator
            desvio = float(np.std(res.hist_best)) / fator
            st.metric("Melhor fitness", f"{melhor:.4f}")
            st.metric("Fitness médio",  f"{media:.4f}")
            st.metric("Desvio padrão",  f"{desvio:.4f}")

        st.subheader("Mapa das Rotas")
        fig_map = build_route_map(res.hospital_base, res.locais, res.rotas_vrp)
        st.plotly_chart(fig_map, use_container_width=True)

        render_vrp_cards(res.resumo_vrp)

# ============================================================================ #
# ABA 2 — Valores por geração
# ============================================================================ #
with tab_valores:
    st.header("Valores por geração")
    res = st.session_state["resultado"]

    if res is None:
        st.info("Execute o GA para ver os valores por geração.")
    else:
        df = res.to_dataframe()
        st.dataframe(df, use_container_width=True, height=350)
        st.info(res.to_summary())

# ============================================================================ #
# ABA 3 — Assistente LLM
# ============================================================================ #
with tab_llm:
    st.header("Assistente LLM")
    res = st.session_state["resultado"]

    if res is None:
        st.info("Execute o GA primeiro para habilitar o assistente.")
    else:
        col_cfg, col_out = st.columns([1, 2])
        with col_cfg:
            st.subheader("Configuração")
            perfil_sel = st.selectbox("Perfil do usuário", list(PERFIS.keys()))
            provider = st.selectbox("Provedor LLM", ["groq", "openai"])
            api_key = st.text_input(f"Chave de API ({provider.upper()})", type="password")
            modelo_custom = st.text_input("Modelo (opcional)", placeholder="Ex.: llama-3.3-70b-versatile")
            gerar = st.button("Gerar análise", type="primary", use_container_width=True)

        with col_out:
            st.subheader("Análise gerada")
            if gerar and api_key.strip():
                env_key = "OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY"
                os.environ[env_key] = api_key.strip()
                os.environ["LLM_PROVIDER"] = provider
                if modelo_custom.strip():
                    os.environ["LLM_MODEL"] = modelo_custom.strip()

                prompt = prompt_por_perfil(perfil_sel, res.relatorio)
                from llm.llm_client import chamar_llm
                resposta = chamar_llm(prompt, max_tokens=2048)
                st.markdown(resposta)
                os.environ.pop(env_key, None)
