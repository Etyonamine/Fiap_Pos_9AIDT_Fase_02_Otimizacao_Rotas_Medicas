# app.py — Streamlit App — Sistema de Otimização de Rotas Médicas (versão final, seguro)
import os
import sys
import json
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import traceback

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


def extract_first_json(text: str):
    """
    Extrai o primeiro objeto JSON válido encontrado em `text`.
    Retorna o objeto Python (dict/list) ou None se não encontrar/parsear.
    Estratégia: encontra a primeira '{' e busca o '}' correspondente por contagem de chaves.
    Em caso de falha no json.loads, tenta substituir aspas simples por duplas como último recurso.
    """
    if not isinstance(text, str):
        return None
    start = text.find('{')
    if start == -1:
        return None
    stack = []
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            stack.append(i)
        elif ch == '}':
            if stack:
                stack.pop()
            if not stack:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # tentativa de correção simples: aspas simples -> duplas
                    candidate_fixed = candidate.replace("'", '"')
                    try:
                        return json.loads(candidate_fixed)
                    except json.JSONDecodeError:
                        return None
    return None


# Validação do relatorio (capacidade/autonomia)
def relatorio_valido(relatorio: dict) -> (bool, str):
    for v in relatorio.get("veiculos", []):
        if v.get("carga_atual", 0) > v.get("capacidade_max", float("inf")):
            return False, f"Capacidade excedida no veículo {v.get('id')}: {v.get('carga_atual')} / {v.get('capacidade_max')}"
        if v.get("distancia_percorrida_km", 0) > v.get("autonomia", float("inf")):
            return False, f"Autonomia excedida no veículo {v.get('id')}: {v.get('distancia_percorrida_km')} / {v.get('autonomia')}"
    return True, "OK"


# Agrega tempos por prioridade e preenche tempo de retorno se possível
def agregar_tempos_por_prioridade(relatorio: dict) -> None:
    totals = {"alta": 0.0, "media": 0.0, "baixa": 0.0}
    for v in relatorio.get("veiculos", []):
        for p in v.get("rota", []):
            pr = p.get("prioridade", "").lower()
            tempo = float(p.get("tempo_desloc_min", 0)) + float(p.get("tempo_atend_min", 0))
            if pr in totals:
                totals[pr] += tempo
    relatorio.setdefault("totais", {})["tempo_por_prioridade_min"] = totals
    relatorio["totais"]["tempo_retorno_min"] = relatorio["totais"].get("tempo_retorno_min", "dado não disponível")


# Aplica ações sugeridas (mock): reatribui pontos entre veículos no relatorio local
def aplicar_actions_suggested(relatorio: dict, actions: list) -> dict:
    """
    Implementação simples para testes: aplica reassign_point e reduce_load no relatorio.
    Retorna relatorio modificado.
    """
    for act in actions:
        try:
            if not isinstance(act, dict):
                continue
            if act.get("action") == "reassign_point":
                point = act.get("point")
                from_v = act.get("from_vehicle")
                to_v = act.get("to_vehicle")
                # remover do veículo from_v
                for v in relatorio.get("veiculos", []):
                    if v.get("id") == from_v:
                        rota = v.get("rota", [])
                        for i, p in enumerate(list(rota)):
                            nome_p = p.get("nome") or p.get("point") or ""
                            if nome_p == point:
                                rota.pop(i)
                                v["carga_atual"] = max(0.0, v.get("carga_atual", 0) - 1.0)
                                break
                # adicionar ao veículo to_v (append)
                for v in relatorio.get("veiculos", []):
                    if v.get("id") == to_v:
                        v.setdefault("rota", []).append({"nome": point, "prioridade": "media", "tempo_desloc_min": 10, "tempo_atend_min": 10})
                        v["carga_atual"] = v.get("carga_atual", 0) + 1.0
            elif act.get("action") == "reduce_load":
                vehicle = act.get("vehicle")
                amount = float(act.get("amount", 0))
                for v in relatorio.get("veiculos", []):
                    if v.get("id") == vehicle:
                        v["carga_atual"] = max(0.0, v.get("carga_atual", 0) - amount)
        except Exception:
            # não falhar a aplicação de ações por causa de um item malformado
            continue
    return relatorio


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "resultado" not in st.session_state:
    st.session_state["resultado"] = None
if "historico_execucoes" not in st.session_state:
    st.session_state["historico_execucoes"] = []
if "historico_execucoes_meta" not in st.session_state:
    st.session_state["historico_execucoes_meta"] = []
if "auditoria" not in st.session_state:
    st.session_state["auditoria"] = []
if "liberacao_manual" not in st.session_state:
    st.session_state["liberacao_manual"] = False

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
            # reset manual release flag after new GA run
            st.session_state["liberacao_manual"] = False
        except Exception as exc:
            st.error(f"Erro durante a execução do GA: {repr(exc)}")
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

        # Cálculo e exibição do balanceamento
        bal = res.calcular_balanceamento()

        st.markdown("### ⚖️ Balanceamento das Rotas")
        st.progress(bal["balanceamento_pontos"] / 100)
        st.caption(f"Distribuição de pontos: {bal['balanceamento_pontos']} %")

        st.progress(bal["balanceamento_distancia"] / 100)
        st.caption(f"Distribuição de distância: {bal['balanceamento_distancia']} %")

        st.progress(bal["balanceamento_carga"] / 100)
        st.caption(f"Distribuição de carga: {bal['balanceamento_carga']} %")

        # -----------------------------------------------------------------------
        # Histórico de execuções
        # -----------------------------------------------------------------------
        st.session_state["historico_execucoes"].append({
            "timestamp": pd.Timestamp.now(),
            "balanceamento": bal,
            "melhor_custo": res.custo_final,
            "geracoes": len(res.hist_best),
            "populacao": tamanho_pop,
            "taxa_crossover": taxa_crossover,
            "taxa_mutacao": taxa_mutacao,
            "n_veiculos": n_veiculos,
            "capacidade": capacidade,
            "autonomia": autonomia
        })

        st.markdown("### 🕒 Histórico de Execuções do GA")
        df_hist = pd.DataFrame(st.session_state["historico_execucoes"])
        df_hist["timestamp"] = df_hist["timestamp"].dt.strftime("%d/%m/%Y %H:%M:%S")
        st.dataframe(df_hist, use_container_width=True, height=250)

        # Botão para limpar histórico
        if st.button("🗑️ Limpar histórico"):
            st.session_state["historico_execucoes"] = []
            st.success("Histórico limpo com sucesso!")

        # Botão para exportar CSV
        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Exportar histórico em CSV",
            data=csv,
            file_name="historico_execucoes.csv",
            mime="text/csv",
            use_container_width=True
        )

        # -----------------------------------------------------------------------
        # Radar comparativo de execuções
        # -----------------------------------------------------------------------
        fig_radar = go.Figure()

        # Paleta de cores para diferenciar cada execução
        cores = ["royalblue", "darkorange", "green", "purple", "red", "teal", "magenta", "gold"]

        for i, execucao in enumerate(st.session_state["historico_execucoes"]):
            bal_exec = execucao["balanceamento"]
            cor = cores[i % len(cores)]  # alterna as cores automaticamente
            fig_radar.add_trace(go.Scatterpolar(
                r=[
                    bal_exec["balanceamento_pontos"],
                    bal_exec["balanceamento_distancia"],
                    bal_exec["balanceamento_carga"]
                ],
                theta=["Pontos", "Distância", "Carga"],
                fill="toself",
                name=f"Execução {i+1} ({execucao['timestamp']})",
                line={"color": cor, "width": 2},
                opacity=0.7
            ))

        fig_radar.update_layout(
            polar={
                "radialaxis": {
                    "visible": True,
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 100],
                    "tickfont": {"size": 10}
                }
            },
            title="📊 Comparativo de Balanceamento — Histórico de Execuções",
            legend={"x": 0.85, "y": 1.1, "bordercolor": "black", "borderwidth": 1}
        )

        st.plotly_chart(fig_radar, use_container_width=True)

# ============================================================================ #
# ABA 3 — Assistente LLM (com bloqueio para Motorista e parsing robusto)
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
                try:
                    env_key = "OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY"
                    os.environ[env_key] = api_key.strip()
                    os.environ["LLM_PROVIDER"] = provider
                    if modelo_custom.strip():
                        os.environ["LLM_MODEL"] = modelo_custom.strip()

                    # Preparar relatorio serializável a partir do objeto res
                    rel = getattr(res, "relatorio", None)
                    if rel is None:
                        st.error("Relatório da rota não encontrado no resultado. Verifique a função gerar_relatorio_rota().")
                    else:
                        # Agregar tempos por prioridade
                        agregar_tempos_por_prioridade(rel)

                        chave_interna = PERFIS.get(perfil_sel, perfil_sel)

                        # BLOQUEIO: se perfil for motorista e relatorio inválido, NÃO gerar roteiro acionável
                        if chave_interna == "motorista":
                            valid, motivo = relatorio_valido(rel)
                            if not valid and not st.session_state.get("liberacao_manual", False):
                                st.error(f"SOLUÇÃO INVÁLIDA — AÇÃO NECESSÁRIA: {motivo}")
                                st.markdown("**Checklist informativo (NÃO EXECUTAR até correção):**")
                                # Exibir checklist informativo em texto (não JSON)
                                for v in rel.get("veiculos", []):
                                    for p in v.get("rota", []):
                                        st.markdown(f"- **{p.get('nome')}**")
                                        st.markdown("  - Confirmar chegada")
                                        st.markdown("  - Registrar horário de início")
                                        st.markdown("  - Entregar medicamento/insumo")
                                        st.markdown("  - Coletar assinatura/confirmacao")
                                        st.markdown("  - Registrar horário de saída")
                                        st.markdown("  - Atualizar status no sistema")
                                # Mostrar ações sugeridas se já existirem (exibir com segurança)
                                if rel.get("ACTIONS_SUGGESTED"):
                                    st.markdown("**Ações sugeridas (automação):**")
                                    try:
                                        st.json(rel["ACTIONS_SUGGESTED"])
                                    except Exception:
                                        st.text(json.dumps(rel["ACTIONS_SUGGESTED"], ensure_ascii=False))
                                # Botões para operador aplicar ações ou forçar liberação
                                st.markdown("---")
                                st.write("Ações disponíveis:")
                                if st.button("Aplicar ações sugeridas e recomputar GA"):
                                    actions = rel.get("ACTIONS_SUGGESTED", {}).get("ACTIONS_SUGGESTED", []) if isinstance(rel.get("ACTIONS_SUGGESTED"), dict) else rel.get("ACTIONS_SUGGESTED", [])
                                    if actions:
                                        with st.spinner("Aplicando ações sugeridas..."):
                                            rel_mod = aplicar_actions_suggested(rel, actions)
                                            st.success("Ações aplicadas localmente. Reexecute o GA para recalcular a solução.")
                                    else:
                                        st.info("Nenhuma ação sugerida disponível para aplicar.")
                                st.markdown("**Forçar liberação (somente com autorização)**")
                                if st.button("Solicitar liberação manual"):
                                    justificativa = st.text_area("Justificativa para liberação (obrigatório)")
                                    if justificativa and st.button("Confirmar liberação manual"):
                                        st.session_state["auditoria"].append({
                                            "acao": "liberacao_manual",
                                            "usuario": "operador",
                                            "justificativa": justificativa,
                                            "timestamp": pd.Timestamp.now()
                                        })
                                        st.session_state["liberacao_manual"] = True
                                        st.success("Liberação manual registrada. Agora você pode gerar o roteiro (use 'Gerar análise' novamente).")
                                # Não gerar roteiro enquanto inválido e sem liberação
                            else:
                                # Se válido ou liberado manualmente, gerar roteiro normalmente
                                prompt = prompt_por_perfil(perfil_sel, rel)
                                from llm.llm_client import chamar_llm                                
                                try:
                                    # bloco amplo: chamar LLM e todo o processamento que vem depois
                                    resp = chamar_llm(prompt, max_tokens=2048, return_metadata=True)
                                    # processamento que normalmente faz com resp (ex.: texto = resp["text"]; parsing; st.markdown(...))
                                    texto = resp["text"] if isinstance(resp, dict) else resp
                                    # exemplo de processamento que pode falhar:
                                    # st.markdown(texto)
                                    # parsing de JSON, exibição de ACTIONS_SUGGESTED, etc.
                                    # coloque aqui TODO o código que normalmente executa após chamar_llm
                                except Exception as e:
                                    tb = traceback.format_exc()
                                    # exibir traceback completo na UI e salvar em arquivo para análise
                                    st.error("Traceback completo (copie e cole aqui):")
                                    st.text(tb)
                                    with open("llm_error_traceback.log", "a", encoding="utf-8") as f:
                                        f.write(tb + "\n\n")
                                    # re-raise se quiser parar a execução
                                    raise
                                # Exibir texto retornado pelo LLM sem formatar com placeholders
                                if isinstance(texto, str):
                                    st.markdown(texto)
                                else:
                                    st.text(json.dumps(texto, ensure_ascii=False))
                                meta = resp.get("meta") if isinstance(resp, dict) else None
                                if meta:
                                    st.session_state["historico_execucoes_meta"].append({
                                        "timestamp": pd.Timestamp.now(),
                                        "provider": meta.get("provider"),
                                        "model": meta.get("model"),
                                        "prompt": meta.get("prompt")[:10000]
                                    })
                                # parsing robusto: extrair primeiro JSON válido e, se for ACTIONS_SUGGESTED ou TEMPO_POR_PRIORIDADE, armazenar
                                try:
                                    maybe = extract_first_json(texto if isinstance(texto, str) else json.dumps(texto, ensure_ascii=False))
                                    if maybe:
                                        # se o JSON extraído contém ACTIONS_SUGGESTED, use-o
                                        if isinstance(maybe, dict) and "ACTIONS_SUGGESTED" in maybe:
                                            rel["ACTIONS_SUGGESTED"] = maybe["ACTIONS_SUGGESTED"]
                                        else:
                                            # se o JSON extraído parece ser o próprio bloco ACTIONS_SUGGESTED
                                            if isinstance(maybe, dict) and ("VALID" in maybe or "ACTIONS_SUGGESTED" in maybe):
                                                rel["ACTIONS_SUGGESTED"] = maybe
                                            # se for TEMPO_POR_PRIORIDADE
                                            if isinstance(maybe, dict) and any(k in maybe for k in ["alta_min", "media_min", "baixa_min", "alta", "media", "baixa"]):
                                                rel.setdefault("totais", {})["tempo_por_prioridade_min"] = maybe
                                except Exception:
                                    # parsing best-effort; não falhar a UI
                                    pass

                        else:
                            # Perfil operador/gerente — gerar normalmente e registrar metadados
                            prompt = prompt_por_perfil(perfil_sel, rel)
                            from llm.llm_client import chamar_llm
                            resp = chamar_llm(prompt, max_tokens=2048, return_metadata=True)
                            texto = resp["text"] if isinstance(resp, dict) else resp
                            if isinstance(texto, str):
                                st.markdown(texto)
                            else:
                                st.text(json.dumps(texto, ensure_ascii=False))
                            meta = resp.get("meta") if isinstance(resp, dict) else None
                            if meta:
                                st.session_state["historico_execucoes_meta"].append({
                                    "timestamp": pd.Timestamp.now(),
                                    "provider": meta.get("provider"),
                                    "model": meta.get("model"),
                                    "prompt": meta.get("prompt")[:10000]
                                })

                except Exception as e:
                    # usar repr para evitar formatação insegura
                    st.error(f"Erro ao gerar análise: {repr(e)}")
                finally:
                    try:
                        os.environ.pop(env_key, None)
                    except Exception:
                        pass
