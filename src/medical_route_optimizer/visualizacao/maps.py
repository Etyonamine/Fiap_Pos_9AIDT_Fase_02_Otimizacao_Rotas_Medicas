import plotly.graph_objects as go
import streamlit as st
from  data.delivery_points import PRIORIDADE_LABEL

_VEHICLE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_PRIO_COLORS   = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4"}
_PRIO_SYMBOLS  = {1: "circle",  2: "square",  3: "diamond"}

def build_route_map(hospital_base, locais, rotas_vrp):
    """Constrói o mapa das rotas."""
    fig = go.Figure()
    # Rotas por veículo
    for idx, rota in enumerate(rotas_vrp):
        cor = _VEHICLE_COLORS[idx % len(_VEHICLE_COLORS)]
        pontos_ciclo = [hospital_base] + rota + [hospital_base]
        xs = [p.coords[0] for p in pontos_ciclo]
        ys = [-p.coords[1] for p in pontos_ciclo]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Veículo {idx+1}", line=dict(color=cor)))
    # Pontos de entrega
    for ponto in locais:
        cor = _PRIO_COLORS.get(ponto.prioridade, "#7f7f7f")
        sym = _PRIO_SYMBOLS.get(ponto.prioridade, "circle")
        label = PRIORIDADE_LABEL.get(ponto.prioridade, "?")
        fig.add_trace(go.Scatter(x=[ponto.coords[0]], y=[-ponto.coords[1]], mode="markers+text",
                                 marker=dict(size=12, color=cor, symbol=sym),
                                 text=[ponto.nome], textposition="top center",
                                 name=f"[{label}] {ponto.nome}", showlegend=False))
    return fig

def render_vrp_cards(resumo_vrp):
    """Renderiza cartões de resumo de cada veículo abaixo do mapa."""
    cols = st.columns(len(resumo_vrp))
    for i, v in enumerate(resumo_vrp):
        status = "✅" if (v["capacidade_ok"] and v["autonomia_ok"]) else "❌"
        with cols[i]:
            st.markdown(f"**Veículo {v['veiculo']} {status}**")
            st.caption(f"{v['n_pontos']} paradas | Carga: {v['peso_total']}/{v['capacidade_veiculo']} | Dist: {v['distancia_pixels']:.0f}/{v['autonomia_veiculo']} px")
