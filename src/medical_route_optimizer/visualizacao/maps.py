import plotly.graph_objects as go
import streamlit as st
from  data.delivery_points import PRIORIDADE_LABEL

_VEHICLE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_PRIO_COLORS   = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4"}
_PRIO_SYMBOLS  = {1: "circle",  2: "square",  3: "diamond"}

def build_route_map(hospital_base, locais, rotas_vrp):
    """Constrói o mapa das rotas com numeração, direção e legenda dos pontos à direita."""
    fig = go.Figure()

    # Rotas por veículo
    for idx, rota in enumerate(rotas_vrp):
        cor = _VEHICLE_COLORS[idx % len(_VEHICLE_COLORS)]
        pontos_ciclo = [hospital_base] + rota + [hospital_base]
        xs = [p.coords[0] for p in pontos_ciclo]
        ys = [-p.coords[1] for p in pontos_ciclo]

        # Linhas da rota
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            name=f"Veículo {idx+1}",
            line=dict(color=cor, width=3)
        ))

        # Pontos numerados
        for i, p in enumerate(pontos_ciclo):
            fig.add_trace(go.Scatter(
                x=[p.coords[0]], y=[-p.coords[1]],
                mode="markers+text",
                marker=dict(size=12, color=cor, symbol="circle"),
                text=[str(i)],
                textposition="top center",
                textfont=dict(size=10, color="black"),
                hovertext=f"{p.nome} — Ponto {i}",
                hoverinfo="text",
                showlegend=False
            ))

        # Setas direcionais
        for i in range(len(pontos_ciclo) - 1):
            x0, y0 = pontos_ciclo[i].coords[0], -pontos_ciclo[i].coords[1]
            x1, y1 = pontos_ciclo[i+1].coords[0], -pontos_ciclo[i+1].coords[1]
            fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1,
                arrowwidth=1.5, arrowcolor=cor, opacity=0.7
            )

    # Hospital base
    fig.add_trace(go.Scatter(
        x=[hospital_base.coords[0]], y=[-hospital_base.coords[1]],
        mode="markers+text",
        marker=dict(size=14, color="#000000", symbol="star"),
        text=["Hospital Base"], textposition="bottom center",
        name="Hospital Base"
    ))

    # Layout principal
    fig.update_layout(
        title="Mapa das Rotas — Veículos com Numeração, Direção e Legenda de Pontos",
        xaxis_title="Coordenada X (px)",
        yaxis_title="Coordenada Y (px)",
        height=850,
        margin=dict(l=40, r=200, t=60, b=40),  # espaço extra à direita
        autosize=True,
        legend=dict(x=0.01, y=0.99, bordercolor="black", borderwidth=1),
        dragmode="pan",
        hovermode="closest",
        template="plotly_white"
    )

    # Mantém proporção e orientação
    fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")

    # 🔧 Legenda dos pontos à direita
    legenda_texto = "<b>Legenda dos Pontos</b><br>"
    for idx, rota in enumerate(rotas_vrp):
        legenda_texto += f"<br><b>Veículo {idx+1}</b><br>"
        for i, p in enumerate(rota):
            legenda_texto += f"{i+1}: {p.nome}<br>"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.02, y=0.95,
        text=legenda_texto,
        showarrow=False,
        align="left",
        font=dict(size=11, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )

    return fig



def render_vrp_cards(resumo_vrp):
    """Renderiza cartões de resumo de cada veículo abaixo do mapa."""
    cols = st.columns(len(resumo_vrp))
    for i, v in enumerate(resumo_vrp):
        status = "✅" if (v["capacidade_ok"] and v["autonomia_ok"]) else "❌"
        with cols[i]:
            st.markdown(f"**Veículo {v['veiculo']} {status}**")
            st.caption(f"{v['n_pontos']} paradas | Carga: {v['peso_total']}/{v['capacidade_veiculo']} | Dist: {v['distancia_pixels']:.0f}/{v['autonomia_veiculo']} px")
