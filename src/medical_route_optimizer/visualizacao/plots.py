import plotly.graph_objects as go

def build_fitness_chart(hist_best, hist_mean, normalizar=True):
    """Constrói gráfico de evolução do fitness."""
    geracoes = list(range(len(hist_best)))
    fator = max(hist_best) if normalizar and max(hist_best) > 0 else 1.0
    y_best = [v / fator for v in hist_best]
    y_mean = [v / fator for v in hist_mean]
    y_label = "Fitness (normalizado)" if normalizar else "Fitness (custo)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=geracoes, y=y_best, name="best", mode="lines"))
    fig.add_trace(go.Scatter(x=geracoes, y=y_mean, name="mean", mode="lines"))
    fig.update_layout(xaxis_title="Geração", yaxis_title=y_label)
    return fig
