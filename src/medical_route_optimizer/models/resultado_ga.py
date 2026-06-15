import pandas as pd
import plotly.graph_objects as go

class ResultadoGA:
    def __init__(self, hist_best, hist_mean, melhor_rota, custo_final,
                 custo_nn, hospital_base, locais, rotas_vrp, resumo_vrp, relatorio):
        self.hist_best = hist_best
        self.hist_mean = hist_mean
        self.melhor_rota = melhor_rota
        self.custo_final = custo_final
        self.custo_nn = custo_nn
        self.hospital_base = hospital_base
        self.locais = locais
        self.rotas_vrp = rotas_vrp
        self.resumo_vrp = resumo_vrp
        self.relatorio = relatorio

    # ---- Métodos utilitários ----
    def to_dataframe(self):
        """Retorna DataFrame com evolução do custo por geração."""
        melhoria_acum = [
            round((self.hist_best[0] - v) / self.hist_best[0] * 100, 3) if self.hist_best[0] > 0 else 0.0
            for v in self.hist_best
        ]
        return pd.DataFrame({
            "Geração": range(1, len(self.hist_best) + 1),
            "Melhor custo": [round(v, 4) for v in self.hist_best],
            "Custo médio": [round(v, 4) for v in self.hist_mean],
            "Melhoria (%)": melhoria_acum,
        })

    def to_plot(self, normalizar=True):
        """Retorna gráfico de evolução do fitness."""
        fator = max(self.hist_best) if normalizar and max(self.hist_best) > 0 else 1.0
        y_best = [v / fator for v in self.hist_best]
        y_mean = [v / fator for v in self.hist_mean]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(self.hist_best))), y=y_best, name="best"))
        fig.add_trace(go.Scatter(x=list(range(len(self.hist_mean))), y=y_mean, name="mean"))
        return fig

    def to_summary(self):
        """Resumo textual da execução."""
        conv_gen = len(self.hist_best)
        reducao = round((self.hist_best[0] - self.hist_best[-1]) / self.hist_best[0] * 100, 2) if self.hist_best[0] > 0 else 0
        return f"O GA convergiu em {conv_gen} gerações. Redução de custo: {reducao}% (de {self.hist_best[0]:.2f} para {self.hist_best[-1]:.2f})."
