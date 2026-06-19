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
        """Retorna DataFrame com evolução do custo por geração, incluindo melhorias acumuladas, por geração, delta e média."""
        if not self.hist_best or not self.hist_mean:
            return pd.DataFrame(columns=[
                "Geração", "Melhor custo", "Custo médio",
                "Melhoria acumulada best (%)", "Melhoria geração best (%)",
                "Melhoria acumulada mean (%)", "Melhoria geração mean (%)",
                "Delta custo best", "Delta custo mean"
            ])

        # Melhoria acumulada (best e mean)
        melhoria_acum_best = [
            round((self.hist_best[0] - v) / self.hist_best[0] * 100, 6)
            for v in self.hist_best
        ]
        melhoria_acum_mean = [
            round((self.hist_mean[0] - v) / self.hist_mean[0] * 100, 6)
            for v in self.hist_mean
        ]

        # Melhoria incremental (best e mean)
        melhoria_geracao_best = [
            0.0 if i == 0 else round((self.hist_best[i-1] - v) / self.hist_best[i-1] * 100, 6)
            for i, v in enumerate(self.hist_best)
        ]
        melhoria_geracao_mean = [
            0.0 if i == 0 else round((self.hist_mean[i-1] - v) / self.hist_mean[i-1] * 100, 6)
            for i, v in enumerate(self.hist_mean)
        ]

        # Delta absoluto (best e mean)
        delta_best = [
            0.0 if i == 0 else round(self.hist_best[i-1] - v, 6)
            for i, v in enumerate(self.hist_best)
        ]
        delta_mean = [
            0.0 if i == 0 else round(self.hist_mean[i-1] - v, 6)
            for i, v in enumerate(self.hist_mean)
        ]

        return pd.DataFrame({
            "Geração": range(1, len(self.hist_best) + 1),
            "Melhor custo": [round(v, 6) for v in self.hist_best],
            "Custo médio": [round(v, 6) for v in self.hist_mean],
            "Melhoria acumulada best (%)": melhoria_acum_best,
            "Melhoria geração best (%)": melhoria_geracao_best,
            "Delta custo best": delta_best,
            "Melhoria acumulada mean (%)": melhoria_acum_mean,
            "Melhoria geração mean (%)": melhoria_geracao_mean,
            "Delta custo mean": delta_mean,
        })

    def to_plot(self, normalizar=True):
        """Retorna gráfico de evolução do fitness com curvas de best, mean e melhoria acumulada da população."""
        fator_best = max(self.hist_best) if normalizar and max(self.hist_best) > 0 else 1.0
        fator_mean = max(self.hist_mean) if normalizar and max(self.hist_mean) > 0 else 1.0

        # Normalização
        y_best = [v / fator_best for v in self.hist_best]
        y_mean = [v / fator_mean for v in self.hist_mean]

        # Melhoria acumulada da população (mean)
        melhoria_acum_mean = [
            (self.hist_mean[0] - v) / self.hist_mean[0] * 100 if self.hist_mean[0] > 0 else 0
            for v in self.hist_mean
        ]

        # Construção do gráfico
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(self.hist_best))), y=y_best,
                                name="Best (normalizado)", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=list(range(len(self.hist_mean))), y=y_mean,
                                name="Mean (normalizado)", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=list(range(len(self.hist_mean))), y=melhoria_acum_mean,
                                name="Melhoria acumulada mean (%)", line=dict(color="purple", dash="dot")))

        fig.update_layout(
            title="Evolução do GA — Best, Mean e Melhoria da População",
            xaxis_title="Geração",
            yaxis_title="Fitness / Melhoria (%)",
            legend=dict(x=0.01, y=0.99, bordercolor="black", borderwidth=1)
        )
        return fig


    def to_summary(self):
        """Resumo textual da execução com evolução detalhada (best e mean)."""
        conv_gen = len(self.hist_best)

        # Redução acumulada best
        reducao_best = round((self.hist_best[0] - self.hist_best[-1]) / self.hist_best[0] * 100, 2) if self.hist_best[0] > 0 else 0
        delta_best = round(self.hist_best[0] - self.hist_best[-1], 2)

        # Redução acumulada mean
        reducao_mean = round((self.hist_mean[0] - self.hist_mean[-1]) / self.hist_mean[0] * 100, 2) if self.hist_mean[0] > 0 else 0
        delta_mean = round(self.hist_mean[0] - self.hist_mean[-1], 2)

        # Contagem de melhorias best
        melhorias_best = sum(
            1 for i in range(1, len(self.hist_best))
            if self.hist_best[i] < self.hist_best[i-1]
        )
        sem_melhoria_best = conv_gen - melhorias_best

        # Contagem de melhorias mean
        melhorias_mean = sum(
            1 for i in range(1, len(self.hist_mean))
            if self.hist_mean[i] < self.hist_mean[i-1]
        )
        sem_melhoria_mean = conv_gen - melhorias_mean

        return (
            f"O GA convergiu em {conv_gen} gerações.\n"
            f"➡️ Melhor indivíduo (best): redução acumulada de {reducao_best}% "
            f"(Δ {delta_best:.2f}, de {self.hist_best[0]:.2f} para {self.hist_best[-1]:.2f}). "
            f"Foram {melhorias_best} gerações com melhoria e {sem_melhoria_best} sem progresso.\n"
            f"➡️ População média (mean): redução acumulada de {reducao_mean}% "
            f"(Δ {delta_mean:.2f}, de {self.hist_mean[0]:.2f} para {self.hist_mean[-1]:.2f}). "
            f"Foram {melhorias_mean} gerações com melhoria e {sem_melhoria_mean} sem progresso."
        )
    
    def calcular_balanceamento(self):
        """Calcula métricas de balanceamento entre veículos."""
        n_pontos = [v["n_pontos"] for v in self.resumo_vrp]
        distancias = [v["distancia_pixels"] for v in self.resumo_vrp]
        cargas = [v["peso_total"] for v in self.resumo_vrp]

        def coeficiente_variacao(valores):
            media = sum(valores) / len(valores)
            variacao = (sum((x - media) ** 2 for x in valores) / len(valores)) ** 0.5
            return variacao / media if media > 0 else 0

        return {
            "balanceamento_pontos": round((1 - coeficiente_variacao(n_pontos)) * 100, 1),
            "balanceamento_distancia": round((1 - coeficiente_variacao(distancias)) * 100, 1),
            "balanceamento_carga": round((1 - coeficiente_variacao(cargas)) * 100, 1),
        }
