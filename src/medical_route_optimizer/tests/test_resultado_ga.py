import pytest
import pandas as pd
from models.resultado_ga import ResultadoGA


def make_dummy_resultado():
    return ResultadoGA(
        hist_best=[100, 80, 60],
        hist_mean=[90, 70, 65],
        melhor_rota=["A", "B", "C"],
        custo_final=60,
        custo_nn=120,
        hospital_base="Hospital X",
        locais=["A", "B", "C"],
        rotas_vrp=[["A", "B"], ["C"]],
        resumo_vrp=[{
            "veiculo": 1,
            "capacidade_ok": True,
            "autonomia_ok": True,
            "n_pontos": 2,
            "peso_total": 10,
            "capacidade_veiculo": 20,
            "distancia_pixels": 100,
            "autonomia_veiculo": 200
        }],
        relatorio={"comparacao_baseline_nn": {"economia_percentual": 50}}
    )


def test_dataframe_generation():
    res = make_dummy_resultado()
    df = res.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "Melhor custo" in df.columns
    # Verifica se houve alguma melhoria
    assert df.iloc[-1]["Melhoria (%)"] > 0


def test_plot_generation():
    res = make_dummy_resultado()
    fig = res.to_plot(normalizar=True)
    assert fig.data[0].name == "best"
    assert fig.data[1].name == "mean"


def test_summary_text():
    res = make_dummy_resultado()
    summary = res.to_summary()
    assert "convergiu" in summary
    assert "Redução de custo" in summary
