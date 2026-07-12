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
        resumo_vrp=[
            {
                "veiculo": 1,
                "capacidade_ok": True,
                "autonomia_ok": True,
                "n_pontos": 2,
                "peso_total": 10,
                "capacidade_veiculo": 20,
                "distancia_pixels": 100,
                "autonomia_veiculo": 200,
            },
            {
                "veiculo": 2,
                "capacidade_ok": True,
                "autonomia_ok": True,
                "n_pontos": 1,
                "peso_total": 5,
                "capacidade_veiculo": 20,
                "distancia_pixels": 80,
                "autonomia_veiculo": 200,
            },
        ],
        relatorio={"comparacao_baseline_nn": {"economia_percentual": 50}},
    )


# ── to_dataframe ──────────────────────────────────────────────────────────────

def test_dataframe_generation():
    res = make_dummy_resultado()
    df = res.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "Melhor custo" in df.columns
    # A coluna de melhoria acumulada usa o nome correto
    assert "Melhoria acumulada best (%)" in df.columns
    # Deve haver melhoria acumulada positiva na última geração
    assert df.iloc[-1]["Melhoria acumulada best (%)"] > 0


def test_dataframe_colunas_completas():
    res = make_dummy_resultado()
    df = res.to_dataframe()
    expected_cols = [
        "Geração",
        "Melhor custo",
        "Custo médio",
        "Melhoria acumulada best (%)",
        "Melhoria geração best (%)",
        "Delta custo best",
        "Melhoria acumulada mean (%)",
        "Melhoria geração mean (%)",
        "Delta custo mean",
    ]
    for col in expected_cols:
        assert col in df.columns


def test_dataframe_vazio_quando_hist_vazio():
    res = ResultadoGA(
        hist_best=[],
        hist_mean=[],
        melhor_rota=[],
        custo_final=0,
        custo_nn=0,
        hospital_base="H",
        locais=[],
        rotas_vrp=[],
        resumo_vrp=[],
        relatorio={},
    )
    df = res.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_dataframe_linhas_correspondem_ao_historico():
    res = make_dummy_resultado()
    df = res.to_dataframe()
    assert len(df) == len(res.hist_best)


# ── to_plot ───────────────────────────────────────────────────────────────────

def test_plot_generation():
    res = make_dummy_resultado()
    fig = res.to_plot(normalizar=True)
    nomes = [t.name for t in fig.data]
    assert "Best (normalizado)" in nomes
    assert "Mean (normalizado)" in nomes


def test_plot_sem_normalizacao():
    res = make_dummy_resultado()
    fig = res.to_plot(normalizar=False)
    assert fig is not None
    assert len(fig.data) >= 2


def test_plot_tem_melhoria_acumulada():
    res = make_dummy_resultado()
    fig = res.to_plot(normalizar=True)
    nomes = [t.name for t in fig.data]
    assert any("Melhoria" in n for n in nomes)


# ── to_summary ────────────────────────────────────────────────────────────────

def test_summary_text():
    res = make_dummy_resultado()
    summary = res.to_summary()
    assert "convergiu" in summary
    assert "redução acumulada" in summary


def test_summary_contem_gerações():
    res = make_dummy_resultado()
    summary = res.to_summary()
    assert "3 gerações" in summary


def test_summary_contem_valores_best():
    res = make_dummy_resultado()
    summary = res.to_summary()
    assert "100.00" in summary or "60.00" in summary


# ── calcular_balanceamento ────────────────────────────────────────────────────

def test_calcular_balanceamento_retorna_dict():
    res = make_dummy_resultado()
    bal = res.calcular_balanceamento()
    assert isinstance(bal, dict)


def test_calcular_balanceamento_campos():
    res = make_dummy_resultado()
    bal = res.calcular_balanceamento()
    assert "balanceamento_pontos" in bal
    assert "balanceamento_distancia" in bal
    assert "balanceamento_carga" in bal


def test_calcular_balanceamento_valores_percentuais():
    res = make_dummy_resultado()
    bal = res.calcular_balanceamento()
    # Valores são percentuais (podem ser negativos se distribuição muito desigual)
    for key in ("balanceamento_pontos", "balanceamento_distancia", "balanceamento_carga"):
        assert isinstance(bal[key], float)
