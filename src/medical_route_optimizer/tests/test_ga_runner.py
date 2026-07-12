import pytest
from services.ga_runner import run_ga
from models.resultado_ga import ResultadoGA


def test_run_ga_returns_resultado(monkeypatch):
    # Monkeypatch funções pesadas para não rodar GA real
    monkeypatch.setattr("services.ga_runner.executar_algoritmo_genetico",
                        lambda **kwargs: (["rota"], 50, [100, 80], [90, 70]))
    monkeypatch.setattr("services.ga_runner.two_opt_vrp",
                        lambda rotas_vrp, hospital, **kwargs: (rotas_vrp, 50))
    monkeypatch.setattr("services.ga_runner.avaliar_baseline_nn",
                        lambda locais, hospital, **kwargs: (["rota_nn"], 120))
    monkeypatch.setattr("services.ga_runner.dividir_rotas_vrp",
                        lambda *args, **kwargs: [["rota"]])
    monkeypatch.setattr("services.ga_runner.resumo_restricoes_vrp",
                        lambda *args, **kwargs: [{"veiculo": 1, "capacidade_ok": True, "autonomia_ok": True,
                                                  "n_pontos": 1, "peso_total": 10, "capacidade_veiculo": 20,
                                                  "distancia_pixels": 100, "autonomia_veiculo": 200}])
    monkeypatch.setattr("services.ga_runner.gerar_relatorio_rota",
                        lambda *args, **kwargs: {"comparacao_baseline_nn": {"economia_percentual": 50}})

    res = run_ga(10, 0.8, 0.1, [1, 10, 500, 3], 2, 16, 1400)
    assert isinstance(res, ResultadoGA)
    assert res.custo_final == 50
    assert res.custo_nn == 120
    assert res.relatorio["comparacao_baseline_nn"]["economia_percentual"] == 50
