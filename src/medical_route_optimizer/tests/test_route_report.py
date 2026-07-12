"""Testes unitários para reports/route_report.py."""

import pytest
from reports.route_report import gerar_relatorio_rota, _gerar_secao_vrp, _tempo_deslocamento
from data.delivery_points import PontoEntrega


def make_ponto(nome, x, y, prioridade=3, peso=1.0, tempo=10):
    return PontoEntrega(nome=nome, coords=(x, y), prioridade=prioridade,
                        tempo_atendimento=tempo, peso=peso)


@pytest.fixture
def hospital():
    return PontoEntrega("Hospital Base", (0, 0), 0, 0, 0.0, is_origin=True)


@pytest.fixture
def rota_simples():
    return [
        make_ponto("A", 5, 0, prioridade=1),
        make_ponto("B", 10, 0, prioridade=2),
        make_ponto("C", 15, 0, prioridade=3),
    ]


@pytest.fixture
def historico_custos():
    return [100.0, 90.0, 80.0, 70.0]


class TestTempoDeslocamento:
    def test_conversao_basica(self):
        from reports.route_report import VELOCIDADE_MEDIA_PIXELS_POR_MINUTO
        resultado = _tempo_deslocamento(VELOCIDADE_MEDIA_PIXELS_POR_MINUTO)
        assert resultado == pytest.approx(1.0)

    def test_distancia_zero(self):
        assert _tempo_deslocamento(0) == pytest.approx(0.0)


class TestGerarRelatorioRota:
    def test_retorna_dicionario(self, hospital, rota_simples, historico_custos):
        rel = gerar_relatorio_rota(rota_simples, hospital, 70.0, historico_custos)
        assert isinstance(rel, dict)

    def test_campo_resumo_presente(self, hospital, rota_simples, historico_custos):
        rel = gerar_relatorio_rota(rota_simples, hospital, 70.0, historico_custos)
        assert "resumo" in rel

    def test_total_pontos_entrega(self, hospital, rota_simples, historico_custos):
        rel = gerar_relatorio_rota(rota_simples, hospital, 70.0, historico_custos)
        assert rel["resumo"]["total_pontos_entrega"] == len(rota_simples)

    def test_sequencia_atendimentos(self, hospital, rota_simples, historico_custos):
        rel = gerar_relatorio_rota(rota_simples, hospital, 70.0, historico_custos)
        assert "sequencia_atendimentos" in rel
        assert len(rel["sequencia_atendimentos"]) == len(rota_simples)

    def test_prioridades_preenchidas(self, hospital, rota_simples, historico_custos):
        rel = gerar_relatorio_rota(rota_simples, hospital, 70.0, historico_custos)
        assert "prioridades" in rel
        assert "alta" in rel["prioridades"]
        assert "media" in rel["prioridades"]
        assert "baixa" in rel["prioridades"]

    def test_comparacao_baseline_com_nn(self, hospital, rota_simples, historico_custos):
        rel = gerar_relatorio_rota(
            rota_simples, hospital, 70.0, historico_custos,
            rota_baseline_nn=rota_simples, custo_baseline_nn=100.0
        )
        comp = rel["comparacao_baseline_nn"]
        assert comp["custo_baseline_nn"] == pytest.approx(100.0)
        assert comp["economia_percentual"] is not None
        assert comp["ga_superou_nn"] is True

    def test_sem_baseline_nn(self, hospital, rota_simples, historico_custos):
        rel = gerar_relatorio_rota(rota_simples, hospital, 70.0, historico_custos)
        comp = rel["comparacao_baseline_nn"]
        assert comp["economia_percentual"] is None
        assert comp["ga_superou_nn"] is None

    def test_historico_evolucao(self, hospital, rota_simples, historico_custos):
        rel = gerar_relatorio_rota(rota_simples, hospital, 70.0, historico_custos)
        hist = rel["historico_evolucao"]
        assert hist["n_geracoes_total"] == len(historico_custos)
        assert hist["n_geracoes_com_melhoria"] == 3

    def test_secao_vrp_presente(self, hospital, rota_simples, historico_custos):
        rotas_vrp = [rota_simples[:2], rota_simples[2:]]
        resumo_vrp = [
            {"veiculo": 1, "n_pontos": 2, "peso_total": 2.0,
             "capacidade_veiculo": 10.0, "capacidade_ok": True,
             "distancia_pixels": 20.0, "autonomia_veiculo": 100.0,
             "autonomia_ok": True, "pontos": ["A", "B"]},
            {"veiculo": 2, "n_pontos": 1, "peso_total": 1.0,
             "capacidade_veiculo": 10.0, "capacidade_ok": True,
             "distancia_pixels": 30.0, "autonomia_veiculo": 100.0,
             "autonomia_ok": True, "pontos": ["C"]},
        ]
        rel = gerar_relatorio_rota(
            rota_simples, hospital, 70.0, historico_custos,
            rotas_vrp=rotas_vrp, resumo_vrp=resumo_vrp
        )
        assert rel["vrp"] is not None
        assert rel["vrp"]["n_veiculos"] == 2

    def test_vrp_none_quando_sem_vrp(self, hospital, rota_simples, historico_custos):
        rel = gerar_relatorio_rota(rota_simples, hospital, 70.0, historico_custos)
        assert rel["vrp"] is None

    def test_campos_sequencia_atendimento(self, hospital, rota_simples, historico_custos):
        rel = gerar_relatorio_rota(rota_simples, hospital, 70.0, historico_custos)
        prim = rel["sequencia_atendimentos"][0]
        assert "posicao" in prim
        assert "nome" in prim
        assert "prioridade" in prim
        assert "tempo_deslocamento_min" in prim
        assert "tempo_acumulado_min" in prim

    def test_historico_vazio(self, hospital, rota_simples):
        rel = gerar_relatorio_rota(rota_simples, hospital, 70.0, [])
        assert rel["resumo"]["custo_inicial_ga"] is None
        assert rel["resumo"]["custo_final_ga"] is None

    def test_economia_percentual_negativa(self, hospital, rota_simples, historico_custos):
        # GA custou mais que NN
        rel = gerar_relatorio_rota(
            rota_simples, hospital, 200.0, historico_custos,
            custo_baseline_nn=100.0
        )
        assert rel["comparacao_baseline_nn"]["ga_superou_nn"] is False


class TestGerarSecaoVrp:
    def test_retorna_none_sem_dados(self):
        assert _gerar_secao_vrp(None, None) is None

    def test_retorna_none_sem_resumo(self):
        assert _gerar_secao_vrp([[]], None) is None

    def test_retorna_none_sem_rotas(self):
        assert _gerar_secao_vrp(None, [{}]) is None

    def test_solucao_valida(self, hospital):
        p = make_ponto("A", 5, 0, peso=1.0)
        rotas_vrp = [[p]]
        resumo_vrp = [{
            "veiculo": 1, "n_pontos": 1, "peso_total": 1.0,
            "capacidade_veiculo": 10.0, "capacidade_ok": True,
            "distancia_pixels": 10.0, "autonomia_veiculo": 100.0,
            "autonomia_ok": True, "pontos": ["A"],
        }]
        secao = _gerar_secao_vrp(rotas_vrp, resumo_vrp)
        assert secao["solucao_valida"] is True
        assert secao["n_veiculos"] == 1

    def test_solucao_invalida(self, hospital):
        p = make_ponto("A", 5, 0, peso=15.0)
        rotas_vrp = [[p]]
        resumo_vrp = [{
            "veiculo": 1, "n_pontos": 1, "peso_total": 15.0,
            "capacidade_veiculo": 10.0, "capacidade_ok": False,
            "distancia_pixels": 10.0, "autonomia_veiculo": 100.0,
            "autonomia_ok": True, "pontos": ["A"],
        }]
        secao = _gerar_secao_vrp(rotas_vrp, resumo_vrp)
        assert secao["solucao_valida"] is False
