"""Testes unitários para core/nearest_neighbor.py."""

import pytest
from core.nearest_neighbor import (
    nearest_neighbor,
    gerar_populacao_nearest_neighbor,
    avaliar_baseline_nn,
)
from data.delivery_points import PontoEntrega


def make_ponto(nome, x, y, prioridade=3, peso=1.0):
    return PontoEntrega(nome=nome, coords=(x, y), prioridade=prioridade,
                        tempo_atendimento=5, peso=peso)


@pytest.fixture
def hospital():
    return PontoEntrega("H", (0, 0), 0, 0, 0.0, is_origin=True)


@pytest.fixture
def pontos_simples():
    # Pontos em linha horizontal para resultado previsível
    return [
        make_ponto("A", 10, 0),
        make_ponto("B", 30, 0),
        make_ponto("C", 20, 0),
    ]


class TestNearestNeighbor:
    def test_visita_todos_os_pontos(self, hospital, pontos_simples):
        rota = nearest_neighbor(pontos_simples, hospital)
        assert set(rota) == set(pontos_simples)

    def test_primeiro_ponto_mais_proximo_do_hospital(self, hospital, pontos_simples):
        rota = nearest_neighbor(pontos_simples, hospital)
        # Hospital em (0,0): A(10,0) é o mais próximo
        assert rota[0].nome == "A"

    def test_ordem_greedy_correta(self, hospital):
        p1 = make_ponto("A", 1, 0)
        p2 = make_ponto("B", 3, 0)
        p3 = make_ponto("C", 2, 0)
        rota = nearest_neighbor([p1, p2, p3], hospital)
        # H(0,0)→A(1,0)→C(2,0)→B(3,0)
        assert [p.nome for p in rota] == ["A", "C", "B"]

    def test_rota_tamanho_correto(self, hospital, pontos_simples):
        rota = nearest_neighbor(pontos_simples, hospital)
        assert len(rota) == len(pontos_simples)

    def test_rota_ponto_unico(self, hospital):
        p = make_ponto("Único", 5, 5)
        rota = nearest_neighbor([p], hospital)
        assert rota == [p]

    def test_rota_vazia(self, hospital):
        rota = nearest_neighbor([], hospital)
        assert rota == []


class TestGerarPopulacaoNearestNeighbor:
    def test_uma_solucao(self, hospital, pontos_simples):
        pop = gerar_populacao_nearest_neighbor(pontos_simples, hospital, n_solucoes=1)
        assert len(pop) == 1
        assert set(pop[0]) == set(pontos_simples)

    def test_multiplas_solucoes(self, hospital, pontos_simples):
        pop = gerar_populacao_nearest_neighbor(pontos_simples, hospital, n_solucoes=3)
        assert len(pop) == 3
        for rota in pop:
            assert set(rota) == set(pontos_simples)

    def test_primeira_solucao_padrao_nn(self, hospital):
        p1 = make_ponto("A", 1, 0)
        p2 = make_ponto("B", 10, 0)
        pop = gerar_populacao_nearest_neighbor([p1, p2], hospital, n_solucoes=1)
        # Primeira solução deve ser NN padrão: A(1,0) mais próximo de H(0,0)
        assert pop[0][0].nome == "A"

    def test_solucoes_sao_permutacoes_validas(self, hospital, pontos_simples):
        pop = gerar_populacao_nearest_neighbor(pontos_simples, hospital, n_solucoes=5)
        for rota in pop:
            assert len(rota) == len(pontos_simples)
            assert set(rota) == set(pontos_simples)


class TestAvaliarBaselineNn:
    def test_retorna_rota_e_custo(self, hospital, pontos_simples):
        rota, custo = avaliar_baseline_nn(pontos_simples, hospital)
        assert isinstance(rota, list)
        assert isinstance(custo, float)
        assert custo > 0

    def test_rota_valida(self, hospital, pontos_simples):
        rota, _ = avaliar_baseline_nn(pontos_simples, hospital)
        assert set(rota) == set(pontos_simples)

    def test_com_parametros_de_custo(self, hospital, pontos_simples):
        _, custo_base = avaliar_baseline_nn(pontos_simples, hospital)
        _, custo_penalidade = avaliar_baseline_nn(
            pontos_simples, hospital,
            fator_penalidade=100.0
        )
        # Com penalidade maior, custo pode ser diferente se há pontos de alta prioridade
        # Neste caso todos têm prioridade=3, então não há diferença
        assert custo_base == pytest.approx(custo_penalidade)

    def test_com_capacidade_restricao(self, hospital):
        pontos = [make_ponto("A", 1, 0, peso=10.0)]
        _, custo_sem = avaliar_baseline_nn(pontos, hospital)
        _, custo_com = avaliar_baseline_nn(
            pontos, hospital,
            fator_penalidade=0.0,
            capacidade_veiculo=5.0,
            fator_penalidade_capacidade=100.0
        )
        assert custo_com >= custo_sem

    def test_com_autonomia_restricao(self, hospital):
        pontos = [make_ponto("A", 100, 0, peso=1.0)]
        _, custo_sem = avaliar_baseline_nn(pontos, hospital)
        _, custo_com = avaliar_baseline_nn(
            pontos, hospital,
            fator_penalidade=0.0,
            autonomia_veiculo=50.0,
            fator_penalidade_autonomia=10.0
        )
        assert custo_com >= custo_sem
