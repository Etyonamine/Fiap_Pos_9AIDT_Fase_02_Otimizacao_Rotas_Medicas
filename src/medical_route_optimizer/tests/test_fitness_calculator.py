"""Testes unitários para core/fitness_calculator.py."""

import pytest
from core.fitness_calculator import (
    calcular_distancia_rota,
    calcular_custo_rota,
    calcular_custo_giant_tour_vrp,
)
from data.delivery_points import PontoEntrega


def make_ponto(nome, x, y, prioridade=3, peso=1.0):
    return PontoEntrega(nome=nome, coords=(x, y), prioridade=prioridade,
                        tempo_atendimento=5, peso=peso)


@pytest.fixture
def hospital():
    return PontoEntrega("H", (0, 0), 0, 0, 0.0, is_origin=True)


class TestCalcularDistanciaRota:
    def test_ciclo_fechado(self, hospital):
        p = make_ponto("A", 5, 0)
        # H→A: 5, A→H: 5  → 10
        assert calcular_distancia_rota([p], hospital) == pytest.approx(10.0)

    def test_rota_vazia(self, hospital):
        assert calcular_distancia_rota([], hospital) == pytest.approx(0.0)

    def test_rota_dois_pontos(self, hospital):
        p1 = make_ponto("A", 3, 0)
        p2 = make_ponto("B", 3, 4)
        # H→A=3, A→B=4, B→H=5  → 12
        assert calcular_distancia_rota([p1, p2], hospital) == pytest.approx(12.0)


class TestCalcularCustoRota:
    def test_sem_penalidades(self, hospital):
        p = make_ponto("A", 5, 0, prioridade=3)
        custo = calcular_custo_rota([p], hospital, fator_penalidade=5.0)
        # distância: 10, nenhuma penalidade (prio=3)
        assert custo == pytest.approx(10.0)

    def test_penalidade_prioridade_alta(self, hospital):
        p = make_ponto("A", 5, 0, prioridade=1)
        # posicao=1, prioridade=1: penalidade += (1/1) * 5.0 = 5
        # dist = 10, total = 15
        custo = calcular_custo_rota([p], hospital, fator_penalidade=5.0)
        assert custo == pytest.approx(15.0)

    def test_penalidade_prioridade_media(self, hospital):
        p = make_ponto("A", 5, 0, prioridade=2)
        # posicao=1, prioridade=2: penalidade += (1/2) * 5.0 = 2.5
        # dist = 10, total = 12.5
        custo = calcular_custo_rota([p], hospital, fator_penalidade=5.0)
        assert custo == pytest.approx(12.5)

    def test_sem_penalidade_prioridade_baixa(self, hospital):
        p = make_ponto("A", 5, 0, prioridade=3)
        custo = calcular_custo_rota([p], hospital, fator_penalidade=5.0)
        assert custo == pytest.approx(10.0)

    def test_penalidade_capacidade(self, hospital):
        # peso=5, capacidade=3: excesso=2, relativo=2/3
        p = make_ponto("A", 0, 0, peso=5.0)
        custo_sem = calcular_custo_rota([p], hospital, fator_penalidade=0.0)
        custo_com = calcular_custo_rota([p], hospital, fator_penalidade=0.0,
                                        capacidade_veiculo=3.0,
                                        fator_penalidade_capacidade=17.0)
        assert custo_com > custo_sem

    def test_sem_penalidade_capacidade_dentro_limite(self, hospital):
        p = make_ponto("A", 0, 0, peso=1.0)
        custo_sem = calcular_custo_rota([p], hospital, fator_penalidade=0.0)
        custo_com = calcular_custo_rota([p], hospital, fator_penalidade=0.0,
                                        capacidade_veiculo=10.0)
        assert custo_sem == pytest.approx(custo_com)

    def test_penalidade_autonomia(self, hospital):
        p = make_ponto("A", 100, 0)
        # distância = 200, autonomia = 50: excesso = 150
        custo_sem = calcular_custo_rota([p], hospital, fator_penalidade=0.0)
        custo_com = calcular_custo_rota([p], hospital, fator_penalidade=0.0,
                                        autonomia_veiculo=50.0,
                                        fator_penalidade_autonomia=2.0)
        assert custo_com > custo_sem

    def test_sem_penalidade_autonomia_dentro_limite(self, hospital):
        p = make_ponto("A", 1, 0)
        # distância = 2, autonomia = 100
        custo_sem = calcular_custo_rota([p], hospital, fator_penalidade=0.0)
        custo_com = calcular_custo_rota([p], hospital, fator_penalidade=0.0,
                                        autonomia_veiculo=100.0)
        assert custo_sem == pytest.approx(custo_com)

    def test_capacidade_none_sem_penalidade(self, hospital):
        p = make_ponto("A", 1, 0, peso=999.0)
        custo = calcular_custo_rota([p], hospital, fator_penalidade=0.0,
                                    capacidade_veiculo=None)
        assert custo == pytest.approx(2.0)

    def test_autonomia_none_sem_penalidade(self, hospital):
        p = make_ponto("A", 100, 0)
        custo = calcular_custo_rota([p], hospital, fator_penalidade=0.0,
                                    autonomia_veiculo=None)
        assert custo == pytest.approx(200.0)


class TestCalcularCustoGiantTourVrp:
    def test_um_veiculo_sem_restricoes(self, hospital):
        p1 = make_ponto("A", 3, 0, peso=1.0)
        p2 = make_ponto("B", 6, 0, peso=1.0)
        custo = calcular_custo_giant_tour_vrp(
            [p1, p2], hospital, n_veiculos=1,
            capacidade_veiculo=None, autonomia_veiculo=None
        )
        assert custo > 0

    def test_split_por_capacidade(self, hospital):
        p1 = make_ponto("A", 3, 0, peso=5.0)
        p2 = make_ponto("B", 6, 0, peso=5.0)
        # capacidade=4: p1 (peso=5) já excede, será forçado split mas com 2 veículos
        custo = calcular_custo_giant_tour_vrp(
            [p1, p2], hospital, n_veiculos=2,
            capacidade_veiculo=4.0, autonomia_veiculo=None
        )
        assert custo > 0

    def test_split_por_autonomia(self, hospital):
        p1 = make_ponto("A", 50, 0, peso=1.0)
        p2 = make_ponto("B", 100, 0, peso=1.0)
        # autonomia curta → split
        custo = calcular_custo_giant_tour_vrp(
            [p1, p2], hospital, n_veiculos=2,
            capacidade_veiculo=None, autonomia_veiculo=60.0
        )
        assert custo > 0

    def test_rota_vazia(self, hospital):
        custo = calcular_custo_giant_tour_vrp(
            [], hospital, n_veiculos=1,
            capacidade_veiculo=None, autonomia_veiculo=None
        )
        assert custo == pytest.approx(0.0)

    def test_sem_split_quando_unico_veiculo(self, hospital):
        p1 = make_ponto("A", 3, 0, peso=5.0)
        p2 = make_ponto("B", 6, 0, peso=5.0)
        # n_veiculos=1: nunca faz split (len(rotas) < n_veiculos - 1 = 0 é sempre False)
        custo = calcular_custo_giant_tour_vrp(
            [p1, p2], hospital, n_veiculos=1,
            capacidade_veiculo=1.0, autonomia_veiculo=None
        )
        assert custo > 0

    def test_veiculos_disponiveis_esgotados(self, hospital):
        # n_veiculos=2: len(rotas) < 1 → pode abrir 1 nova rota
        p1 = make_ponto("A", 3, 0, peso=5.0)
        p2 = make_ponto("B", 6, 0, peso=5.0)
        p3 = make_ponto("C", 0, 3, peso=5.0)
        # capacidade=4: todos excedem, mas só pode abrir 1 extra
        custo = calcular_custo_giant_tour_vrp(
            [p1, p2, p3], hospital, n_veiculos=2,
            capacidade_veiculo=4.0, autonomia_veiculo=None
        )
        assert custo > 0
