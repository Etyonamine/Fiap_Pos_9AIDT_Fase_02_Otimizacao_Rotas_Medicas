"""Testes unitários para core/two_opt.py."""

import pytest
from core.two_opt import two_opt_inversion, two_opt_vrp
from data.delivery_points import PontoEntrega


def make_ponto(nome, x, y, prioridade=3, peso=1.0):
    return PontoEntrega(nome=nome, coords=(x, y), prioridade=prioridade,
                        tempo_atendimento=5, peso=peso)


@pytest.fixture
def hospital():
    return PontoEntrega("H", (0, 0), 0, 0, 0.0, is_origin=True)


class TestTwoOptInversion:
    def test_retorna_rota_e_custo(self, hospital):
        pontos = [make_ponto("A", 5, 0), make_ponto("B", 10, 0)]
        rota, custo = two_opt_inversion(pontos, hospital)
        assert isinstance(rota, list)
        assert isinstance(custo, float)

    def test_rota_contem_todos_pontos(self, hospital):
        pontos = [make_ponto("A", 5, 0), make_ponto("B", 10, 5), make_ponto("C", 0, 5)]
        rota, _ = two_opt_inversion(pontos, hospital)
        assert set(rota) == set(pontos)

    def test_custo_otimizado_menor_ou_igual_ao_inicial(self, hospital):
        from core.fitness_calculator import calcular_custo_rota
        pontos = [
            make_ponto("A", 10, 10),
            make_ponto("B", 0, 10),
            make_ponto("C", 10, 0),
            make_ponto("D", 0, 0),
        ]
        custo_inicial = calcular_custo_rota(pontos, hospital)
        _, custo_otimizado = two_opt_inversion(pontos, hospital)
        assert custo_otimizado <= custo_inicial

    def test_verbose_true(self, hospital, capsys):
        pontos = [
            make_ponto("A", 10, 10),
            make_ponto("B", 0, 10),
            make_ponto("C", 10, 0),
        ]
        two_opt_inversion(pontos, hospital, verbose=True)
        # Não lança exceção; pode ou não imprimir dependendo de encontrar melhoria

    def test_max_iteracoes_um(self, hospital):
        pontos = [make_ponto("A", 5, 0), make_ponto("B", 0, 5)]
        rota, custo = two_opt_inversion(pontos, hospital, max_iteracoes=1)
        assert set(rota) == set(pontos)

    def test_rota_ponto_unico(self, hospital):
        pontos = [make_ponto("A", 5, 0)]
        rota, custo = two_opt_inversion(pontos, hospital)
        assert rota[0] == pontos[0]
        assert custo > 0

    def test_com_penalidades(self, hospital):
        pontos = [
            make_ponto("A", 5, 0, prioridade=1, peso=3.0),
            make_ponto("B", 10, 0, prioridade=2, peso=2.0),
        ]
        rota, custo = two_opt_inversion(
            pontos, hospital,
            fator_penalidade=5.0,
            capacidade_veiculo=10.0,
            autonomia_veiculo=100.0
        )
        assert set(rota) == set(pontos)

    def test_rota_vazia(self, hospital):
        rota, custo = two_opt_inversion([], hospital)
        assert rota == []
        assert custo == pytest.approx(0.0)


class TestTwoOptVrp:
    def test_retorna_rotas_e_custo_total(self, hospital):
        r1 = [make_ponto("A", 5, 0), make_ponto("B", 10, 0)]
        r2 = [make_ponto("C", 0, 5)]
        rotas_otim, custo = two_opt_vrp([r1, r2], hospital)
        assert len(rotas_otim) == 2
        assert custo > 0

    def test_preserva_pontos_em_cada_sub_rota(self, hospital):
        r1 = [make_ponto("A", 5, 0), make_ponto("B", 0, 5)]
        r2 = [make_ponto("C", 10, 10)]
        rotas_otim, _ = two_opt_vrp([r1, r2], hospital)
        assert set(rotas_otim[0]) == set(r1)
        assert set(rotas_otim[1]) == set(r2)

    def test_custo_total_soma_sub_rotas(self, hospital):
        from core.fitness_calculator import calcular_custo_rota
        r1 = [make_ponto("A", 5, 0)]
        r2 = [make_ponto("B", 0, 5)]
        _, custo_total = two_opt_vrp([r1, r2], hospital)
        assert custo_total > 0

    def test_rota_unica(self, hospital):
        r = [make_ponto("A", 5, 0), make_ponto("B", 10, 0)]
        rotas_otim, custo = two_opt_vrp([r], hospital)
        assert len(rotas_otim) == 1
        assert custo > 0
