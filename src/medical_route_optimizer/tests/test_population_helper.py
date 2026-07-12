"""Testes unitários para core/population_helper.py."""

import pytest
from core.population_helper import gerar_populacao_aleatoria, ordenar_populacao
from data.delivery_points import PontoEntrega


def make_ponto(nome, x, y):
    return PontoEntrega(nome=nome, coords=(x, y), prioridade=3, tempo_atendimento=5)


@pytest.fixture
def locais():
    return [make_ponto(str(i), i * 10, 0) for i in range(5)]


class TestGerarPopulacaoAleatoria:
    def test_tamanho_correto(self, locais):
        pop = gerar_populacao_aleatoria(locais, tamanho_populacao=10)
        assert len(pop) == 10

    def test_cada_individuo_e_permutacao(self, locais):
        pop = gerar_populacao_aleatoria(locais, tamanho_populacao=5)
        for rota in pop:
            assert set(rota) == set(locais)
            assert len(rota) == len(locais)

    def test_tamanho_um(self, locais):
        pop = gerar_populacao_aleatoria(locais, tamanho_populacao=1)
        assert len(pop) == 1

    def test_nenhum_hospital_base_nas_rotas(self, locais):
        pop = gerar_populacao_aleatoria(locais, tamanho_populacao=3)
        for rota in pop:
            assert all(not p.is_origin for p in rota)


class TestOrdenarPopulacao:
    def test_ordena_pelo_custo_crescente(self):
        p1 = make_ponto("A", 0, 0)
        p2 = make_ponto("B", 1, 0)
        p3 = make_ponto("C", 2, 0)
        pop = [[p3], [p1], [p2]]
        custos = [30.0, 10.0, 20.0]
        pop_ord, custos_ord = ordenar_populacao(pop, custos)
        assert custos_ord == [10.0, 20.0, 30.0]

    def test_populacao_correspondente_ao_custo(self):
        p1 = make_ponto("A", 0, 0)
        p2 = make_ponto("B", 1, 0)
        pop = [[p1], [p2]]
        custos = [50.0, 10.0]
        pop_ord, custos_ord = ordenar_populacao(pop, custos)
        assert pop_ord[0] == [p2]
        assert pop_ord[1] == [p1]

    def test_retorna_listas(self):
        p = make_ponto("A", 0, 0)
        pop, custos = ordenar_populacao([[p]], [5.0])
        assert isinstance(pop, list)
        assert isinstance(custos, list)

    def test_preserva_todos_elementos(self):
        pontos = [make_ponto(str(i), i, 0) for i in range(4)]
        pop = [[p] for p in pontos]
        custos = [40.0, 10.0, 30.0, 20.0]
        pop_ord, custos_ord = ordenar_populacao(pop, custos)
        assert len(pop_ord) == 4
        assert sorted(custos_ord) == custos_ord
