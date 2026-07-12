"""Testes unitários para core/genetic_algorithm.py."""

import random
import pytest
from unittest.mock import MagicMock, patch
from core.genetic_algorithm import executar_algoritmo_genetico
from data.delivery_points import PontoEntrega


def make_ponto(nome, x, y, prioridade=3, peso=1.0):
    return PontoEntrega(nome=nome, coords=(x, y), prioridade=prioridade,
                        tempo_atendimento=5, peso=peso)


@pytest.fixture
def hospital():
    return PontoEntrega("H", (0, 0), 0, 0, 0.0, is_origin=True)


@pytest.fixture
def pontos_simples():
    return [
        make_ponto("A", 10, 0, prioridade=3),
        make_ponto("B", 20, 0, prioridade=3),
        make_ponto("C", 30, 0, prioridade=3),
    ]


@pytest.fixture
def pop_inicial(pontos_simples):
    return [list(pontos_simples), list(reversed(pontos_simples))]


class TestExecutarAlgoritmoGenetico:
    def test_retorna_quatro_valores(self, pontos_simples, hospital, pop_inicial):
        result = executar_algoritmo_genetico(
            locais_entrega=pontos_simples,
            hospital_base=hospital,
            populacao_inicial=pop_inicial,
            paciencia=1,
            verbose=False,
            limite_tempo=None,
        )
        assert len(result) == 4

    def test_melhor_rota_e_permutacao_valida(self, pontos_simples, hospital, pop_inicial):
        melhor_rota, _, _, _ = executar_algoritmo_genetico(
            locais_entrega=pontos_simples,
            hospital_base=hospital,
            populacao_inicial=pop_inicial,
            paciencia=1,
            verbose=False,
            limite_tempo=None,
        )
        assert set(melhor_rota) == set(pontos_simples)

    def test_custo_e_float(self, pontos_simples, hospital, pop_inicial):
        _, custo, _, _ = executar_algoritmo_genetico(
            locais_entrega=pontos_simples,
            hospital_base=hospital,
            populacao_inicial=pop_inicial,
            paciencia=1,
            verbose=False,
            limite_tempo=None,
        )
        assert isinstance(custo, float)

    def test_historico_nao_vazio(self, pontos_simples, hospital, pop_inicial):
        _, _, hist_best, hist_mean = executar_algoritmo_genetico(
            locais_entrega=pontos_simples,
            hospital_base=hospital,
            populacao_inicial=pop_inicial,
            paciencia=1,
            verbose=False,
            limite_tempo=None,
        )
        assert len(hist_best) > 0
        assert len(hist_mean) > 0

    def test_modo_vrp(self, hospital):
        pontos = [
            make_ponto("A", 5, 0, peso=5.0),
            make_ponto("B", 10, 0, peso=5.0),
        ]
        pop = [list(pontos), list(reversed(pontos))]
        melhor_rota, custo, _, _ = executar_algoritmo_genetico(
            locais_entrega=pontos,
            hospital_base=hospital,
            populacao_inicial=pop,
            paciencia=1,
            verbose=False,
            limite_tempo=None,
            n_veiculos=2,
            capacidade_veiculo=4.0,
        )
        assert set(melhor_rota) == set(pontos)
        assert isinstance(custo, float)

    def test_para_por_limite_tempo(self, pontos_simples, hospital):
        pop = [list(pontos_simples)] * 4
        melhor_rota, _, _, _ = executar_algoritmo_genetico(
            locais_entrega=pontos_simples,
            hospital_base=hospital,
            populacao_inicial=pop,
            paciencia=10000,
            verbose=False,
            limite_tempo=0,  # 0 segundos → para imediatamente
        )
        assert set(melhor_rota) == set(pontos_simples)

    def test_para_por_convergencia_std(self, hospital):
        # Pontos idênticos em coords → todos os custos iguais → std=0
        p1 = PontoEntrega("A", (5, 5), 3, 5, peso=1.0)
        p2 = PontoEntrega("B", (5, 5), 3, 5, peso=1.0)
        p3 = PontoEntrega("C", (5, 5), 3, 5, peso=1.0)
        pontos = [p1, p2, p3]
        pop = [list(pontos)] * 4
        melhor_rota, custo, _, _ = executar_algoritmo_genetico(
            locais_entrega=pontos,
            hospital_base=hospital,
            populacao_inicial=pop,
            paciencia=1000,
            verbose=False,
            limite_tempo=None,
        )
        assert isinstance(custo, float)

    def test_verbose_true_nao_lanca_excecao(self, pontos_simples, hospital, pop_inicial):
        executar_algoritmo_genetico(
            locais_entrega=pontos_simples,
            hospital_base=hospital,
            populacao_inicial=pop_inicial,
            paciencia=1,
            verbose=True,
            limite_tempo=None,
        )

    def test_com_animacao_mock(self, pontos_simples, hospital, pop_inicial):
        animacao_mock = MagicMock()
        executar_algoritmo_genetico(
            locais_entrega=pontos_simples,
            hospital_base=hospital,
            populacao_inicial=pop_inicial,
            paciencia=1,
            verbose=False,
            limite_tempo=None,
            animacao=animacao_mock,
        )
        animacao_mock.registrar.assert_called()
        animacao_mock.finalizar.assert_called_once()

    def test_mutacao_adaptativa_ativa(self, hospital):
        """Testa que a mutação adaptativa é ativada após 20+ gerações sem melhoria."""
        pontos = [
            make_ponto("A", 10, 0, prioridade=1),
            make_ponto("B", 20, 0, prioridade=2),
            make_ponto("C", 30, 0, prioridade=3),
        ]
        pop = [list(pontos)] * 6
        # tolerancia alta → nunca melhora → adaptativa ativa após 21 gens
        # paciencia=25 → para após 25 gerações sem melhoria
        melhor_rota, _, hist, _ = executar_algoritmo_genetico(
            locais_entrega=pontos,
            hospital_base=hospital,
            populacao_inicial=pop,
            paciencia=25,
            tamanho_elite=1,
            tolerancia=1e10,
            verbose=True,
            limite_tempo=None,
        )
        assert len(hist) >= 20

    def test_modo_vrp_autonomia(self, hospital):
        pontos = [
            make_ponto("A", 100, 0, peso=1.0),
            make_ponto("B", 200, 0, peso=1.0),
        ]
        pop = [list(pontos), list(reversed(pontos))]
        melhor_rota, _, _, _ = executar_algoritmo_genetico(
            locais_entrega=pontos,
            hospital_base=hospital,
            populacao_inicial=pop,
            paciencia=1,
            verbose=False,
            limite_tempo=None,
            n_veiculos=2,
            autonomia_veiculo=50.0,
        )
        assert set(melhor_rota) == set(pontos)
