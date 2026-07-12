"""Testes unitários para core/genetic_operator.py."""

import random
import pytest
from core.genetic_operator import (
    pmx_crossover,
    order_crossover,
    mutate,
    mutate_segment_inversion,
)
from data.delivery_points import PontoEntrega


def make_pontos(n):
    return [
        PontoEntrega(nome=str(i), coords=(i * 10, 0), prioridade=3, tempo_atendimento=5)
        for i in range(n)
    ]


class TestPmxCrossover:
    def test_retorna_permutacao_valida(self):
        pontos = make_pontos(5)
        filho = pmx_crossover(pontos, list(reversed(pontos)))
        assert len(filho) == len(pontos)
        assert set(filho) == set(pontos)

    def test_sem_duplicatas(self):
        pontos = make_pontos(6)
        filho = pmx_crossover(pontos, list(reversed(pontos)))
        assert len(filho) == len(set(filho))

    def test_pais_identicos_retorna_pais(self):
        pontos = make_pontos(4)
        filho = pmx_crossover(pontos, list(pontos))
        assert set(filho) == set(pontos)

    def test_tamanho_preservado(self):
        pontos = make_pontos(8)
        filho = pmx_crossover(pontos, list(reversed(pontos)))
        assert len(filho) == 8

    def test_com_dois_pontos(self):
        pontos = make_pontos(2)
        filho = pmx_crossover(pontos, list(reversed(pontos)))
        assert set(filho) == set(pontos)


class TestOrderCrossover:
    def test_retorna_permutacao_valida(self):
        pontos = make_pontos(5)
        filho = order_crossover(pontos, list(reversed(pontos)))
        assert len(filho) == len(pontos)
        assert set(filho) == set(pontos)

    def test_sem_duplicatas(self):
        pontos = make_pontos(6)
        filho = order_crossover(pontos, list(reversed(pontos)))
        assert len(filho) == len(set(filho))

    def test_tamanho_preservado(self):
        pontos = make_pontos(7)
        filho = order_crossover(pontos, list(reversed(pontos)))
        assert len(filho) == 7

    def test_pais_identicos(self):
        pontos = make_pontos(4)
        filho = order_crossover(pontos, list(pontos))
        assert set(filho) == set(pontos)


class TestMutate:
    def test_probabilidade_zero_sem_mudanca(self):
        pontos = make_pontos(5)
        rota_original = list(pontos)
        mutada = mutate(pontos, probabilidade_mutacao=0.0)
        assert mutada == rota_original

    def test_probabilidade_um_muda_adjacentes(self):
        random.seed(42)
        pontos = make_pontos(5)
        rota_mutada = mutate(pontos, probabilidade_mutacao=1.0)
        # Com prob=1, deve trocar dois adjacentes
        assert set(rota_mutada) == set(pontos)
        assert len(rota_mutada) == len(pontos)

    def test_rota_muito_curta_sem_mutacao(self):
        pontos = make_pontos(1)
        mutada = mutate(pontos, probabilidade_mutacao=1.0)
        assert mutada == pontos

    def test_preserva_permutacao(self):
        random.seed(7)
        pontos = make_pontos(8)
        mutada = mutate(pontos, probabilidade_mutacao=0.9)
        assert set(mutada) == set(pontos)

    def test_retorna_copia_nao_modifica_original(self):
        pontos = make_pontos(4)
        original_copy = list(pontos)
        mutate(pontos, probabilidade_mutacao=1.0)
        assert pontos == original_copy


class TestMutateSegmentInversion:
    def test_probabilidade_zero_sem_mudanca(self):
        pontos = make_pontos(5)
        rota_original = list(pontos)
        mutada = mutate_segment_inversion(pontos, probabilidade_mutacao=0.0)
        assert mutada == rota_original

    def test_probabilidade_um_inverte_segmento(self):
        random.seed(42)
        pontos = make_pontos(6)
        mutada = mutate_segment_inversion(pontos, probabilidade_mutacao=1.0)
        assert set(mutada) == set(pontos)
        assert len(mutada) == len(pontos)

    def test_rota_curta_sem_mutacao(self):
        pontos = make_pontos(2)
        mutada = mutate_segment_inversion(pontos, probabilidade_mutacao=1.0)
        assert set(mutada) == set(pontos)

    def test_preserva_permutacao(self):
        random.seed(99)
        pontos = make_pontos(7)
        mutada = mutate_segment_inversion(pontos, probabilidade_mutacao=0.9)
        assert set(mutada) == set(pontos)
        assert len(mutada) == 7

    def test_retorna_copia(self):
        pontos = make_pontos(5)
        original = list(pontos)
        mutate_segment_inversion(pontos, probabilidade_mutacao=1.0)
        assert pontos == original
