"""Testes unitários para core/route_calculator.py."""

import math
import pytest
from core.route_calculator import calcular_distancia, calcular_distancia_rota
from data.delivery_points import PontoEntrega


def make_ponto(nome, x, y):
    return PontoEntrega(nome=nome, coords=(x, y), prioridade=3, tempo_atendimento=5)


class TestCalcularDistancia:
    def test_triangulo_3_4_5(self):
        p1 = make_ponto("A", 0, 0)
        p2 = make_ponto("B", 3, 4)
        assert calcular_distancia(p1, p2) == pytest.approx(5.0)

    def test_distancia_zero(self):
        p = make_ponto("A", 5, 5)
        assert calcular_distancia(p, p) == 0.0

    def test_distancia_simetrica(self):
        p1 = make_ponto("A", 1, 2)
        p2 = make_ponto("B", 4, 6)
        assert calcular_distancia(p1, p2) == pytest.approx(calcular_distancia(p2, p1))

    def test_distancia_horizontal(self):
        p1 = make_ponto("A", 0, 0)
        p2 = make_ponto("B", 10, 0)
        assert calcular_distancia(p1, p2) == pytest.approx(10.0)

    def test_distancia_vertical(self):
        p1 = make_ponto("A", 0, 0)
        p2 = make_ponto("B", 0, 7)
        assert calcular_distancia(p1, p2) == pytest.approx(7.0)

    def test_distancia_diagonal(self):
        p1 = make_ponto("A", 0, 0)
        p2 = make_ponto("B", 1, 1)
        assert calcular_distancia(p1, p2) == pytest.approx(math.sqrt(2))


class TestCalcularDistanciaRota:
    def test_rota_simples(self):
        hospital = PontoEntrega("H", (0, 0), 0, 0, 0.0, is_origin=True)
        p1 = make_ponto("A", 3, 0)
        p2 = make_ponto("B", 3, 4)
        # H→A: 3, A→B: 4, B→H: 5  → total = 12
        resultado = calcular_distancia_rota([p1, p2], hospital)
        assert resultado == pytest.approx(12.0)

    def test_rota_vazia_retorna_zero(self):
        hospital = PontoEntrega("H", (0, 0), 0, 0, 0.0, is_origin=True)
        assert calcular_distancia_rota([], hospital) == pytest.approx(0.0)

    def test_rota_ponto_unico(self):
        hospital = PontoEntrega("H", (0, 0), 0, 0, 0.0, is_origin=True)
        p = make_ponto("A", 4, 0)
        # H→A: 4, A→H: 4  → total = 8
        assert calcular_distancia_rota([p], hospital) == pytest.approx(8.0)
