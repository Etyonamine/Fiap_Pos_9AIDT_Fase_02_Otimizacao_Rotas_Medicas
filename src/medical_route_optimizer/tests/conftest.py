"""Fixtures compartilhadas para todos os testes do medical_route_optimizer."""

import os

import pytest
from data.delivery_points import PontoEntrega

os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def hospital():
    return PontoEntrega(
        nome="Hospital Base",
        coords=(0, 0),
        prioridade=0,
        tempo_atendimento=0,
        peso=0.0,
        is_origin=True,
    )


@pytest.fixture
def ponto_a():
    return PontoEntrega(
        nome="A",
        coords=(3, 4),
        prioridade=1,
        tempo_atendimento=10,
        peso=2.0,
    )


@pytest.fixture
def ponto_b():
    return PontoEntrega(
        nome="B",
        coords=(6, 8),
        prioridade=2,
        tempo_atendimento=5,
        peso=1.5,
    )


@pytest.fixture
def ponto_c():
    return PontoEntrega(
        nome="C",
        coords=(0, 8),
        prioridade=3,
        tempo_atendimento=15,
        peso=3.0,
    )


@pytest.fixture
def pontos_tres(ponto_a, ponto_b, ponto_c):
    return [ponto_a, ponto_b, ponto_c]
