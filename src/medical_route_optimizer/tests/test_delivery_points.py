"""Testes unitários para data/delivery_points.py."""

import pytest
from data.delivery_points import (
    PontoEntrega,
    PONTOS_ENTREGA,
    PRIORIDADE_LABEL,
    get_hospital_base,
    get_pontos_entrega_sem_origem,
)


class TestPontoEntrega:
    def test_criacao_basica(self):
        p = PontoEntrega(nome="X", coords=(1, 2), prioridade=1, tempo_atendimento=10)
        assert p.nome == "X"
        assert p.coords == (1, 2)
        assert p.prioridade == 1
        assert p.tempo_atendimento == 10
        assert p.peso == 1.0  # default
        assert p.is_origin is False  # default

    def test_peso_customizado(self):
        p = PontoEntrega(nome="Y", coords=(5, 5), prioridade=2, tempo_atendimento=5, peso=3.5)
        assert p.peso == 3.5

    def test_is_origin_true(self):
        p = PontoEntrega(nome="Base", coords=(0, 0), prioridade=0, tempo_atendimento=0, is_origin=True)
        assert p.is_origin is True

    def test_hash_usa_nome_e_coords(self):
        p1 = PontoEntrega(nome="A", coords=(1, 1), prioridade=1, tempo_atendimento=5)
        p2 = PontoEntrega(nome="A", coords=(1, 1), prioridade=2, tempo_atendimento=10)
        assert hash(p1) == hash(p2)

    def test_hash_diferente_para_coords_diferentes(self):
        p1 = PontoEntrega(nome="A", coords=(1, 1), prioridade=1, tempo_atendimento=5)
        p2 = PontoEntrega(nome="A", coords=(2, 2), prioridade=1, tempo_atendimento=5)
        assert hash(p1) != hash(p2)

    def test_eq_mesmas_coords(self):
        p1 = PontoEntrega(nome="A", coords=(3, 4), prioridade=1, tempo_atendimento=5)
        p2 = PontoEntrega(nome="B", coords=(3, 4), prioridade=2, tempo_atendimento=10)
        assert p1 == p2

    def test_eq_coords_diferentes(self):
        p1 = PontoEntrega(nome="A", coords=(1, 1), prioridade=1, tempo_atendimento=5)
        p2 = PontoEntrega(nome="A", coords=(2, 2), prioridade=1, tempo_atendimento=5)
        assert p1 != p2

    def test_eq_nao_ponto_entrega(self):
        p = PontoEntrega(nome="A", coords=(1, 1), prioridade=1, tempo_atendimento=5)
        assert p != "não é um PontoEntrega"
        assert p != 42
        assert p != None


class TestGetHospitalBase:
    def test_retorna_ponto_com_is_origin_true(self):
        base = get_hospital_base()
        assert base.is_origin is True

    def test_retorna_hospital_base_nome(self):
        base = get_hospital_base()
        assert "Hospital" in base.nome

    def test_retorna_coords_corretas(self):
        base = get_hospital_base()
        assert isinstance(base.coords, tuple)
        assert len(base.coords) == 2


class TestGetPontosEntregaSemOrigem:
    def test_nenhum_is_origin(self):
        pontos = get_pontos_entrega_sem_origem()
        assert all(not p.is_origin for p in pontos)

    def test_retorna_lista_nao_vazia(self):
        pontos = get_pontos_entrega_sem_origem()
        assert len(pontos) > 0

    def test_total_pontos_menos_um(self):
        todos = PONTOS_ENTREGA
        sem_origem = get_pontos_entrega_sem_origem()
        n_origens = sum(1 for p in todos if p.is_origin)
        assert len(sem_origem) == len(todos) - n_origens


class TestPrioridadeLabel:
    def test_chaves_corretas(self):
        assert 1 in PRIORIDADE_LABEL
        assert 2 in PRIORIDADE_LABEL
        assert 3 in PRIORIDADE_LABEL

    def test_valores(self):
        assert PRIORIDADE_LABEL[1] == "Alta"
        assert PRIORIDADE_LABEL[2] == "Média"
        assert PRIORIDADE_LABEL[3] == "Baixa"
