"""Testes unitários para core/vrp_split.py."""

import pytest
from core.vrp_split import dividir_rotas_vrp, calcular_custo_vrp, resumo_restricoes_vrp
from data.delivery_points import PontoEntrega


def make_ponto(nome, x, y, prioridade=3, peso=1.0):
    return PontoEntrega(nome=nome, coords=(x, y), prioridade=prioridade,
                        tempo_atendimento=5, peso=peso)


@pytest.fixture
def hospital():
    return PontoEntrega("H", (0, 0), 0, 0, 0.0, is_origin=True)


class TestDividirRotasVrp:
    def test_sem_restricoes_uma_rota(self, hospital):
        pontos = [make_ponto("A", 5, 0), make_ponto("B", 10, 0)]
        rotas = dividir_rotas_vrp(pontos, hospital)
        assert len(rotas) == 1
        assert set(rotas[0]) == set(pontos)

    def test_split_por_capacidade(self, hospital):
        p1 = make_ponto("A", 5, 0, peso=3.0)
        p2 = make_ponto("B", 10, 0, peso=3.0)
        p3 = make_ponto("C", 15, 0, peso=3.0)
        # capacidade=4: p1(3)+p2(3)>4 → split
        rotas = dividir_rotas_vrp([p1, p2, p3], hospital,
                                   capacidade_veiculo=4.0)
        assert len(rotas) >= 2
        # Todos os pontos cobertos
        todos = [p for r in rotas for p in r]
        assert set(todos) == {p1, p2, p3}

    def test_split_por_autonomia(self, hospital):
        p1 = make_ponto("A", 50, 0)
        p2 = make_ponto("B", 100, 0)
        # autonomia=60: H→A(50) + A→B(50) + B→H(100) = 200 → split
        rotas = dividir_rotas_vrp([p1, p2], hospital,
                                   autonomia_veiculo=60.0)
        assert len(rotas) >= 2

    def test_n_veiculos_limite(self, hospital):
        pontos = [make_ponto(str(i), i * 5, 0, peso=10.0) for i in range(4)]
        # capacidade=5: cada ponto excede, mas n_veiculos=2 limita a 2 rotas
        rotas = dividir_rotas_vrp(pontos, hospital,
                                   capacidade_veiculo=5.0,
                                   n_veiculos=2)
        assert len(rotas) <= 2

    def test_n_veiculos_none_sem_limite(self, hospital):
        pontos = [make_ponto(str(i), i * 5, 0, peso=10.0) for i in range(3)]
        rotas = dividir_rotas_vrp(pontos, hospital,
                                   capacidade_veiculo=5.0,
                                   n_veiculos=None)
        todos = [p for r in rotas for p in r]
        assert set(todos) == set(pontos)

    def test_rota_vazia(self, hospital):
        rotas = dividir_rotas_vrp([], hospital)
        assert rotas == []

    def test_ponto_unico(self, hospital):
        p = make_ponto("A", 5, 0)
        rotas = dividir_rotas_vrp([p], hospital)
        assert len(rotas) == 1
        assert rotas[0] == [p]

    def test_todos_pontos_cobertos(self, hospital):
        pontos = [make_ponto(str(i), i * 5, 0, peso=2.0) for i in range(6)]
        rotas = dividir_rotas_vrp(pontos, hospital, capacidade_veiculo=4.0, n_veiculos=4)
        todos = [p for r in rotas for p in r]
        assert set(todos) == set(pontos)


class TestCalcularCustoVrp:
    def test_retorna_custo_total_e_lista(self, hospital):
        r1 = [make_ponto("A", 5, 0)]
        r2 = [make_ponto("B", 0, 5)]
        total, custos = calcular_custo_vrp([r1, r2], hospital)
        assert isinstance(total, float)
        assert isinstance(custos, list)
        assert len(custos) == 2

    def test_custo_total_e_soma_individuais(self, hospital):
        r1 = [make_ponto("A", 5, 0)]
        r2 = [make_ponto("B", 0, 5)]
        total, custos = calcular_custo_vrp([r1, r2], hospital)
        assert total == pytest.approx(sum(custos))

    def test_com_restricoes(self, hospital):
        r1 = [make_ponto("A", 5, 0, peso=2.0)]
        total, _ = calcular_custo_vrp([r1], hospital,
                                       capacidade_veiculo=5.0,
                                       autonomia_veiculo=100.0)
        assert total > 0


class TestResumoRestricoesVrp:
    def test_retorna_lista_de_dicts(self, hospital):
        r1 = [make_ponto("A", 5, 0, peso=2.0)]
        resumo = resumo_restricoes_vrp([r1], hospital,
                                        capacidade_veiculo=10.0,
                                        autonomia_veiculo=100.0)
        assert isinstance(resumo, list)
        assert len(resumo) == 1

    def test_campos_obrigatorios(self, hospital):
        r1 = [make_ponto("A", 5, 0, peso=2.0)]
        resumo = resumo_restricoes_vrp([r1], hospital,
                                        capacidade_veiculo=10.0,
                                        autonomia_veiculo=100.0)
        v = resumo[0]
        assert "veiculo" in v
        assert "n_pontos" in v
        assert "peso_total" in v
        assert "capacidade_ok" in v
        assert "autonomia_ok" in v
        assert "pontos" in v

    def test_capacidade_ok_true(self, hospital):
        p = make_ponto("A", 5, 0, peso=2.0)
        resumo = resumo_restricoes_vrp([[p]], hospital,
                                        capacidade_veiculo=10.0,
                                        autonomia_veiculo=None)
        assert resumo[0]["capacidade_ok"] is True

    def test_capacidade_ok_false(self, hospital):
        p = make_ponto("A", 5, 0, peso=15.0)
        resumo = resumo_restricoes_vrp([[p]], hospital,
                                        capacidade_veiculo=10.0,
                                        autonomia_veiculo=None)
        assert resumo[0]["capacidade_ok"] is False

    def test_autonomia_ok_true(self, hospital):
        p = make_ponto("A", 1, 0)
        resumo = resumo_restricoes_vrp([[p]], hospital,
                                        capacidade_veiculo=None,
                                        autonomia_veiculo=100.0)
        assert resumo[0]["autonomia_ok"] is True

    def test_autonomia_ok_false(self, hospital):
        p = make_ponto("A", 1000, 0)
        resumo = resumo_restricoes_vrp([[p]], hospital,
                                        capacidade_veiculo=None,
                                        autonomia_veiculo=10.0)
        assert resumo[0]["autonomia_ok"] is False

    def test_restricoes_none(self, hospital):
        p = make_ponto("A", 5, 0, peso=2.0)
        resumo = resumo_restricoes_vrp([[p]], hospital,
                                        capacidade_veiculo=None,
                                        autonomia_veiculo=None)
        assert resumo[0]["capacidade_ok"] is True
        assert resumo[0]["autonomia_ok"] is True

    def test_numeracao_veiculos(self, hospital):
        r1 = [make_ponto("A", 5, 0)]
        r2 = [make_ponto("B", 0, 5)]
        resumo = resumo_restricoes_vrp([r1, r2], hospital, None, None)
        assert resumo[0]["veiculo"] == 1
        assert resumo[1]["veiculo"] == 2

    def test_pontos_nomes(self, hospital):
        p = make_ponto("Ponto X", 5, 0)
        resumo = resumo_restricoes_vrp([[p]], hospital, None, None)
        assert "Ponto X" in resumo[0]["pontos"]
