"""Testes unitários para llm/prompts_perfil.py."""

import pytest
from llm.prompts_perfil import (
    PERFIS,
    prompt_motorista,
    prompt_operador,
    prompt_gerente,
    prompt_por_perfil,
    _build_prompt_motorista,
    _build_prompt_operador,
    _build_prompt_gerente,
)


@pytest.fixture
def relatorio_simples():
    return {
        "resumo": {"total_pontos_entrega": 2, "custo_otimizado": 50.0},
        "vrp": {
            "n_veiculos": 1,
            "solucao_valida": True,
            "veiculos": [{"veiculo": 1, "n_pontos": 2}],
        },
    }


class TestPerfis:
    def test_chaves_exibicao_presentes(self):
        assert "🚚 Motorista" in PERFIS
        assert "🖥️ Operador" in PERFIS
        assert "📊 Gerente" in PERFIS

    def test_valores_internos(self):
        assert PERFIS["🚚 Motorista"] == "motorista"
        assert PERFIS["🖥️ Operador"] == "operador"
        assert PERFIS["📊 Gerente"] == "gerente"


class TestBuildPromptMotorista:
    def test_retorna_string(self):
        resultado = _build_prompt_motorista("dados de teste")
        assert isinstance(resultado, str)

    def test_contem_dados(self):
        resultado = _build_prompt_motorista("DADOS_ESPECIAIS_123")
        assert "DADOS_ESPECIAIS_123" in resultado

    def test_nao_vazio(self):
        resultado = _build_prompt_motorista("x")
        assert len(resultado) > 50


class TestBuildPromptOperador:
    def test_retorna_string(self):
        resultado = _build_prompt_operador("dados de teste")
        assert isinstance(resultado, str)

    def test_contem_dados(self):
        resultado = _build_prompt_operador("DADOS_OP_456")
        assert "DADOS_OP_456" in resultado

    def test_nao_vazio(self):
        resultado = _build_prompt_operador("x")
        assert len(resultado) > 50


class TestBuildPromptGerente:
    def test_retorna_string(self):
        resultado = _build_prompt_gerente("dados de teste")
        assert isinstance(resultado, str)

    def test_contem_dados(self):
        resultado = _build_prompt_gerente("DADOS_GER_789")
        assert "DADOS_GER_789" in resultado

    def test_nao_vazio(self):
        resultado = _build_prompt_gerente("x")
        assert len(resultado) > 50


class TestPromptMotorista:
    def test_retorna_string(self, relatorio_simples):
        resultado = prompt_motorista(relatorio_simples)
        assert isinstance(resultado, str)

    def test_contem_dados_json(self, relatorio_simples):
        resultado = prompt_motorista(relatorio_simples)
        assert "total_pontos_entrega" in resultado


class TestPromptOperador:
    def test_retorna_string(self, relatorio_simples):
        resultado = prompt_operador(relatorio_simples)
        assert isinstance(resultado, str)

    def test_contem_dados_json(self, relatorio_simples):
        resultado = prompt_operador(relatorio_simples)
        assert "total_pontos_entrega" in resultado


class TestPromptGerente:
    def test_retorna_string(self, relatorio_simples):
        resultado = prompt_gerente(relatorio_simples)
        assert isinstance(resultado, str)

    def test_contem_dados_json(self, relatorio_simples):
        resultado = prompt_gerente(relatorio_simples)
        assert "total_pontos_entrega" in resultado


class TestPromptPorPerfil:
    def test_chave_exibicao_motorista(self, relatorio_simples):
        resultado = prompt_por_perfil("🚚 Motorista", relatorio_simples)
        assert isinstance(resultado, str)
        assert len(resultado) > 50

    def test_chave_exibicao_operador(self, relatorio_simples):
        resultado = prompt_por_perfil("🖥️ Operador", relatorio_simples)
        assert isinstance(resultado, str)

    def test_chave_exibicao_gerente(self, relatorio_simples):
        resultado = prompt_por_perfil("📊 Gerente", relatorio_simples)
        assert isinstance(resultado, str)

    def test_chave_interna_motorista(self, relatorio_simples):
        resultado = prompt_por_perfil("motorista", relatorio_simples)
        assert isinstance(resultado, str)

    def test_chave_interna_operador(self, relatorio_simples):
        resultado = prompt_por_perfil("operador", relatorio_simples)
        assert isinstance(resultado, str)

    def test_chave_interna_gerente(self, relatorio_simples):
        resultado = prompt_por_perfil("gerente", relatorio_simples)
        assert isinstance(resultado, str)

    def test_perfil_invalido_lanca_value_error(self, relatorio_simples):
        with pytest.raises(ValueError, match="não encontrado"):
            prompt_por_perfil("perfil_inexistente", relatorio_simples)
