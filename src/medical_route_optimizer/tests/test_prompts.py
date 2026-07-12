"""Testes unitários para llm/prompts.py."""

import pytest
from llm.prompts import (
    prompt_instrucoes_operacionais,
    prompt_relatorio_gerencial,
    prompt_pergunta_linguagem_natural,
)


@pytest.fixture
def relatorio_simples():
    return {
        "resumo": {"total_pontos_entrega": 3, "custo_otimizado": 100.0},
        "origem_retorno": "Hospital Base",
        "sequencia_atendimentos": [],
    }


class TestPromptInstrucoesOperacionais:
    def test_retorna_string(self, relatorio_simples):
        resultado = prompt_instrucoes_operacionais(relatorio_simples)
        assert isinstance(resultado, str)

    def test_contem_dados_do_relatorio(self, relatorio_simples):
        resultado = prompt_instrucoes_operacionais(relatorio_simples)
        assert "Hospital Base" in resultado

    def test_contem_instrucoes(self, relatorio_simples):
        resultado = prompt_instrucoes_operacionais(relatorio_simples)
        assert "DADOS DA ROTA" in resultado

    def test_nao_vazio(self, relatorio_simples):
        resultado = prompt_instrucoes_operacionais(relatorio_simples)
        assert len(resultado) > 100


class TestPromptRelatorioGerencial:
    def test_retorna_string(self, relatorio_simples):
        resultado = prompt_relatorio_gerencial(relatorio_simples)
        assert isinstance(resultado, str)

    def test_contem_dados(self, relatorio_simples):
        resultado = prompt_relatorio_gerencial(relatorio_simples)
        assert "Hospital Base" in resultado

    def test_contem_secao_gerencial(self, relatorio_simples):
        resultado = prompt_relatorio_gerencial(relatorio_simples)
        assert "DADOS DA ROTA" in resultado

    def test_nao_vazio(self, relatorio_simples):
        resultado = prompt_relatorio_gerencial(relatorio_simples)
        assert len(resultado) > 100


class TestPromptPerguntaLinguagemNatural:
    def test_retorna_string(self, relatorio_simples):
        resultado = prompt_pergunta_linguagem_natural(
            relatorio_simples, "Qual é o total de pontos?"
        )
        assert isinstance(resultado, str)

    def test_contem_pergunta(self, relatorio_simples):
        pergunta = "Quantos pontos foram visitados?"
        resultado = prompt_pergunta_linguagem_natural(relatorio_simples, pergunta)
        assert pergunta in resultado

    def test_contem_dados_relatorio(self, relatorio_simples):
        resultado = prompt_pergunta_linguagem_natural(
            relatorio_simples, "pergunta teste"
        )
        assert "Hospital Base" in resultado

    def test_nao_vazio(self, relatorio_simples):
        resultado = prompt_pergunta_linguagem_natural(
            relatorio_simples, "Alguma pergunta"
        )
        assert len(resultado) > 50
