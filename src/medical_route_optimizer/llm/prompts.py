"""
Templates de prompts para integração com LLM.

Três prompts especializados:
1. Instruções operacionais para a equipe de entrega.
2. Relatório gerencial de eficiência de rota.
3. Respostas a perguntas em linguagem natural sobre a rota.

Todos os prompts recebem os dados estruturados do relatório da rota
e orientam o modelo a responder APENAS com base nas informações fornecidas,
evitando alucinações ou informações inventadas.
"""

import json
from typing import Dict, Any


def prompt_instrucoes_operacionais(relatorio: Dict[str, Any]) -> str:
    """
    Gera prompt para instruções operacionais direcionadas à equipe de entrega.

    A LLM deve produzir um texto claro, objetivo e em português,
    com a sequência de atendimentos, alertas de prioridade e tempos estimados.

    Parâmetros:
    - relatorio: dicionário gerado por gerar_relatorio_rota()

    Retorno:
    - prompt formatado como string
    """
    dados_rota = json.dumps(relatorio, ensure_ascii=False, indent=2)

    return f"""Você é um assistente operacional de um hospital responsável pela distribuição de medicamentos e insumos.

Com base nos dados da rota otimizada abaixo, gere instruções claras e objetivas em português para a equipe de entrega.

As instruções devem conter:
1. Ponto de partida e retorno (hospital base).
2. Sequência completa de atendimentos, numerada.
3. Destaque para os pacientes ou unidades de ALTA PRIORIDADE (prioridade 1), indicando que devem ser atendidos com urgência.
4. Tempo estimado de deslocamento e atendimento em cada parada.
5. Tempo total estimado da rota.
6. Alertas relevantes com base nos dados fornecidos.

IMPORTANTE: Responda SOMENTE com base nos dados abaixo. Não invente informações, endereços ou horários que não estejam presentes.

--- DADOS DA ROTA ---
{dados_rota}
--- FIM DOS DADOS ---

Gere as instruções operacionais agora:"""


def prompt_relatorio_gerencial(relatorio: Dict[str, Any]) -> str:
    """
    Gera prompt para relatório gerencial de eficiência da rota.

    A LLM deve produzir um relatório executivo com indicadores de desempenho,
    comparação com o baseline e sugestões de melhoria.

    Parâmetros:
    - relatorio: dicionário gerado por gerar_relatorio_rota()

    Retorno:
    - prompt formatado como string
    """
    dados_rota = json.dumps(relatorio, ensure_ascii=False, indent=2)

    return f"""Você é um analista de operações logísticas em uma instituição de saúde.

Com base nos dados da rota otimizada abaixo, elabore um relatório gerencial em português contendo:

1. RESUMO EXECUTIVO: principais métricas da rota do dia.
2. INDICADORES DE EFICIÊNCIA:
   - Distância total percorrida.
   - Tempo total estimado.
   - Número de atendimentos realizados por prioridade.
3. COMPARAÇÃO COM BASELINE: ganho de eficiência em relação à rota do método do Vizinho Mais Próximo (Nearest Neighbor).
4. ANÁLISE DO PROCESSO EVOLUTIVO: como o Algoritmo Genético melhorou a rota ao longo das gerações.
5. SUGESTÕES DE MELHORIA: com base nos padrões identificados nos dados, sugira pelo menos 2 melhorias operacionais.

IMPORTANTE: Responda SOMENTE com base nos dados abaixo. Indique claramente quando uma informação não estiver disponível.

--- DADOS DA ROTA ---
{dados_rota}
--- FIM DOS DADOS ---

Elabore o relatório gerencial agora:"""


def prompt_pergunta_linguagem_natural(
    relatorio: Dict[str, Any],
    pergunta: str
) -> str:
    """
    Gera prompt para resposta a uma pergunta em linguagem natural sobre a rota.

    A LLM deve responder de forma direta e em português, baseando-se
    exclusivamente nos dados da rota fornecidos.

    Parâmetros:
    - relatorio: dicionário gerado por gerar_relatorio_rota()
    - pergunta: pergunta do usuário em linguagem natural

    Retorno:
    - prompt formatado como string
    """
    dados_rota = json.dumps(relatorio, ensure_ascii=False, indent=2)

    return f"""Você é um assistente inteligente de rotas de entrega médica.

Um usuário fez a seguinte pergunta sobre a rota de entregas do dia:

PERGUNTA: {pergunta}

Responda à pergunta de forma direta, clara e em português, baseando-se EXCLUSIVAMENTE nos dados da rota abaixo.
Se a informação solicitada não estiver disponível nos dados, diga explicitamente que não é possível responder com as informações disponíveis.
NÃO invente dados, endereços, horários ou qualquer informação não presente nos dados fornecidos.

--- DADOS DA ROTA ---
{dados_rota}
--- FIM DOS DADOS ---

Resposta:"""
