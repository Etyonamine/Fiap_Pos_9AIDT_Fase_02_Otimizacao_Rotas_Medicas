"""
Templates de prompts genéricos — versão final.

Inclui:
- prompt_instrucoes_operacionais
- prompt_relatorio_gerencial
- prompt_pergunta_linguagem_natural

Reforço para estrutura de saída e instrução para não inventar dados.
"""

import json
from typing import Dict, Any


def prompt_instrucoes_operacionais(relatorio: Dict[str, Any]) -> str:
    dados_rota = json.dumps(relatorio, ensure_ascii=False, indent=2)
    return f"""Você é um assistente operacional de um hospital responsável pela distribuição de medicamentos e insumos.

Gere instruções claras e objetivas em português para a equipe de entrega, seguindo esta estrutura:
- RESUMO RÁPIDO
- SEQUÊNCIA NUMERADA DE PARADAS (com prioridade, tempo deslocamento em minutos, tempo atendimento em minutos)
- CHECKLIST OPERACIONAL (por parada)
- TEMPO TOTAL ESTIMADO
- ALERTAS CRÍTICOS
- PROMPT_USADO (incluir o prompt exato enviado)

IMPORTANTE: Responda SOMENTE com base nos dados abaixo. Se alguma informação não estiver disponível, escreva explicitamente "dado não disponível".
--- DADOS DA ROTA ---
{dados_rota}
--- FIM DOS DADOS ---

Gere as instruções operacionais agora:"""


def prompt_relatorio_gerencial(relatorio: Dict[str, Any]) -> str:
    dados_rota = json.dumps(relatorio, ensure_ascii=False, indent=2)
    return f"""Você é um analista de operações logísticas em uma instituição de saúde.

Elabore um relatório gerencial em português contendo:
- RESUMO EXECUTIVO
- INDICADORES (distância total em km, tempo total em minutos, atendimentos por prioridade)
- COMPARAÇÃO COM BASELINE (Nearest Neighbor)
- ANÁLISE DO PROCESSO EVOLUTIVO (GA)
- SUGESTÕES DE MELHORIA (mín. 2)
- PROMPT_USADO

IMPORTANTE: Responda SOMENTE com base nos dados abaixo. Indique quando informação não estiver disponível.
--- DADOS DA ROTA ---
{dados_rota}
--- FIM DOS DADOS ---

Elabore o relatório gerencial agora:"""


def prompt_pergunta_linguagem_natural(relatorio: Dict[str, Any], pergunta: str) -> str:
    dados_rota = json.dumps(relatorio, ensure_ascii=False, indent=2)
    return f"""Você é um assistente inteligente de rotas de entrega médica.

PERGUNTA: {pergunta}

Responda de forma direta e em português, baseando-se EXCLUSIVAMENTE nos dados da rota abaixo.
Se a informação solicitada não estiver disponível, diga explicitamente que não é possível responder com as informações disponíveis.
--- DADOS DA ROTA ---
{dados_rota}
--- FIM DOS DADOS ---

Resposta:"""
