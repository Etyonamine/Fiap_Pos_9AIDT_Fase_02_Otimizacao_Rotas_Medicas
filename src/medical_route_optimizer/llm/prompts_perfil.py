"""
Engenharia de Prompts por Perfil de Usuário.

Três perfis especializados para o sistema de distribuição de medicamentos:

    motorista  → instruções operacionais diretas, sequência de paradas, alertas urgentes
    operador   → monitoramento de execução, restrições VRP, ações corretivas
    gerente    → KPIs de eficiência, comparação com baseline, insights estratégicos

Cada perfil recebe o mesmo relatório estruturado (dict do gerar_relatorio_rota),
mas o prompt orienta o modelo a adotar tom, foco e nível de detalhamento adequados
ao papel do usuário, garantindo respostas úteis e sem alucinações.

Escalabilidade: para adicionar um novo perfil, basta registrá-lo em PERFIS e
implementar a função de prompt correspondente, sem alterar o restante do sistema.
"""

import json
from typing import Dict, Any, Callable


# ---------------------------------------------------------------------------
# Registro de perfis (chave de exibição → chave interna)
# ---------------------------------------------------------------------------
PERFIS: Dict[str, str] = {
    "🚚 Motorista":  "motorista",
    "🖥️ Operador":   "operador",
    "📊 Gerente":    "gerente",
}


# ---------------------------------------------------------------------------
# Prompts por perfil
# ---------------------------------------------------------------------------

def prompt_motorista(relatorio: Dict[str, Any]) -> str:
    """
    Prompt para o motorista da entrega.

    Foco: o que fazer, onde ir, em que ordem, com quais alertas de urgência.
    Tom: direto, simples, operacional.
    """
    dados = json.dumps(relatorio, ensure_ascii=False, indent=2)

    return f"""Você é um assistente de logística hospitalar que prepara o roteiro diário para motoristas de entrega.

PERFIL DO USUÁRIO: Motorista — lê as instruções no dispositivo antes de sair do hospital.
TOM ESPERADO: direto, sem termos técnicos, fácil de seguir ao volante.

Com base nos dados da rota abaixo, gere um roteiro de entrega claro e numerado em português, contendo:

1. PONTO DE PARTIDA: nome e descrição breve do hospital base.
2. SEQUÊNCIA DE PARADAS: liste cada parada numerada com:
   - Nome do destino
   - Nível de urgência (🔴 URGENTE se prioridade Alta, 🟡 ATENÇÃO se Média, 🟢 Normal se Baixa)
   - Tempo estimado de deslocamento até o destino
   - Tempo de atendimento no local
3. ALERTAS CRÍTICOS: destaque em negrito qualquer entrega de prioridade ALTA (prioridade 1) e oriente o motorista a não atrasar essas paradas.
4. RETORNO: instruções de retorno ao hospital base com tempo estimado.
5. TEMPO TOTAL DA ROTA: informe o tempo total estimado da jornada.

REGRAS:
- Responda SOMENTE com base nos dados fornecidos.
- Não invente endereços, horários ou informações ausentes.
- Se um veículo tiver rota própria (VRP), gere o roteiro separado por veículo.
- Use linguagem acessível — o motorista não é especialista em logística.

--- DADOS DA ROTA ---
{dados}
--- FIM DOS DADOS ---

Gere o roteiro de entrega agora:"""


def prompt_operador(relatorio: Dict[str, Any]) -> str:
    """
    Prompt para o operador da central de distribuição.

    Foco: monitoramento do status das rotas, violações de restrições VRP,
    ações corretivas recomendadas.
    Tom: técnico-operacional, focado em exceções e conformidade.
    """
    dados = json.dumps(relatorio, ensure_ascii=False, indent=2)

    return f"""Você é um sistema de apoio a operadores da central de logística hospitalar.

PERFIL DO USUÁRIO: Operador — monitora em tempo real a execução das rotas de distribuição.
TOM ESPERADO: técnico, objetivo, focado em exceções, violações e ações corretivas.

Com base nos dados abaixo, elabore um painel de monitoramento operacional em português, contendo:

1. STATUS GERAL DA OPERAÇÃO
   - Número de veículos em rota
   - Total de pontos de entrega (discriminados por prioridade)
   - Solução VRP válida ou com violações (capacidade/autonomia por veículo)

2. DETALHAMENTO POR VEÍCULO (para cada veículo na solução VRP)
   - Carga atual vs capacidade máxima: ✅ dentro / ❌ excedida
   - Distância percorrida vs autonomia: ✅ dentro / ❌ excedida
   - Lista dos pontos de entrega com prioridade

3. ALERTAS E VIOLAÇÕES
   - Liste qualquer restrição violada (capacidade, autonomia, prioridade crítica tardia)
   - Para cada violação, sugira uma ação corretiva concreta (ex.: reatribuir ponto a outro veículo)

4. QUALIDADE DA SOLUÇÃO
   - Custo otimizado (GA + Two Opt) vs custo baseline (Nearest Neighbor)
   - Percentual de melhoria alcançado
   - Número de gerações necessárias para convergência

5. RECOMENDAÇÕES OPERACIONAIS
   - Pelo menos 2 ajustes que o operador pode fazer para melhorar a execução atual

REGRAS:
- Baseie-se EXCLUSIVAMENTE nos dados fornecidos.
- Indique claramente quando uma informação não estiver disponível.
- Priorize alertas críticos (prioridade 1) no topo da análise.

--- DADOS DA ROTA ---
{dados}
--- FIM DOS DADOS ---

Gere o painel de monitoramento agora:"""


def prompt_gerente(relatorio: Dict[str, Any]) -> str:
    """
    Prompt para o gerente de distribuição.

    Foco: KPIs estratégicos, eficiência da frota, comparação com baseline,
    tendências e recomendações de melhoria de longo prazo.
    Tom: executivo, analítico, orientado a resultados.
    """
    dados = json.dumps(relatorio, ensure_ascii=False, indent=2)

    return f"""Você é um analista sênior de operações logísticas de uma rede hospitalar.

PERFIL DO USUÁRIO: Gerente de Distribuição — toma decisões estratégicas sobre a frota e os processos.
TOM ESPERADO: executivo, analítico, orientado a KPIs e resultados de negócio.

Com base nos dados abaixo, elabore um relatório gerencial executivo em português, contendo:

1. RESUMO EXECUTIVO (3-5 linhas)
   - Síntese da operação do dia: rotas realizadas, entregas, eficiência geral.

2. KPIs DE EFICIÊNCIA
   - Distância total percorrida (km equivalente)
   - Tempo total estimado da operação (minutos)
   - Custo de rota otimizado vs. custo baseline (Nearest Neighbor)
   - Percentual de economia gerado pelo Algoritmo Genético
   - Utilização de capacidade da frota (% da capacidade total utilizada)
   - Conformidade de autonomia (% de rotas dentro do limite)

3. ANÁLISE DO PROCESSO DE OTIMIZAÇÃO
   - Em quantas gerações o GA convergiu
   - Custo inicial vs. custo final após evolução e refinamento Two Opt
   - Interpretação da curva de convergência (estagnação precoce, boa diversidade, etc.)

4. DISTRIBUIÇÃO POR PRIORIDADE
   - Quantos pontos de alta, média e baixa prioridade foram atendidos
   - Análise se a rota privilegiou corretamente as entregas críticas

5. INSIGHTS ESTRATÉGICOS E RECOMENDAÇÕES
   - Com base nos padrões identificados, sugira pelo menos 3 ações de melhoria operacional
     (ex.: ajuste de parâmetros do GA, redistribuição da frota, janelas de tempo, etc.)
   - Avalie o custo-benefício de adicionar mais um veículo à operação, se aplicável

REGRAS:
- Baseie-se EXCLUSIVAMENTE nos dados fornecidos.
- Quando uma métrica não estiver disponível, indique "dado não disponível nesta execução".
- Evite linguagem técnica de algoritmos — traduza para impacto de negócio.

--- DADOS DA ROTA ---
{dados}
--- FIM DOS DADOS ---

Elabore o relatório gerencial agora:"""


# ---------------------------------------------------------------------------
# Dispatcher público
# ---------------------------------------------------------------------------

_PROMPT_FUNC: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "motorista": prompt_motorista,
    "operador":  prompt_operador,
    "gerente":   prompt_gerente,
}


def prompt_por_perfil(chave_exibicao: str, relatorio: Dict[str, Any]) -> str:
    """
    Retorna o prompt correto para o perfil selecionado.

    Parâmetros:
    - chave_exibicao: chave de exibição conforme definida em PERFIS
                      (ex.: "🚚 Motorista") ou chave interna (ex.: "motorista")
    - relatorio: dicionário gerado por gerar_relatorio_rota()

    Retorno:
    - prompt formatado como string pronto para envio ao LLM
    """
    # Aceita tanto a chave de exibição quanto a chave interna
    chave_interna = PERFIS.get(chave_exibicao, chave_exibicao)
    func = _PROMPT_FUNC.get(chave_interna)
    if func is None:
        raise ValueError(
            f"Perfil '{chave_exibicao}' não encontrado. "
            f"Perfis disponíveis: {list(PERFIS.keys())}"
        )
    return func(relatorio)
