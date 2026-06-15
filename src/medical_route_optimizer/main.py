"""
Orquestrador principal — Sistema de Otimização de Rotas Médicas.

Pipeline completo:
    1. Carrega pontos de entrega do domínio médico (hospital base + locais de entrega).
    2. Gera população inicial híbrida: Nearest Neighbor + aleatória.
    3. Executa o Algoritmo Genético com elitismo e fitness com prioridade.
    4. Aplica Two Opt Inversion para refinamento local da melhor rota.
    5. Gera relatório estruturado com métricas e comparação com baseline NN.
    6. Envia dados para LLM e exibe:
        a. Instruções operacionais para a equipe de entrega.
        b. Relatório gerencial de eficiência.
    7. Modo interativo: aceita perguntas em linguagem natural sobre a rota.

Uso:
    python -m main

Variáveis de ambiente (configurar antes de executar com LLM):
    LLM_PROVIDER    → "openai" ou "groq" (padrão: openai)
    OPENAI_API_KEY  → chave de API OpenAI (se usar openai)
    GROQ_API_KEY    → chave de API Groq   (se usar groq)
    LLM_MODEL       → modelo específico (opcional)
    USE_LLM         → "true" para habilitar chamadas à LLM (padrão: false)
"""

import os
import sys
import json

# Garante que o diretório src está no path para imports relativos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.delivery_points import (
    get_hospital_base,
    get_pontos_entrega_sem_origem,
    PRIORIDADE_LABEL,
)
from core.genetic_algorithm import (
    gerar_populacao_aleatoria,
    executar_algoritmo_genetico,
)
from core.nearest_neighbor import (
    gerar_populacao_nearest_neighbor,
    avaliar_baseline_nn,
)
from core.two_opt import two_opt_inversion
from core.vrp_split import (
    dividir_rotas_vrp,
    resumo_restricoes_vrp,
)
from reports.route_report import gerar_relatorio_rota
from llm.prompts import (
    prompt_instrucoes_operacionais,
    prompt_relatorio_gerencial,
    prompt_pergunta_linguagem_natural,
)


# ---------------------------------------------------------------------------
# Parâmetros do GA
# ---------------------------------------------------------------------------
TAMANHO_POPULACAO = 150
N_GERACOES = 200        # limite máximo de gerações
PACIENCIA = 50          # gerações sem melhora para parada antecipada
PROBABILIDADE_MUTACAO = 0.3
PROPORCAO_NN = 0.15  # 15% da população inicial gerada por Nearest Neighbor

# ---------------------------------------------------------------------------
# Parâmetros VRP (Roteamento de Veículos)
# ---------------------------------------------------------------------------
N_VEICULOS = 2           # número máximo de veículos disponíveis
CAPACIDADE_VEICULO = 16  # carga máxima por veículo (unidades/kg)
AUTONOMIA_VEICULO = 1400 # distância máxima por ciclo em pixels (~140 km)


def _separador(titulo: str = "") -> None:
    linha = "=" * 60
    if titulo:
        print(f"\n{linha}")
        print(f"  {titulo}")
        print(linha)
    else:
        print(linha)


def _usar_llm() -> bool:
    return os.getenv("USE_LLM", "false").lower() == "true"


def _chamar_llm_seguro(prompt: str, descricao: str) -> str:
    """Chama a LLM com tratamento de erro gracioso."""
    try:
        from llm.llm_client import chamar_llm
        return chamar_llm(prompt)
    except (EnvironmentError, ImportError, ValueError) as e:
        return f"[LLM indisponível — {descricao}]\nErro: {e}"
    except Exception as e:
        return f"[Erro inesperado ao chamar LLM — {descricao}]\nErro: {e}"


def main():
    _separador("Sistema de Otimização de Rotas Médicas")
    print("  Algoritmo Genético + Nearest Neighbor + Two Opt Inversion + VRP Split")
    print("  Integração com LLM para instruções e relatórios")
    _separador()

    # ------------------------------------------------------------------
    # 1. Carregar dados
    # ------------------------------------------------------------------
    hospital_base = get_hospital_base()
    locais_entrega = get_pontos_entrega_sem_origem()

    print(f"\n📍 Hospital base: {hospital_base.nome}")
    print(f"📦 Pontos de entrega: {len(locais_entrega)}")
    peso_total_carga = sum(p.peso for p in locais_entrega)
    for p in locais_entrega:
        print(f"   [{PRIORIDADE_LABEL.get(p.prioridade, '?')}] {p.nome}  ({p.peso:.1f} un.)")
    print(f"\n⚙️  Restrições VRP:")
    print(f"   Veículos disponíveis : {N_VEICULOS}")
    print(f"   Capacidade/veículo   : {CAPACIDADE_VEICULO} un.  "
          f"(carga total: {peso_total_carga:.1f} un.)")
    print(f"   Autonomia/veículo    : {AUTONOMIA_VEICULO} px")

    # ------------------------------------------------------------------
    # 2. Baseline: Nearest Neighbor
    # ------------------------------------------------------------------
    _separador("Etapa 1/5 — Baseline: Nearest Neighbor")
    rota_nn, custo_nn = avaliar_baseline_nn(locais_entrega, hospital_base)
    print(f"✅ Custo NN (baseline): {custo_nn:.2f}")
    print("   Rota NN: " + " → ".join(p.nome for p in rota_nn))

    # ------------------------------------------------------------------
    # 3. Gerar população inicial híbrida e executar GA com restrições VRP
    # ------------------------------------------------------------------
    _separador("Etapa 2/5 — Algoritmo Genético (Giant Tour com restrições VRP)")
    n_nn = max(1, int(TAMANHO_POPULACAO * PROPORCAO_NN))
    n_aleatorio = TAMANHO_POPULACAO - n_nn

    pop_nn = gerar_populacao_nearest_neighbor(locais_entrega, hospital_base, n_nn)
    pop_aleatoria = gerar_populacao_aleatoria(locais_entrega, n_aleatorio)
    populacao_inicial = pop_nn + pop_aleatoria

    print(f"🧬 População inicial: {len(populacao_inicial)} rotas "
          f"({n_nn} Nearest Neighbor + {n_aleatorio} aleatórias)")
    print(f"🔄 Evoluindo (máx. {N_GERACOES} gerações, parada por convergência após {PACIENCIA} sem melhora)...\n")

    melhor_rota_ga, custo_ga, historico, _ = executar_algoritmo_genetico(
        locais_entrega=locais_entrega,
        hospital_base=hospital_base,
        populacao_inicial=populacao_inicial,
        n_geracoes=N_GERACOES,
        probabilidade_mutacao=PROBABILIDADE_MUTACAO,
        paciencia=PACIENCIA,
        capacidade_veiculo=CAPACIDADE_VEICULO,
        autonomia_veiculo=AUTONOMIA_VEICULO,
        verbose=True,
    )

    print(f"\n✅ Melhor custo GA (giant tour com penalidades): {custo_ga:.2f}")

    # ------------------------------------------------------------------
    # 4. Refinamento: Two Opt Inversion
    # ------------------------------------------------------------------
    _separador("Etapa 3/5 — Refinamento: Two Opt Inversion")
    melhor_rota_final, custo_final = two_opt_inversion(
        melhor_rota_ga, hospital_base, verbose=True
    )
    print(f"\n✅ Custo final (GA + Two Opt): {custo_final:.2f}")
    if custo_ga > 0:
        melhoria_two_opt = custo_ga - custo_final
        print(f"📉 Melhoria Two Opt: {melhoria_two_opt:.2f} ({melhoria_two_opt/custo_ga*100:.1f}%)")

    # ------------------------------------------------------------------
    # 5. VRP Split: particionamento em rotas por veículo
    # ------------------------------------------------------------------
    _separador("Etapa 4/5 — VRP Split: Particionamento por Veículo")
    rotas_vrp = dividir_rotas_vrp(
        giant_tour=melhor_rota_final,
        hospital_base=hospital_base,
        capacidade_veiculo=CAPACIDADE_VEICULO,
        autonomia_veiculo=AUTONOMIA_VEICULO,
        n_veiculos=N_VEICULOS,
    )
    resumo_vrp = resumo_restricoes_vrp(
        rotas_vrp, hospital_base, CAPACIDADE_VEICULO, AUTONOMIA_VEICULO
    )

    emoji_prioridade = {1: "🔴", 2: "🟡", 3: "🟢"}
    print(f"\n🚐 Solução VRP: {len(rotas_vrp)} veículo(s) necessário(s)\n")
    for v in resumo_vrp:
        cap_ok = "✅" if v["capacidade_ok"] else "❌"
        aut_ok = "✅" if v["autonomia_ok"] else "❌"
        print(f"  Veículo {v['veiculo']}:")
        print(f"    Pontos       : {v['n_pontos']}")
        print(f"    Carga total  : {v['peso_total']} / {v['capacidade_veiculo']} un.  {cap_ok}")
        print(f"    Distância    : {v['distancia_pixels']:.0f} / {v['autonomia_veiculo']} px  {aut_ok}")
        rota_v = rotas_vrp[v["veiculo"] - 1]
        rota_str = hospital_base.nome
        for ponto in rota_v:
            prio = emoji_prioridade.get(ponto.prioridade, "⚪")
            rota_str += f" → {prio} {ponto.nome}"
        rota_str += f" → {hospital_base.nome}"
        print(f"    Rota         : {rota_str}\n")

    # ------------------------------------------------------------------
    # 6. Gerar relatório estruturado
    # ------------------------------------------------------------------
    _separador("Etapa 5/5 — Relatório e Integração com LLM")
    relatorio = gerar_relatorio_rota(
        rota_otimizada=melhor_rota_final,
        hospital_base=hospital_base,
        custo_otimizado=custo_final,
        historico_custos=historico,
        rota_baseline_nn=rota_nn,
        custo_baseline_nn=custo_nn,
        rotas_vrp=rotas_vrp,
        resumo_vrp=resumo_vrp,
    )

    print("\n📊 Resumo da rota otimizada:")
    r = relatorio["resumo"]
    print(f"   Pontos de entrega    : {r['total_pontos_entrega']}")
    print(f"   Distância total      : {r['distancia_total_pixels']:.0f} unidades")
    print(f"   Tempo total estimado : {r['tempo_total_estimado_min']:.0f} minutos")
    print(f"   Custo otimizado      : {r['custo_otimizado']:.2f}")

    comp = relatorio["comparacao_baseline_nn"]
    if comp["economia_percentual"] is not None:
        sinal = "✅" if comp["ga_superou_nn"] else "⚠️"
        print(f"\n{sinal} GA superou NN: {comp['ga_superou_nn']} "
              f"| Economia: {comp['economia_percentual']}%")

    print("\n🗺️  Rota final (giant tour):")
    print(f"   {hospital_base.nome}", end="")
    for ponto in melhor_rota_final:
        prio = emoji_prioridade.get(ponto.prioridade, "⚪")
        print(f" → {prio} {ponto.nome}", end="")
    print(f" → {hospital_base.nome}")

    # ------------------------------------------------------------------
    # 7. Integração com LLM
    # ------------------------------------------------------------------
    if _usar_llm():
        print("\n\n🤖 Gerando instruções operacionais via LLM...")
        prompt_instrucoes = prompt_instrucoes_operacionais(relatorio)
        instrucoes = _chamar_llm_seguro(prompt_instrucoes, "instruções operacionais")
        _separador("Instruções para a Equipe de Entrega")
        print(instrucoes)

        print("\n\n📈 Gerando relatório gerencial via LLM...")
        prompt_rel = prompt_relatorio_gerencial(relatorio)
        relatorio_gerencial = _chamar_llm_seguro(prompt_rel, "relatório gerencial")
        _separador("Relatório Gerencial")
        print(relatorio_gerencial)

        # ------------------------------------------------------------------
        # 8. Modo interativo de perguntas
        # ------------------------------------------------------------------
        _separador("Modo Interativo — Perguntas sobre a Rota")
        print("Digite sua pergunta sobre a rota de entregas (ou 'sair' para encerrar):\n")
        while True:
            try:
                pergunta = input("❓ Pergunta: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not pergunta or pergunta.lower() in ("sair", "exit", "quit"):
                break

            prompt_qa = prompt_pergunta_linguagem_natural(relatorio, pergunta)
            resposta = _chamar_llm_seguro(prompt_qa, "resposta Q&A")
            print(f"\n💬 Resposta:\n{resposta}\n")
    else:
        print("\nℹ️  LLM não habilitada. Para ativar, configure USE_LLM=true e a chave de API.")
        print("   Bash: export USE_LLM=true && export OPENAI_API_KEY=sk-... && python -m  main")
        print("   PowerShell: $env:USE_LLM='true'; $env:OPENAI_API_KEY='sk-...'; python -m  main")
        print("\n   Os prompts foram gerados e estão prontos para uso:")
        print("   - prompt_instrucoes_operacionais(relatorio)")
        print("   - prompt_relatorio_gerencial(relatorio)")
        print("   - prompt_pergunta_linguagem_natural(relatorio, pergunta)")

    # Salva relatório em JSON para referência
    output_path = os.path.join(os.path.dirname(__file__), "relatorio_rota.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Relatório salvo em: {output_path}")
    _separador()


if __name__ == "__main__":
    main()
