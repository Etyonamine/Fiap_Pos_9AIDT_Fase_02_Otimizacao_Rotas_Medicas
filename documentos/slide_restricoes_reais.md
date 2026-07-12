# Restrições Reais Incorporadas ao Modelo

---

## Objetivo

Demonstrar que o `medical_route_optimizer` não resolve um TSP genérico,
mas sim um **VRP com restrições operacionais reais** da logística médica,
tornando cada solução gerada concretamente viável em campo.

---

## Restrições Modeladas

| Restrição | Como foi modelada no código | Onde atua | Unidade |
| --- | --- | --- | --- |
| **Prioridade de Entrega** | `penalidade += (posição / prioridade) × fator_penalidade` — pontos críticos atendidos tarde acumulam maior penalidade | `fitness_calculator.py → calcular_custo_rota` | 1 = Alta · 2 = Média · 3 = Baixa |
| **Capacidade de Carga** | Se `soma(peso) > capacidade_veiculo`, aplica `(excesso / capacidade) × fator_penalidade_capacidade` | `fitness_calculator.py → calcular_custo_rota` | kg por veículo |
| **Autonomia do Veículo** | Se `distancia_total > autonomia_veiculo`, aplica `excesso_km × fator_penalidade_autonomia` | `fitness_calculator.py → calcular_custo_rota` | km por ciclo |
| **Múltiplos Veículos (VRP)** | Giant tour dividido em sub-rotas por `dividir_rotas_vrp()` — nova rota abre quando capacidade ou autonomia seriam violadas | `vrp_split.py → dividir_rotas_vrp` | Nº de veículos (configurável) |
| **Janelas de Tempo** | `tempo_atendimento` (minutos) cadastrado em cada `PontoEntrega`; prioridade alta favorece atendimento precoce via penalidade de posição | `delivery_points.py → PontoEntrega.tempo_atendimento` | minutos no local |

---

## Como as Restrições Interagem no Pipeline

```text
População Inicial (NN + Aleatória)
        ↓
  GA — giant tour penalizado
  (penalidade prioridade + capacidade + autonomia)
        ↓
  VRP Split — divide giant tour por veículo
  (greedy split: abre novo veículo ao violar capacidade ou autonomia)
        ↓
  Two-Opt — refina cada sub-rota individualmente
  (mesmos parâmetros de custo do GA)
        ↓
  Resultado: rotas válidas por veículo ✔
```

---

## Parâmetros Configuráveis (Interface Streamlit)

| Parâmetro | Variável no código | Padrão |
| --- | --- | --- |
| Nº de veículos | `n_veiculos` | configurável |
| Capacidade máxima | `capacidade_veiculo` | kg |
| Autonomia máxima | `autonomia_veiculo` | km |
| Peso da penalidade de prioridade | `fator_penalidade` | `2.0` |
| Peso da penalidade de capacidade | `fator_penalidade_capacidade` | `5.0` |
| Peso da penalidade de autonomia | `fator_penalidade_autonomia` | `1.5` |

---

## Exemplo Real do Dataset

```text
Hospital Base → coords (512, 317) | origem e retorno de todas as rotas

UBS Vila Nova             → prioridade 1 (Alta)  | peso: 2.0 kg
Posto Saúde Centro        → prioridade 1 (Alta)  | peso: 3.0 kg
Paciente - Rua das Flores → prioridade 1 (Alta)  | peso: 1.0 kg
Farmácia Popular Norte    → prioridade 3 (Baixa) | peso: 4.0 kg
```

> Pontos de prioridade Alta são naturalmente puxados para o início das rotas
> pela função de custo — sem necessidade de regras explícitas de ordenação.

---

## Conclusão do Slide

As restrições **não são filtros pós-processamento** — elas são parte integral
da **função de fitness** do Algoritmo Genético.
O GA aprende a construir rotas que respeitam carga, distância e urgência
**ao mesmo tempo**, durante a própria evolução.

---

Tech Challenge — Fase 2 | FIAP Pós-Tech AIDT | Julho 2026
