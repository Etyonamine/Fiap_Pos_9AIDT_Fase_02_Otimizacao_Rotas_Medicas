# Otimização de Rotas Médicas

Repositório do **Tech Challenge Fase 2 (FIAP Pós Tech 9AIDT)** com foco em otimização de rotas de entrega de medicamentos usando **Algoritmo Genético (GA)**, baseline **Nearest Neighbor**, refinamento **Two-Opt** e particionamento **VRP Split**.

## Overview dos artefatos

| Artefato | Caminho | Objetivo |
|---|---|---|
| Projeto principal Python | `src/medical_route_optimizer/` | Motor de otimização, relatórios, integração opcional com LLM e app Streamlit |
| Notebook de demonstração | `notebook/GA_pipeline_demo.ipynb` | Demonstração passo a passo do pipeline GA/VRP |
| Diagramas de arquitetura (formal) | `documentos/arquitetura/README.md` | Visão arquitetural consolidada em local único (contexto, containers, componentes, sequência, implantação) |
| Documentação complementar | `documentos/` | Materiais de apoio do desafio |
| Relatórios gerados (exemplo) | `src/medical_route_optimizer/relatorio_rota.json` | Saída estruturada da execução |

## Requisitos de execução

- **Python 3.11** (recomendado)
- `pip` atualizado
- Dependências em `src/medical_route_optimizer/requirements.txt`

## Como criar o ambiente virtual (venv) com Python 3.11

### Linux/macOS

```bash
cd /path/to/Fiap_Pos_9AIDT_Fase_02_Otimizacao_Rotas_Medicas
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r src/medical_route_optimizer/requirements.txt
```

### Windows (PowerShell)

```powershell
cd C:\path\to\Fiap_Pos_9AIDT_Fase_02_Otimizacao_Rotas_Medicas
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r src/medical_route_optimizer/requirements.txt
```

## Projeto `medical_route_optimizer`

### Execução via terminal (pipeline principal)

```bash
cd src/medical_route_optimizer
python -m main
```
Observação o ambiente virtual .venv311 deve estar ativo
### Execução da interface web (Streamlit)

```bash
cd src/medical_route_optimizer
streamlit run app.py
```

### Integração opcional com LLM

Defina variáveis de ambiente antes da execução:

- `USE_LLM=true`
- `LLM_PROVIDER=groq` ou `LLM_PROVIDER=openai`
- `GROQ_API_KEY` (se Groq) ou `OPENAI_API_KEY` (se OpenAI)

## Notebook Jupyter `GA_pipeline_demo`

O notebook está em:

- `notebook/GA_pipeline_demo.ipynb`

Para executar:

```bash
cd /path/to/Fiap_Pos_9AIDT_Fase_02_Otimizacao_Rotas_Medicas
jupyterlab
```

Depois, abra `notebook/GA_pipeline_demo.ipynb`.

## Testes
Deve ser executado a partir da pasta raiz do repositorio no prompt de comando do power shell com este comando abaixo:
```bash
 pytest src/medical_route_optimizer/tests -v
```
Observação o ambiente virtual .venv311 deve estar ativo
