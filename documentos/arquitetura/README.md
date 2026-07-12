# Arquitetura — medical_route_optimizer

Este diretório concentra, em um **local único**, os diagramas arquiteturais formais do projeto.

## 1) Diagrama de Contexto (nível sistema)

```mermaid
flowchart LR
    U[Usuário / Operação Logística] --> CLI[Execução CLI main.py]
    U --> WEB[Interface Streamlit app.py]
    CLI --> MRO[medical_route_optimizer]
    WEB --> MRO
    MRO --> LLM[(Groq/OpenAI API)]
    MRO --> OUT[Relatórios JSON + Visualizações]
```

## 2) Diagrama de Containers (aplicação)

```mermaid
flowchart TD
    subgraph APP[medical_route_optimizer]
        DATA[data/\ndelivery_points.py]
        CORE[core/\nGA + NN + Two-Opt + VRP Split + Fitness]
        SVC[services/\nga_runner.py]
        REP[reports/\nroute_report.py]
        LLM[llm/\nprompts + llm_client]
        UI[app.py (Streamlit)]
        CLI[main.py (pipeline)]
        VIS[visualizacao/\nplots + maps + animacao]
    end

    DATA --> CORE
    CORE --> SVC
    CORE --> REP
    SVC --> REP
    REP --> LLM
    SVC --> VIS
    CLI --> SVC
    CLI --> LLM
    UI --> SVC
    UI --> LLM
```

## 3) Diagrama de Componentes (núcleo de otimização)

```mermaid
flowchart LR
    POP[population_helper.py] --> GA[genetic_algorithm.py]
    NN[nearest_neighbor.py] --> GA
    OPS[genetic_operator.py] --> GA
    FIT[fitness_calculator.py] --> GA
    RC[route_calculator.py] --> FIT
    GA --> SPLIT[vrp_split.py]
    SPLIT --> TWO[two_opt.py]
    TWO --> REPORT[reports/route_report.py]
```

## 4) Diagrama de Sequência (execução principal)

```mermaid
sequenceDiagram
    participant User as Usuário
    participant Entry as main.py / app.py
    participant Runner as services.ga_runner
    participant Core as core/*
    participant Report as reports.route_report
    participant LLM as llm.llm_client

    User->>Entry: Executa otimização
    Entry->>Runner: run_ga(...)
    Runner->>Core: baseline NN + GA + VRP Split + Two-Opt
    Core-->>Runner: rota otimizada + métricas
    Runner->>Report: gerar_relatorio_rota(...)
    Report-->>Runner: relatório estruturado
    Runner-->>Entry: ResultadoGA
    Entry->>LLM: prompts (opcional, USE_LLM=true)
    LLM-->>Entry: instruções / relatório / respostas Q&A
    Entry-->>User: mapas, métricas e saídas finais
```

## 5) Diagrama de Implantação (mínimo)

```mermaid
flowchart LR
    subgraph LOCAL[Ambiente local (Python 3.11 + venv)]
        CLI[CLI: python -m main]
        WEB[Web: streamlit run app.py]
        PKG[Pacote medical_route_optimizer]
        FILES[(Arquivos locais\nJSON/Notebook/README)]
    end

    CLI --> PKG
    WEB --> PKG
    PKG --> FILES
    PKG --> EXT[(API LLM externa\nGroq/OpenAI)]
```

## Escopo e objetivo

Estes diagramas cobrem o conjunto mínimo recomendado para apresentação técnica do projeto:
- contexto do sistema;
- containers/camadas da aplicação;
- componentes internos do núcleo de otimização;
- sequência de execução ponta a ponta;
- visão mínima de implantação.
