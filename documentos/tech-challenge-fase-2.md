# Tech Challenge - Fase 2

## Overview do projeto
Este projeto implementa otimização de rotas médicas para entrega de medicamentos, combinando Algoritmo Genético (GA), baseline Nearest Neighbor, refinamento Two-Opt e estratégia de particionamento VRP Split.

## Clone do repositório
```bash
git clone https://github.com/Etyonamine/Fiap_Pos_9AIDT_Fase_02_Otimizacao_Rotas_Medicas.git
```

## Orientação para uso no notebook
O notebook de demonstração está em `notebook/GA_pipeline_demo.ipynb`.

Passos sugeridos:
1. Instalar dependências em `src/medical_route_optimizer/requirements.txt`.
2. Iniciar Jupyter Lab na raiz do projeto.
3. Abrir e executar o notebook `GA_pipeline_demo.ipynb`.

## Orientação para execução com integração LLM (Groq)
Antes de executar o projeto, configure as variáveis de ambiente:

```bash
export USE_LLM=true
export LLM_PROVIDER=groq
export GROQ_API_KEY="<SUA_GROQ_API_KEY>"
```

> Segurança: não publique chaves reais em arquivos versionados. Use sempre variável de ambiente local.

## Observação sobre validade da chave
A chave de API utilizada para testes está prevista para expirar em **31/07/2026**.

## Vídeo no YouTube
https://www.youtube.com/watch?v=TCF2-otimizacao-rotas-demo
