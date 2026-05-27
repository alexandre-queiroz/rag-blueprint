# Requirements — rag-production

> Objetivo: POC de um sistema RAG de produção com múltiplos agentes, controle de custo por token e fallback de modelo. Foco em aprendizado prático de cada camada do system design.

---

## Functional Requirements

### FR-01 — Ingestão de Documentos
O sistema deve aceitar documentos (PDF, TXT, MD) e processá-los em um pipeline de ingestão que inclui chunking, geração de embeddings e armazenamento em vector DB.

### FR-02 — Estratégia de Chunking
O pipeline de ingestão deve suportar ao menos duas estratégias de chunking configuráveis: fixed-size e hierárquico (chunk pai + filho), permitindo comparação de qualidade.

### FR-03 — Avaliação Offline do RAG
O sistema deve incluir um módulo de avaliação que, dado um conjunto de perguntas e respostas esperadas, mensure a qualidade do retrieval (faithfulness, answer relevance, context precision) antes de qualquer deploy.

### FR-04 — Classificação de Query
Ao receber uma query, o sistema deve classificá-la por complexidade (simples / média / complexa) para rotear ao modelo mais adequado.

### FR-05 — Busca no Vector DB
O sistema deve realizar busca semântica no vector DB e, quando aplicável, combinar com busca léxica (BM25) via hybrid search.

### FR-06 — LLM Gateway
Todas as chamadas a modelos LLM devem passar por um gateway central que aplica tiering de modelos, contabiliza tokens consumidos e expõe métricas de custo por request.

### FR-07 — Controle de Custo por Token
O sistema deve aplicar um orçamento de tokens por sessão. Requests que excedam o orçamento devem ser rejeitados com erro controlado antes de chamar qualquer modelo.

### FR-08 — Fallback de Modelo
O LLM Gateway deve implementar fallback automático entre providers (ex: Claude → Gemini Flash Lite → cache) com circuit breaker por provider.

### FR-09 — Cache Semântico
Queries semanticamente similares a queries anteriores devem retornar resposta cacheada sem chamar o modelo, com threshold de similaridade configurável.

### FR-10 — Observabilidade
O sistema deve registrar por request: tokens consumidos, custo estimado, latência por camada, modelo utilizado e se a resposta veio de cache.

### FR-11 — Monitoramento de Qualidade em Produção
O sistema deve amostrar uma fração configurável do tráfego real, rodar métricas RAGAS (faithfulness, answer_relevancy, context_precision) sobre as respostas geradas e emitir alerta quando qualquer métrica cair abaixo do threshold definido em config. Deve também registrar chunk hit rate por documento para identificar problemas de ingestão ou chunking.

---

## Non-Functional Requirements

| NFR | Descrição |
|-----|-----------|
| NFR-01 | Latência p95 < 3s para queries simples (modelo barato + cache hit) |
| NFR-02 | Latência p95 < 8s para queries complexas (modelo principal) |
| NFR-03 | Custo médio por query deve ser monitorável e ter alerta configurável |
| NFR-04 | O sistema deve operar com degradação graciosa quando todos os LLMs estiverem indisponíveis |
| NFR-05 | Código com tipagem estrita — sem `any` |
| NFR-06 | Cada camada deve ser testável de forma isolada |

---

## Fora de Escopo

- Autenticação e autorização de usuários finais
- Interface de usuário (UI)
- Multi-tenancy
- Fine-tuning de modelos
- Deploy em cloud (foco em execução local para o POC)

---

## Perguntas de Domínio — Respondidas

| # | Pergunta | Decisão |
|---|---|---|
| PD-01 | Qual vector DB? | **Chroma** |
| PD-02 | Framework de avaliação offline? | **RAGAS** |
| PD-03 | Cache semântico? | **Redis + embedding próprio** |
| PD-04 | Linguagem principal? | **Python** |
| PD-05 | Provider primário / fallback? | **Claude Sonnet → Gemini → GPT-4o-mini** |
