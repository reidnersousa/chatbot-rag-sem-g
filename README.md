# RAG sem G — Chatbot de recuperação sem LLM

> Sistema de busca semântica sobre documentos PDF que retorna respostas pré-definidas ao usuário, sem uso de modelos de linguagem generativos.

---

## Sobre o projeto

O sistema implementa a etapa de **Retrieval** do RAG (Retrieval-Augmented Generation) sem a etapa de **Generation**. Em vez de gerar uma resposta com uma LLM, o sistema recupera o chunk mais relevante dos documentos e retorna uma resposta humana pré-mapeada para aquele chunk.

**Vantagens dessa abordagem:**
- Custo zero de inferência (sem chamadas a APIs de LLM)
- Respostas totalmente controladas e auditáveis
- Comportamento previsível e fácil de corrigir
- Adequado para bases de conhecimento estáveis (políticas, FAQs, regulamentos)

---

## Arquitetura

```
Fase 1 — Indexação (offline, roda uma vez)
─────────────────────────────────────────
PDFs → Extração de texto → Chunking por seção → Embeddings → chunks_metadata.json + doc_embeddings.npy

Fase 2 — Consulta (online, a cada pergunta)
────────────────────────────────────────────
Query
 ├── Embedding (paraphrase-multilingual-MiniLM-L12-v2) → top-20 semântico
 └── BM25 (rank_bm25)                                  → top-20 por termos
         ↓
      Fusão RRF (Reciprocal Rank Fusion)
         ↓
      Re-rank (cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
         ↓
      Matcher (chunk vencedor → responses.json → resposta ao cliente)

Fase 3 — Mapeamento e validação (feito uma vez por base de documentos)
───────────────────────────────────────────────────────────────────────
gerar_responses.py → preencher_responses.py → inspecionar_chunks.py → validar_responses.py
```

---

## Stack técnica

| Componente | Tecnologia | Motivo |
|---|---|---|
| Extração de PDF | `PyMuPDF (fitz)` | Rápido, preserva estrutura de texto |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Suporte nativo a PT-BR |
| Busca vetorial | `sentence-transformers util.cos_sim` | Simples, sem dependência de banco vetorial |
| Busca por termos | `BM25Okapi (rank_bm25)` | Complementa embedding para termos técnicos exatos |
| Fusão de rankings | RRF — Reciprocal Rank Fusion | Padrão da literatura, sem pesos manuais |
| Re-ranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder multilingual, lê query + chunk juntos |
| Ambiente de teste | Kaggle Notebooks | GPU gratuita para encoding |
| Persistência | `.npy` + `.json` | Sem banco vetorial por enquanto |

---

## Estrutura de pastas

```
rag-sem-g/
├── fase1.py                   # Indexação: carrega PDFs, gera chunks e embeddings
├── fase2.py                   # Consulta: busca híbrida, re-rank e matcher
├── main.py                    # Ponto de entrada para testes
│
├── gerar_responses.py         # Gera o template do responses.json
├── preencher_responses.py     # Preenche respostas via lista no código
├── inspecionar_chunks.py      # Inspeciona estado atual do responses.json
├── validar_responses.py       # Testa qualidade das respostas com queries reais
│
├── responses.example.json     # Exemplo do formato do responses.json
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── pdf/                   # PDFs de entrada (não versionado)
│
└── embedding/
    └── rag_embeddings_policies/
        ├── doc_embeddings.npy      # Matriz de embeddings (não versionado)
        ├── chunks_metadata.json    # Chunks com metadados (não versionado)
        └── responses.json          # Mapeamento chunk → resposta (não versionado)
```

---

## Como rodar

### 1. Instalar dependências

```bash
pip install pymupdf sentence-transformers rank_bm25 numpy
```

### 2. Fase 1 — Indexação

Coloque os PDFs em `data/pdf/` e rode:

```bash
python fase1.py
```

Gera `doc_embeddings.npy` e `chunks_metadata.json` em `embedding/rag_embeddings_policies/`.

### 3. Gerar template de respostas

```bash
python gerar_responses.py
```

Gera o `responses.json` com uma entrada por chunk e campo `"texto"` marcado como `[PREENCHER]`.

### 4. Preencher as respostas

Edite o dicionário `respostas_definidas` em `preencher_responses.py` com as respostas humanas aprovadas para cada chunk e rode:

```bash
python preencher_responses.py
```

Veja o exemplo de formato em `responses.example.json`.

### 5. Inspecionar e validar

```bash
python inspecionar_chunks.py   # mostra status de cada chunk por categoria
python validar_responses.py    # roda queries de teste e exibe resumo
```

### 6. Consulta

```bash
python main.py
```

---

## Como mapear respostas

O arquivo `responses.json` é o coração do sistema. Cada chave é formada por `nome_do_arquivo.pdf::numero_da_secao`.

```json
{
  "Polticas de Home Office.pdf::1": {
    "texto": "A empresa adota modelo híbrido com mínimo de 2 dias presenciais por semana. Exceções devem ser aprovadas pelo gestor e RH.",
    "categoria": "home_office"
  },
  "Poltica de Reembolsos (Viagens e Despesas).pdf::2": {
    "texto": "O limite para alimentação em viagem é de R$ 70 por dia por pessoa. Bebidas alcoólicas não são reembolsáveis.",
    "categoria": "reembolso"
  }
}
```

**Regras para escrever boas respostas:**
- Seja direto — responda a pergunta, não copie o chunk inteiro
- Use linguagem clara, como se estivesse respondendo a um colega
- Se o chunk for ambíguo, cubra os dois ângulos na resposta
- Sempre indique onde buscar mais informações (`"Consulte o RH"`, `"Veja a política completa em..."`)

---

## Métricas e scores

O sistema usa três scores em camadas:

### bi_score (embedding)

Similaridade de cosseno entre a query e o chunk. Range fixo de `0.0` a `1.0`.

| Range | Interpretação |
|---|---|
| `< 0.35` | Sem relação |
| `0.35 – 0.50` | Zona cinza |
| `0.50 – 0.70` | Boa similaridade |
| `> 0.70` | Muito alta |

### rerank_score (cross-encoder com sigmoid)

O cross-encoder produz um logit bruto normalizado via `sigmoid` para `0.0 – 1.0`.

```python
sigmoid(x) = 1 / (1 + e^(-x))
```

| Range após sigmoid | Interpretação |
|---|---|
| `< 0.50` | Modelo inseguro |
| `0.50 – 0.60` | Zona cinza |
| `> 0.60` | Confiável |

### confiança híbrida (score final)

```python
confianca = 0.3 * bi_score + 0.7 * rerank_score
```

### Três zonas de resposta

| Confiança | Comportamento |
|---|---|
| `>= 0.30` | Resposta normal |
| `0.10 – 0.30` | Resposta com aviso de incerteza + orientação para o RH |
| `< 0.10` | Fallback total |

---

## Busca híbrida — como funciona

O sistema roda dois tipos de busca em paralelo e funde os resultados com RRF antes do re-rank:

### Embedding (dense retrieval)
Converte a query em vetor e busca os chunks mais similares por similaridade de cosseno. Ótimo para paráfrases e linguagem indireta ("apoio financeiro" encontra "subsídio").

### BM25 (sparse retrieval)
Busca por frequência de termos. Ótimo para termos técnicos exatos ("VPN", "NF", "LGPD") que o embedding pode não capturar bem.

### Fusão RRF

```python
score_rrf = Σ 1 / (60 + rank)
```

Soma os rankings das duas buscas sem pesos manuais. Chunks que aparecem bem rankeados nos dois ganham score alto.

---

## Resultados da validação

Testado com 15 queries divididas em 4 categorias (home office, segurança, reembolso, fora do domínio):

| Resultado | Quantidade |
|---|---|
| Respostas corretas | 8 |
| Aviso de incerteza com chunk correto | 3 |
| Fallback correto (fora do domínio) | 4 |
| Problemas | 0 |

Os 3 casos de aviso de incerteza são queries com linguagem indireta onde o cross-encoder tem confiança baixa, mas o chunk retornado está correto. O cliente recebe a informação certa com orientação para confirmar com o RH.

---

## Limitações

- **Respostas são tão boas quanto o mapeamento manual** — se o `responses.json` estiver incompleto ou mal escrito, o sistema erra mesmo acertando o chunk
- **Chunks sem seção numerada** usam seção `"0"` como fallback — podem gerar chaves duplicadas em PDFs sem numeração
- **Modelo multilingual tem limitações em PT-BR** — queries indiretas com linguagem muito diferente do documento podem ter confiança baixa no re-ranker

---

## Roadmap

- [x] Fase 1 — indexação de PDFs com chunking por seção
- [x] Fase 2 — busca híbrida (embedding + BM25) com re-rank e matcher
- [x] Fase 3 — mapeamento de respostas e validação
- [ ] Interface web com Streamlit
- [ ] Migrar busca vetorial para FAISS
- [ ] Testar cross-encoder nativo PT-BR (`unicamp-dl/mt5-base-en-pt-msmarco-v2`)
- [ ] API REST com FastAPI
- [ ] Migrar ambiente de Kaggle para PyTorch local

---

## Referências

- [Sentence Transformers — Pretrained Models](https://www.sbert.net/docs/pretrained_models.html)
- [BM25 — rank_bm25](https://github.com/dorianbrown/rank_bm25)
- [Reciprocal Rank Fusion — Cormack et al. 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Cross-encoders for Re-ranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [mMARCO — Multilingual MS MARCO](https://github.com/unicamp-dl/mMARCO)