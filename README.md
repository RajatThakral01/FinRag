# 📊 Financial Intelligence RAG System
### Multi-Company 10-K Analysis · Adaptive RAG · LangGraph · NVIDIA NIM · Docling

A **production-grade Retrieval-Augmented Generation (RAG) pipeline** built on 2024 Annual Report (10-K) filings from 9 major technology companies. Users ask natural language questions about financial data and receive accurate, grounded, verifiable answers — with hallucination detection, relevance grading, query rewriting, and intelligent routing all built in.

---

## 🏢 Companies Covered

| Ticker | Company | Sector |
|---|---|---|
| AAPL | Apple Inc. | Consumer Electronics / Software |
| MSFT | Microsoft Corporation | Cloud / Enterprise Software |
| AMZN | Amazon.com Inc. | E-Commerce / Cloud (AWS) |
| NVDA | NVIDIA Corporation | Semiconductors / AI Hardware |
| TSLA | Tesla Inc. | Electric Vehicles / Energy |
| META | Meta Platforms Inc. | Social Media / VR |
| GOOGL | Alphabet Inc. | Search / Cloud / Advertising |
| NFLX | Netflix Inc. | Streaming / Content |
| ADBE | Adobe Inc. | Creative / Document Software |

---

## 🏗️ System Architecture

The pipeline has two phases:

### Phase 1 — Ingestion (runs once)
```
PDF files  →  text_extractor.py (Docling)  →  extracted_text/*.md
                    ↓
         Docling converts each PDF to structured markdown
         (table structure detection on, OCR off, run once per file)
                    ↓
ingestion/line_classifier.py   — classify lines (section header / table / prose)
ingestion/chunker.py           — split into ≤450-token chunks (two-track strategy)
ingestion/metadata_tagger.py   — tag with company/ticker/year/section/chunk_type/chunk_id
ingestion/parent_linker.py     — link adjacent prose↔table chunk pairs
ingestion/embed_and_store.py   — embed (all-mpnet-base-v2) → store in ChromaDB
```

> **Why Docling?** Docling's built-in table structure detection produces well-formed markdown tables from 10-K financial statements — a critical quality advantage over character-based extraction. Each PDF is converted in its own process to avoid memory accumulation across files.

### Phase 2 — Query (every user question)
```
User Question
     ↓
[Router Node]        — classify: retrieve / calculate / direct  (8B LLM)
     ↓
[Retrieve Node]      — vector search in ChromaDB with company metadata filter
     ↓
[Grade Node]         — relevance check, per-company completeness  (70B LLM)
   ↙       ↘
[Rewrite]   [Generate / Calculator]   — answer synthesis (70B LLM) or Python math
                ↓
[Hallucination Check]  — verify every figure is traceable to source chunks  (8B LLM)
   ↙       ↘
[Final Answer]    [Retry / Honest Failure]
```

### LangGraph State
```python
class GraphState(TypedDict):
    question:             str
    rewritten_question:   str
    route:                str            # retrieve / calculate / direct
    companies_mentioned:  List[str]      # extracted company names or ["all"]
    retrieved_chunks:     List[str]
    chunk_sources:        List[dict]
    relevant:             str            # "yes" / "no"
    answer:               str
    grounded:             str            # "grounded" / "not_grounded"
    retry_count:          int
    final_answer:         str
    error_message:        Optional[str]
```

---

## 🔑 Key Design Decisions

### Two-Track Chunking Strategy
Financial documents cannot be naively split. A raw income statement table cut mid-row by a character-based splitter produces useless fragments (a number without its column header has no meaning).

| Track | Applied To | Method |
|---|---|---|
| **Track A — Prose** | MD&A, Risk Factors, Business, Notes | LangChain `RecursiveCharacterTextSplitter`, 450 tokens, 50 overlap |
| **Track B — Tables** | Financial Statements (Item 8), quantitative sections | Custom row-aware chunker — never splits a row, prepends column headers to every chunk |

Every table chunk is self-contained:
```
Apple Inc. | 2024 | Income Statement | Columns: 2024, 2023, 2022
Net income: 97,329 (2024)  96,995 (2023)  102,962 (2022)
Total revenue: 391,035 (2024)  383,285 (2023)  394,328 (2022)
```

### Per-Company Metadata Filtering
ChromaDB pre-filters by metadata before running vector search — no cross-company contamination, and you always get top-K results from the right company.

| Query Type | Retrieval Strategy | Chunks Retrieved |
|---|---|---|
| Single company | `filter={"company": "Apple Inc."}`, top-5 | 5 |
| Two companies | Separate filtered search per company, top-4 each | 8 |
| All companies | Top-4 per company × 9 companies | 36 |

### Calculator — Two-Step Design
Arithmetic never goes through an LLM. The Calculator node uses a deliberate two-step design:
1. **LLM extraction step** — identifies `{operation, values}` JSON from chunk text
2. **Python `compute()` step** — performs the actual math deterministically

Supported operations: `percent_change`, `difference`, `sum`, `average`, `ratio`, `margin`, `max`, `min`

### Right-Sized Models Per Node
| Node | Model | Why |
|---|---|---|
| Router | `meta/llama-3.1-8b-instruct` | Simple 3-way classification |
| Grade | `meta/llama-3.1-70b-instruct` | Per-company reasoning requires quality |
| Rewrite | `meta/llama-3.1-8b-instruct` | Rephrasing task |
| Generate | `meta/llama-3.1-70b-instruct` | Main answer quality |
| Hallucination Check | `meta/llama-3.1-8b-instruct` | Binary verification |
| Direct Answer | `meta/llama-3.1-70b-instruct` | User-facing quality |

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| PDF → Markdown | **Docling** (`text_extractor.py`) — table structure detection on, OCR off |
| Section Detection | Custom Python regex (SEC Item boundaries) via `ingestion/line_classifier.py` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (prose) + custom row-aware chunker (tables) |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` via `langchain-huggingface` |
| Vector Store | ChromaDB via `langchain-chroma` |
| LLM | NVIDIA NIM API (`langchain-nvidia-ai-endpoints`) |
| Orchestration | LangGraph |
| UI | Streamlit *(planned — not yet built)* |
| Language | Python 3.11+ |

---

## 📁 Project Structure

```
RAG_Project/
├── config.py                  # Central config — all model names, paths, k values
├── .env                       # API keys (never committed)
│
├── graph/
│   ├── state.py               # GraphState TypedDict + create_initial_state()
│   ├── nodes.py               # All 10 node functions + prompt templates
│   ├── edges.py               # Conditional edge routing functions
│   └── graph.py               # build_graph() + run_query()
│
├── tools/
│   ├── vectorstore.py         # ChromaDB singleton (lazy-loaded)
│   ├── calculator.py          # compute() — pure Python arithmetic
│   ├── output_parsers.py      # Defensive LLM output parsers
│   └── company_names.py       # SHORT_TO_FULL mapping + get_all_full_names()
│
├── ingestion/
│   ├── line_classifier.py     # Classify + group lines into typed blocks
│   ├── chunker.py             # Prose chunking (LangChain) + table chunking (custom)
│   ├── metadata_tagger.py     # Build tagged chunk dicts from extracted text files
│   ├── parent_linker.py       # Link adjacent prose↔table chunk pairs
│   ├── embed_and_store.py     # Embed all chunks and populate ChromaDB
│   └── embedding_check.py     # One-off sanity check for embedding model
│
├── extracted_text/            # Docling-extracted markdown (one .md per company PDF)
├── chroma_db/                 # Persistent ChromaDB store (3,063 chunks, 9 companies)
│
├── test_pipeline.py           # Manual-chain + full-graph integration tests
├── test_setup.py              # Environment setup verification
├── text_extractor.py          # PDF → markdown via Docling (run once per PDF)
└── inspect_extract.py         # Extraction inspection utility
```

---

## ⚙️ Setup

### 1. Prerequisites
```bash
python --version   # must be 3.11+
```

### 2. Clone and create virtual environment
```bash
git clone <repo-url>
cd RAG_Project
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on Docling:** `text_extractor.py` requires Docling (`pip install docling`). PDF extraction is a one-time step — the `extracted_text/*.md` files are already committed to the repo, so you do **not** need to re-run extraction unless you add new PDFs.

### 4. Set up environment variables
Create a `.env` file in the project root (never commit this):
```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```
Get a free NVIDIA NIM API key at [build.nvidia.com](https://build.nvidia.com).

### 5. Verify setup
```bash
python test_setup.py
```

### 6. Run a query
```bash
# Always run from RAG_Project/ root — never from inside graph/
python -c "
from graph.graph import run_query
result = run_query('What was Apple total revenue in fiscal year 2024?')
print(result['final_answer'])
"
```

---

## 🧪 Running Tests

```bash
# Integration tests — router, retrieve, grade, rewrite, generate, hallucination check
python test_pipeline.py

# Node-level tests (run from project root)
python -u graph/nodes.py
```

> **Important:** Always run scripts from `RAG_Project/` root. Running from inside `graph/` causes Python to misresolve the package and throws `'graph' is not a package`.

---

## 📊 ChromaDB — Current State

The ChromaDB collection `financial_10k` is pre-populated with **3,063 chunks** across all 9 companies:

| Company | Chunks |
|---|---|
| Microsoft Corporation | 471 |
| Meta Platforms Inc. | 419 |
| Tesla Inc. | 344 |
| Adobe Inc. | 340 |
| Apple Inc. | 317 |
| Alphabet Inc. | 315 |
| NVIDIA Corporation | 308 |
| Amazon.com Inc. | 278 |
| Netflix Inc. | 271 |
| **Total** | **3,063** |

---

## 📈 Development Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | PDF ingestion, chunking, embedding, ChromaDB | ✅ Complete |
| Phase 2 | Basic RAG — Retrieve + Generate | ✅ Complete |
| Phase 3 | Router node — retrieve / calculate / direct | ✅ Complete |
| Phase 4 | Grade + Rewrite retry loop | ✅ Complete |
| Phase 5 | Hallucination Check + retry/flag | ✅ Complete |
| Phase 6 | Multi-company queries (fixes written) | ⚠️ Re-validation pending |
| Phase 7 | Calculator — single-company validated | ⚠️ Multi-company re-validation pending |
| Phase 8 | Streamlit UI | ❌ Not started |
| Phase 9 | Hybrid Search — BM25 + Vector (RRF) | ❌ Confirmed plan, not yet implemented |

---

## 🔮 Planned: Hybrid Search (BM25 + Vector via RRF)

**Current limitation:** Pure vector (embedding) similarity search misses chunks that are *literally about* the right metric. In real testing, narrative MD&A paragraphs *mentioning* a metric consistently out-ranked the actual numeric table *containing* that metric's total — even after retrieval-breadth increases and multiple query rewrites.

**Planned fix:** Run two independent retrieval methods per query and merge via **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(chunk) = 1/(k + rank_in_vector_results) + 1/(k + rank_in_bm25_results)
```

- **Dense/vector search** (already built): good at semantics, synonyms, paraphrases
- **Sparse/BM25 search** (to build): good at exact term matches — rewards chunks where the query's important words are concentrated, weighted by how rare those words are across the corpus (IDF)

See `bm25_hybrid_search_addendum.md` and PRD Chapter 21 for the full confirmed plan and judgment calls.

---

## 🛡️ Failure Modes Handled

| Scenario | What Happens |
|---|---|
| Irrelevant chunks retrieved | Grade returns "no" → Rewrite reformulates → retry (up to 3 times) |
| Retries exhausted, still no relevant chunks | Generate with best available + low-confidence warning shown to user |
| Answer contains figures not in source chunks | Hallucination Check returns "not_grounded" → retry Generate |
| Hallucination retries exhausted | Honest failure message returned — the unverified answer is never shown |
| Calculation crash (division by zero, wrong operation) | `compute()` exception caught → descriptive error in answer |
| Missing company figures in multi-company calculation | Placeholder `(not found in retrieved chunks)` emitted; excluded from max/min; surfaced as caveat in answer |

---

## 📝 Configuration Reference

All tunable parameters live in `config.py`. Change values there, nowhere else.

| Parameter | Value | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Local, 512 token limit |
| `CHUNK_SIZE` | 450 tokens | Safely under model's 512 token limit |
| `CHUNK_OVERLAP` | 50 tokens | Prose only — tables use no overlap |
| `TOP_K` | 5 (single), 4/company (multi) | 36 total for all-9-company queries |
| `MAX_RETRY` | 3 | Shared across rewrite and hallucination retry cycles |
| `MODEL_ROUTER` | `meta/llama-3.1-8b-instruct` | Fast classification |
| `MODEL_GRADER` | `meta/llama-3.1-70b-instruct` | Per-company reasoning |
| `MODEL_GENERATOR` | `meta/llama-3.1-70b-instruct` | Main answer quality |
| `MODEL_HALLUC` | `meta/llama-3.1-8b-instruct` | Binary verification |
| `MODEL_REWRITE` | `meta/llama-3.1-8b-instruct` | Query reformulation |

---

## 💡 Example Questions

| Type | Question |
|---|---|
| Factual | What was Apple's total revenue in fiscal year 2024? |
| Factual | What are NVIDIA's primary risk factors for 2024? |
| Calculation | What was Amazon's revenue growth rate from 2023 to 2024? |
| Calculation | What is Apple's gross margin percentage? |
| Comparative | Compare Apple and Microsoft operating income |
| Comparative | Which company invested the most in R&D in 2024? |
| Comparative | Compare Tesla and NVIDIA gross profit margins |
| Direct | What does EBITDA stand for? |

---

## 🎓 Skills Demonstrated

| Skill | Where in This Project |
|---|---|
| Adaptive RAG pipeline | Full ingestion → grading → rewriting → generation → hallucination detection |
| LangGraph stateful agents | Multi-node graph with cycles, conditional edges, typed state |
| LLM prompt engineering | 7 distinct prompts with defensive output parsing |
| Vector database design | ChromaDB with metadata filtering, multi-company retrieval strategy |
| Embedding model selection | Local sentence-transformers, chunk/model token alignment |
| PDF processing at scale | **Docling** for structured PDF→markdown conversion across 9 large 10-K filings |
| Financial domain knowledge | SEC 10-K structure, Item numbers, table-aware chunking |
| Failure mode awareness | Grading, rewriting, hallucination detection, honest failure messages |
| Model cost optimisation | 8B for classification tasks, 70B for generation only |
| Production system thinking | Retry limits, deterministic math tool, central config, defensive parsers |

---

## 📄 License

This project is for educational and portfolio purposes.
