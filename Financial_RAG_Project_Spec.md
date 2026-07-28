# Financial Intelligence RAG System — Project Specification

*Last updated: 2026-07-28 (Post Session Memory, Semantic Cache, and Groq Migration — Backend Complete)*

---

## 1. Project Overview

A production-grade Retrieval-Augmented Generation (RAG) pipeline built on **2024 Annual 10-K Reports** for **9 technology companies**: Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), NVIDIA (NVDA), Tesla (TSLA), Meta (META), Alphabet (GOOGL), Netflix (NFLX), Adobe (ADBE).

The pipeline uses **LangGraph** for orchestration with a fully stateful, cyclic graph featuring:
- Intelligent query routing (retrieve / calculate / direct)
- Per-company relevance grading
- Query rewriting with retry loop
- Hybrid retrieval (BM25 + Vector + RRF) — fully implemented
- Calculator node (LLM extraction + deterministic Python math)
- Hallucination detection with retry/honest-failure fallback
- **Multi-session conversational memory** with pre-graph context resolution — fully implemented
- **Metric-aware semantic retrieval cache** with automatic invalidation — fully implemented
- **FastAPI REST layer** exposing the pipeline for a future frontend — fully implemented

The backend (RAG pipeline + session memory + caching + API) is **complete and tested**. A modern frontend (replacing the originally-planned Streamlit UI) is the only remaining phase, deferred pending tooling availability.

---

## 2. Technology Stack & Configuration

### Core Tech Stack

| Layer | Technology |
|---|---|
| PDF → Markdown | **Docling** (`text_extractor.py`) — table structure detection ON, OCR OFF |
| Section Detection | Custom Python regex (SEC Item boundaries) via `ingestion/line_classifier.py` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (prose) + custom row-aware chunker (tables) |
| Token Counting | `tiktoken` (cl100k_base encoding) |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` via `langchain-huggingface` (used for retrieval AND cache key similarity) |
| Vector Store | ChromaDB (`chroma_db/`) via `langchain-chroma` |
| Sparse Index | BM25Okapi (`rank_bm25`), pickled to `./bm25_index.pkl` |
| **LLM Provider** | **Groq API** (`langchain-groq`, `ChatGroq`) — *migrated from NVIDIA NIM* |
| Session Storage | SQLite (`session_data.db`) — sessions, turns, and retrieval cache tables, WAL mode |
| Graph Orchestration | LangGraph (`StateGraph`) |
| API Layer | FastAPI + Uvicorn |
| UI Framework | *Deferred — Streamlit rejected; modern custom frontend planned, not yet built* |
| Language | Python 3.11+ |

> **Provider migration note:** This project originally used NVIDIA NIM. It was fully migrated to Groq as the sole LLM provider (chosen for availability and speed during development). All five model-role constants below now reference Groq model names. See Section 15 for rate-limit implications of this choice.

### `config.py` — All Tunable Parameters

```python
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")     # from .env

EMBEDDING_MODEL    = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE         = 450      # tokens (safely under model's 512 token limit)
CHUNK_OVERLAP      = 50       # tokens (prose only — tables use no overlap)
TOP_K              = 5        # chunks for single-company queries
MAX_RETRY          = 3        # shared across rewrite and hallucination retry cycles

CHROMA_PATH        = "./chroma_db"
COLLECTION_NAME    = "financial_10k"
BM25_INDEX_PATH    = "./bm25_index.pkl"

# --- LLM model roles (Groq) ---
MODEL_ROUTER       = "llama-3.1-8b-instant"        # fast — routing + company/metric extraction
MODEL_GRADER       = "llama-3.3-70b-versatile"     # per-company relevance reasoning
MODEL_GENERATOR    = "llama-3.3-70b-versatile"     # main answer generation
MODEL_HALLUC       = "llama-3.1-8b-instant"        # binary groundedness verification
MODEL_REWRITE      = "llama-3.1-8b-instant"        # query reformulation
MODEL_CALCULATOR   = "llama-3.3-70b-versatile"     # numeric extraction for calculator_node

GROQ_MAX_RETRIES   = 5        # SDK-level retry/backoff on 429 rate-limit responses

# --- Session memory ---
SESSION_DB_PATH            = "./session_data.db"
CONTEXT_WINDOW             = 5     # turns of history fed to the context resolver

# --- Semantic retrieval cache ---
CACHE_SIMILARITY_THRESHOLD = 0.88  # cosine similarity floor, applied only WITHIN a matched metric_category
```

### Key Dependencies (`requirements.txt`)

```
pymupdf==1.24.0
langchain==0.2.0
langchain-core==0.2.28
langchain-community==0.2.0
langchain-groq==0.1.9              # pinned — newer versions require langchain-core >= 1.4.0
langchain-huggingface==0.0.3
langchain-chroma==0.1.4
chromadb==0.5.0
sentence-transformers==3.0.0
langgraph==0.1.19
fastapi
uvicorn
python-dotenv==1.0.0
tiktoken==0.7.0
rank_bm25                          # BM25Okapi for sparse retrieval
```

*(Removed: `langchain-nvidia-ai-endpoints`, `streamlit`)*

---

## 3. Project File Structure

```
RAG_Project/
├── config.py                  # Central config — model names, paths, k values, cache/session settings
├── .env                       # API keys (GROQ_API_KEY) — never committed
├── api.py                     # FastAPI application — REST layer over the graph
│
├── graph/
│   ├── state.py               # GraphState TypedDict + create_initial_state()
│   ├── nodes.py                # All node functions + all prompt templates
│   ├── edges.py                # Conditional edge routing functions
│   └── graph.py                # build_graph(), run_query(), run_session_query()
│
├── tools/
│   ├── vectorstore.py          # ChromaDB singleton getter (lazy-loaded)
│   ├── bm25_index.py           # BM25 index build/cache/query — full implementation
│   ├── calculator.py           # compute() — pure Python arithmetic (8 operations)
│   ├── output_parsers.py       # Defensive LLM output parsers
│   ├── company_names.py        # SHORT_TO_FULL mapping + get_all_full_names()
│   ├── session_store.py        # SQLite session/turn CRUD — NEW
│   ├── context_resolver.py     # Pre-graph follow-up question resolution — NEW
│   └── retrieval_cache.py      # Semantic retrieval cache + invalidation — NEW
│
├── ingestion/
│   ├── line_classifier.py      # Classify lines (section_header/table_row/prose/blank)
│   ├── chunker.py               # Prose chunking (LangChain) + table chunking (custom)
│   ├── metadata_tagger.py       # Build tagged chunk dicts from extracted text files
│   ├── parent_linker.py         # Link adjacent prose<->table chunk pairs
│   ├── embed_and_store.py       # Embed all chunks and populate ChromaDB
│   └── embedding_check.py       # One-off sanity check for embedding model
│
├── extracted_text/             # Docling-extracted markdown (one .md per company PDF)
├── chroma_db/                  # Persistent ChromaDB store (3,063 chunks, 9 companies)
├── bm25_index.pkl              # Pickled BM25Okapi bundle (~8.8 MB, build-once cache)
├── session_data.db             # SQLite — sessions, turns, retrieval_cache, cache_metadata tables
│
├── text_extractor.py           # PDF -> markdown via Docling (run once per PDF)
├── inspect_extract.py           # Extraction inspection utility
└── requirements.txt
```

---

## 4. Data & Ingestion Pipeline

*(Unchanged from original design — Phases 1–6 below are stable and complete.)*

### Phase 1: PDF Extraction (`text_extractor.py`)
Docling with table structure detection ON, OCR OFF. One `.md` file per company in `extracted_text/`.

### Phase 2: Line Classification (`ingestion/line_classifier.py`)
Classifies lines into `section_header` / `table_row` / `prose` / `blank`, using SEC Item regex matching and a `STANDARD_ITEM_TITLES` dict (Items 1–16).

### Phase 3: Two-Track Chunking (`ingestion/chunker.py`)
- **Track A — Prose**: LangChain `RecursiveCharacterTextSplitter`, 450 tokens, 50 overlap, `cl100k_base`.
- **Track B — Tables**: custom row-aware chunker; never splits a row mid-number; prepends column header row to every chunk.

### Phase 4: Metadata Tagging (`ingestion/metadata_tagger.py`)
Each chunk tagged with `company`, `ticker`, `year`, `item_number`, `section_name`, `chunk_type`, `table_name`, `chunk_id`, `parent_chunk_id`, `page_start`, `block_idx`.

Company slug → full name mapping (`COMPANY_MAP`):
```python
{
    "apple": ("Apple Inc.", "AAPL"), "microsoft": ("Microsoft Corporation", "MSFT"),
    "amazon": ("Amazon.com Inc.", "AMZN"), "nvidia": ("NVIDIA Corporation", "NVDA"),
    "tesla": ("Tesla Inc.", "TSLA"), "meta": ("Meta Platforms Inc.", "META"),
    "alphabet": ("Alphabet Inc.", "GOOGL"), "netflix": ("Netflix Inc.", "NFLX"),
    "adobe": ("Adobe Inc.", "ADBE"),
}
```

### Phase 5: Parent Linking (`ingestion/parent_linker.py`)
Links adjacent prose↔table blocks within the same section via `parent_chunk_id`.

### Phase 6: Boilerplate Filtering + Embedding (`ingestion/embed_and_store.py`)
Chunks without an `item_number` excluded. Embedded via `HuggingFaceEmbeddings`, persisted to `./chroma_db`, collection `financial_10k`.

### ChromaDB — Current Corpus State

**3,063 chunks** across 9 companies:

| Company | Ticker | Chunks |
|---|---|---|
| Microsoft Corporation | MSFT | 471 |
| Meta Platforms Inc. | META | 419 |
| Tesla Inc. | TSLA | 344 |
| Adobe Inc. | ADBE | 340 |
| Apple Inc. | AAPL | 317 |
| Alphabet Inc. | GOOGL | 315 |
| NVIDIA Corporation | NVDA | 308 |
| Amazon.com Inc. | AMZN | 278 |
| Netflix Inc. | NFLX | 271 |
| **Total** | | **3,063** |

---

## 5. Graph State

Defined in `graph/state.py`:

```python
class GraphState(TypedDict):
    # --- original fields ---
    question:             str
    rewritten_question:   str
    route:                str
    companies_mentioned:  List[str]
    retrieved_chunks:      List[str]
    chunk_sources:         List[dict]
    relevant:              str
    answer:                str
    grounded:              str
    retry_count:           int
    final_answer:          str
    error_message:         Optional[str]

    # --- added for semantic cache ---
    metric_category:       str               # controlled-vocabulary classification (see Section 9)
    cache_hit:             bool              # True if retrieve/grade were skipped via cache

    # --- added for conversational memory ---
    conversation_context:  Optional[str]     # last 2-3 Q&A pairs, formatted; used for tone only, never facts
```

**Design note:** full conversation history is deliberately *not* stored in `GraphState`. Session history lives in SQLite (`tools/session_store.py`); the context resolver runs *before* the graph is invoked and produces a single resolved, standalone question that enters the graph exactly as if the user had typed it.

---

## 6. LangGraph Node Architecture

### All Nodes (`graph/nodes.py`)

| Node | Function | Model | Purpose |
|---|---|---|---|
| `cache_lookup` | `cache_lookup_node()` | embedding only | Composite-key semantic cache check — **NEW** |
| `router` | `router_node()` | 8B | Classify route + extract companies + classify `metric_category` (merged into one call) |
| `retrieve` | `retrieve_node()` | — | Hybrid BM25+Vector search, RRF merge per company (skipped on cache hit) |
| `grade` | `grade_node()` | 70B | Per-company relevance check (skipped on cache hit) |
| `rewrite` | `rewrite_node()` | 8B | Reformulate question with standard financial terminology |
| `generate` | `generate_node()` | 70B | Synthesize answer from labeled context chunks; now also receives `conversation_context` for tone |
| `calculator` | `calculator_node()` | 70B (`MODEL_CALCULATOR`) | Extract numbers from chunks → Python `compute()` |
| `direct_answer` | `direct_answer_node()` | 70B | Answer general finance concepts (no document lookup); no cache interaction |
| `hallucination_check` | `hallucination_check_node()` | 8B | Verify every figure is traceable to source chunks — **always runs, cache hit or miss** |
| `grade_exhausted_warning` | `grade_exhausted_warning_node()` | — | Write low-confidence warning to `error_message` |
| `hallucination_exhausted` | `hallucination_exhausted_node()` | — | Write honest failure message to `final_answer` |

### Graph Entry Point & Edge Routing (`graph/edges.py`)

Entry point: `router` node → **`cache_lookup`** (new) → `retrieve` (miss) / `generate`|`calculator` (hit)

```
router
  |--[route == "direct"]-----------> direct_answer --> END
  |--[route == "retrieve"/"calculate"] -> cache_lookup
                                            |
                              [HIT: grade="yes" previously]   [MISS]
                                    |                            |
                          generate/calculator                 retrieve
                                    |                            |
                                    |                          grade
                                    |              [yes]     [rewrite]    [exhausted]
                                    |               |            |             |
                                    |          generate/     retrieve    grade_exhausted_warning
                                    |          calculator      (^)         |-> generate/calculator
                                    |               |
                                    +---------------+
                                            |
                                   hallucination_check
                                [grounded] [retry]  [exhausted]
                                    |        |           |
                                   END  generate/   hallucination_exhausted -> END
                                        calculator
```

**Edge functions in `graph/edges.py`:**

```python
def route_after_router(state) -> str:
    return "direct" if state["route"] == "direct" else "cache_lookup"

def route_after_cache(state) -> str:
    if state["cache_hit"]:
        return "calculate" if state["route"] == "calculate" else "generate"
    return "retrieve"

def route_after_grade(state) -> str:
    if state["relevant"] == "yes":
        return "calculate" if state["route"] == "calculate" else "generate"
    if state["retry_count"] < config.MAX_RETRY:
        return "rewrite"
    return "exhausted"

def route_by_calc_type(state) -> str:
    return "calculate" if state["route"] == "calculate" else "generate"

def route_after_hallucination(state) -> str:
    if state["grounded"] == "grounded":
        return "end"
    if state["retry_count"] < config.MAX_RETRY:
        return "calculate" if state["route"] == "calculate" else "generate"
    return "exhausted"
```

**Cache bypass matrix:**

| Pipeline Stage | Cache Hit | Cache Miss |
|---|---|---|
| Router | ✅ Always runs | ✅ Always runs |
| Cache Lookup | ✅ Runs, returns hit | ✅ Runs, returns miss |
| Retrieve | ⏭ Skipped | ✅ Runs |
| Grade | ⏭ Skipped | ✅ Runs |
| Rewrite loop | ⏭ Skipped | ✅ Runs if grade="no" |
| Generate/Calculator | ✅ Always runs (fresh) | ✅ Always runs |
| Hallucination Check | ✅ Always runs | ✅ Always runs |
| Hallucination retry | ✅ Normal retry logic | ✅ Normal retry logic |

---

## 7. Retrieval Strategy — Hybrid Search

*(Unchanged — fully implemented and stable.)*

### Why Hybrid? The Pure-Vector Problem
Pure vector similarity favors narrative text over numeric tables; MD&A prose mentioning a metric can outrank the table containing the exact total. BM25 fixes this by rewarding chunks where the query's exact terms are densely concentrated.

### Architecture (`tools/bm25_index.py`)
1. One global BM25 index (corpus-wide IDF).
2. Pickle cache at `./bm25_index.pkl` (~8.8 MB); delete to force rebuild.
3. Corpus pulled from Chroma `.get()` — exact parity with the vector index.
4. Module-level lazy singleton, loaded once per process lifetime.

### Tokenizer
Abbreviation expansion (`R&D` → `research and development`, `SG&A` → `selling general and administrative`) runs *before* punctuation stripping, so ampersands don't collapse into near-zero-weight single-letter tokens.

### RRF Merge
```
RRF_score(chunk) = 1/(60 + rank_in_vector) + 1/(60 + rank_in_bm25)
```
Run per company; chunks deduplicated by `chunk_id`; a chunk found by only one method still receives a score.

### Chunk Budget per Query Type

| Query Type | Vector k | BM25 top_k | Final k (after RRF) | Total Chunks |
|---|---|---|---|---|
| Single company | 5 | 20 | 5 | 5 |
| Two+ specific companies | 4 per company | 20 per company | 4 per company | 8+ |
| All companies (`["all"]`) | 4 per company | 20 per company | 4 per company | 36 |

---

## 8. Context Formatting — Cross-Company Safety

*(Unchanged.)* Every chunk fed to an LLM is wrapped in a company/section header (`=== APPLE INC. — Financial Statements ===`) to prevent cross-company number contamination.

---

## 9. Multi-Session Conversational Memory — NEW

### The Problem
Follow-up questions ("What about their R&D expense?") arrive as bare, ambiguous strings. Without resolution, the router cannot classify them and company extraction returns `["all"]`, triggering a 36-chunk retrieval instead of the correct single-company lookup.

### Architecture: Pre-Graph Context Resolution
A `resolve_context()` step runs **before** the graph is invoked. It takes the raw question + the last `CONTEXT_WINDOW` (5) turns of session history and produces a fully self-contained, standalone question. The graph itself — every node, every prompt, every edge — is completely unchanged; it always receives a resolved question as if the user had typed it directly.

```
User: "What about their R&D expense?"
     ↓
[Context Resolver — Groq 8B, temperature=0.0]  ← sees last 5 turns of history
     ↓
Resolved: "What was Apple's R&D expense?"
     ↓
[Existing Graph — unchanged]
```

This was chosen over two alternatives:
- **Injecting full history into every prompt** — rejected: contaminates narrow, single-purpose prompts (router, grader, hallucination check) with thousands of tokens of irrelevant history, degrading their precision.
- **LangGraph checkpointing (thread-based persistence)** — rejected: checkpointing solves *resuming an interrupted execution*, not *carrying resolved context between independent single-shot invocations*. `run_query()`'s single-shot invoke model isn't a natural fit for append-style threading.

### Resolver Prompt Rules (validated via adversarial testing — see Section 14)
- Pronoun resolution ("their", "the company") → substitute the correct company from history.
- Metric-only follow-ups ("And net income?") → attach the most recent company.
- **New-company follow-ups** ("What about Tesla?") → expand using the *metric* from the most recent turn, but explicitly do **not** carry over the previous company or inject a false comparison.
- Explicit comparisons that are already self-contained → pass through **unchanged**, verbatim.
- Standalone/definition questions → pass through unchanged.
- `temperature=0.0` explicitly set on the Groq call.

### Session Storage (`tools/session_store.py`)
SQLite, WAL mode, two tables:

```sql
CREATE TABLE sessions (
    session_id    TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    title         TEXT,               -- auto-generated from first raw_question (truncated)
    last_active   TEXT NOT NULL
);
CREATE TABLE turns (
    turn_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT NOT NULL REFERENCES sessions(session_id),
    turn_number        INTEGER NOT NULL,
    raw_question       TEXT NOT NULL,   -- exactly what the user typed
    resolved_question  TEXT NOT NULL,   -- what actually entered the graph
    route              TEXT,
    companies          TEXT,            -- JSON array
    final_answer       TEXT,
    error_message       TEXT,
    created_at          TEXT NOT NULL,
    UNIQUE(session_id, turn_number)
);
```

Both `raw_question` and `resolved_question` are stored — `raw` is the debugging ground truth if the resolver ever misfires; `resolved` is what the pipeline actually processed.

### Generate Prompt Enhancement
`generate_node` receives an optional `conversation_context` (last 2–3 Q&A pairs) purely for **conversational tone** — an explicit instruction forbids using it for facts:

> *"...use this to understand the flow and adopt a natural conversational tone, but DO NOT use it for factual numbers — rely only on the provided 10-K Context for facts."*

`calculator_node` intentionally does **not** receive `conversation_context` — its output is a rigid JSON schema, and conversational history would only risk contaminating structured extraction.

---

## 10. Semantic Retrieval Cache — NEW

### The Problem
Hybrid retrieval (BM25 + vector + RRF) is not free — `["all"]` queries multiply into 9 ChromaDB calls + 9 corpus-wide BM25 scoring passes + 9 RRF merges. Repeated or paraphrased questions should reuse prior results rather than re-running full retrieval.

### Composite Cache Key
Three dimensions, evaluated in order — the first two are **exact-match hard filters**, the third is a similarity threshold applied *only within an already-matched category*:

1. **`route`** — exact match ("retrieve" / "calculate")
2. **`companies_mentioned`** — exact set match (sorted JSON array)
3. **`metric_category`** — exact match, controlled vocabulary (see below)
4. **Embedding similarity** (`CACHE_SIMILARITY_THRESHOLD = 0.88`) — distinguishes phrasing variation *within* the same company + route + metric

**Why not pure embedding similarity alone (rejected approach):** validated by direct testing — "What was Apple's revenue?" vs. "What was Apple's operating income?" scores **0.9104** cosine similarity; "Amazon's revenue" vs. "Amazon's net income" scores **0.9476**. Same-company, different-metric pairs reliably clear a 0.88–0.95 threshold on sentence-structure similarity alone, which would produce dangerous cross-metric false-positive cache hits (e.g. serving revenue chunks for a net-income question). This is why `metric_category` was added as a hard, exact-match filter rather than folded into the similarity score.

### `metric_category` Controlled Vocabulary
Classified by the same 8B model call that already does company extraction (merged into a single `query_analysis_prompt`, avoiding a third LLM call per query):

| Category | Covers |
|---|---|
| `revenue_sales` | Revenue, Net Sales, Top Line |
| `net_income_profit` | Net Income, Net Profit, Bottom Line |
| `operating_income` | Operating Income, EBIT, Operating Margin |
| `gross_profit` | Gross Profit, Gross Margin, COGS |
| `cash_flow` | Operating/Free Cash Flow, Investing, Financing |
| `assets_liabilities_equity` | Balance sheet items |
| `r_and_d` | R&D Expense |
| `s_g_and_a` | SG&A, Sales & Marketing, G&A |
| `eps` | Earnings Per Share (basic/diluted) |
| `business_description` | Business overview, product segments |
| `risk_factors` | Risk factors, legal proceedings, competition |
| `general` | Multi-metric, ambiguous, or non-fitting queries — **automatic cache bypass** |

**Safety rule:** any query classified as `general` bypasses the cache entirely (`get_cache()` and `put_cache()` both return early). A missed cache hit costs latency; a false hit on an ambiguous/multi-metric question risks serving a wrong financial figure — the system is biased toward the safer failure mode.

### What the Cache Stores
Retrieved **chunks**, not final answers:

```sql
CREATE TABLE retrieval_cache (
    cache_id           TEXT PRIMARY KEY,
    route              TEXT NOT NULL,
    companies_json     TEXT NOT NULL,
    metric_category    TEXT NOT NULL,
    question_embedding BLOB NOT NULL,
    resolved_question  TEXT NOT NULL,
    chunks_json        TEXT NOT NULL,
    sources_json       TEXT NOT NULL,
    grade_result       TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    hit_count          INTEGER DEFAULT 0
);
CREATE INDEX idx_cache_key ON retrieval_cache(route, companies_json, metric_category);
```

**Deliberate design decision:** caching chunks (not answers) means `generate`/`calculator` always runs fresh, and `hallucination_check` always verifies a newly-generated answer — the cache never lets an unverified or stale answer reach the user.

**Cache write gating:** entries are only written when `grade_result == "yes"`. Chunks that failed grading are never cached — a slightly different phrasing on a subsequent query gets a genuine fresh retrieval rather than inheriting a known-bad result.

### Cache Invalidation (Phase B3)
Fully automatic, hash-based — no manual TTL, since the 10-K corpus is static until re-ingested.

`_compute_corpus_hash()` incorporates:
- All 6 model config values (`MODEL_ROUTER`, `MODEL_GRADER`, `MODEL_GENERATOR`, `MODEL_HALLUC`, `MODEL_REWRITE`, `MODEL_CALCULATOR`) + `TOP_K`
- `chroma_db/chroma.sqlite3` file mtime (explicitly the file, not the containing directory — directory mtimes do not update on in-place file modification, which was caught and fixed during implementation)
- `bm25_index.pkl` file mtime

`_check_and_invalidate_cache()` runs on every import of `retrieval_cache.py`. If the computed hash differs from the stored value in the `cache_metadata` table, the entire `retrieval_cache` table is cleared and the new hash is stored. Verified via direct before/after testing (hash mismatch → log message → table cleared to 0 rows).

---

## 11. FastAPI Layer (`api.py`) — NEW

REST API wrapping `run_session_query()`, designed as the stable contract for a future frontend.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Returns `{"status": "ok"}` — frontend availability check |
| `POST /sessions` | Creates a new session, returns `session_id` |
| `GET /sessions` | Lists all sessions, ordered by `last_active` |
| `GET /sessions/{session_id}/turns` | Returns full turn history for a session; 404 on invalid `session_id` |
| `POST /sessions/{session_id}/query` | Primary query endpoint |

**Query endpoint response contract:**

```json
{
  "raw_question": "What about Tesla?",
  "resolved_question": "What was Tesla's net income?",
  "question_was_resolved": true,
  "final_answer": "Tesla's net income in 2024 was $7,153 million.",
  "cache_hit": false,
  "chunk_sources": [{"company": "Tesla Inc.", "year": "2024", "section": "Item 8"}],
  "error_message": null
}
```

`question_was_resolved` is computed **entirely server-side** (`raw_question.strip() != resolved_question.strip()`) — the frontend never needs to perform this comparison itself, keeping frontend logic dumb and the resolver logic centralized.

**CORS:** pinned explicitly to `http://localhost:3000` (not wildcarded) — flagged for further tightening before any real deployment.

**Streaming decision:** token-by-token streaming was evaluated and **explicitly rejected**. Streaming would expose unverified text before `hallucination_check` runs; if the check subsequently fails, the user would have already read a potentially hallucinated answer with no clean way to retract it. The API remains a blocking JSON REST call. A future enhancement (not yet built) could add an SSE/WebSocket endpoint for status updates ("Resolving context...", "Retrieving...", "Checking for hallucinations...") without compromising this safety guarantee.

---

## 12. Prompt Engineering & LLM Nodes

*(Router, Grade, Rewrite, Generate, Calculator Extraction, Hallucination Check, and Direct Answer prompts retain their original design intent from Sections 9–10 of the prior spec. Two changes:)*

- **Router / Company Extraction prompt** was merged into `query_analysis_prompt`, now returning both `companies` and `metric_category` in one JSON object, in a single LLM call (avoiding a separate third call per query). Company-extraction accuracy was regression-tested after the merge to confirm no degradation from the added classification task.
- **Generate Prompt** now optionally accepts `conversation_context`, with an explicit tone-only instruction (see Section 9).

All original parsers (`parse_route`, `parse_grade`, `parse_hallucination`, `parse_calculation`) are unchanged. `parse_companies` was renamed/expanded to `parse_query_analysis`, returning `{companies, metric_category}`.

---

## 13. Calculator Node — Two-Step Design

*(Unchanged.)* Arithmetic never goes through an LLM: LLM extraction (`calculator_extract_prompt`, now on `MODEL_CALCULATOR`) identifies `{operation, values}`; Python `compute()` performs the math deterministically. Same 8 operations, same missing-company placeholder handling as original design.

---

## 14. Testing & Validation Performed

| Component | Test | Result |
|---|---|---|
| Context resolver | 12-case adversarial suite (pronouns, new-company, comparisons, definitions, year shifts) | 12/12 passing |
| Context resolver | 3x consistency reruns of all 12 cases (checking `temperature=0.0` determinism on Groq) | 36/36 consistent |
| Semantic cache | 4 same-company/different-metric pairs (Microsoft, Alphabet, Apple, Amazon) | All correctly registered as cache **misses** post-fix |
| Semantic cache | `general` bypass (multi-metric query) | Correctly bypassed cache on both first and repeat asks |
| Semantic cache | `risk_factors` bypass/cache interaction | Confirmed cache-miss → cache-hit transition behaves correctly once resolved-question embeddings match |
| Cache invalidation | Config value change (`TOP_K`, later `MODEL_CALCULATOR` addition) | Hash mismatch detected, cache cleared, verified via row-count before/after |
| Cache invalidation | File-level vs. directory-level mtime bug | Caught during review — directory mtime doesn't update on in-place file changes; fixed to target `chroma.sqlite3` directly |
| FastAPI layer | All 4 endpoints incl. 404 on invalid `session_id` | Verified via live `curl` tests |
| Full multi-turn flow | 3-turn live session via API (base question → pronoun follow-up → new-company follow-up) | Resolved correctly at each turn, cache hit triggered appropriately |

---

## 15. Known Issues & Deferred Items

### Active / Deferred

- **No temporal (year) dimension in the cache key.** The composite key covers `route` + `companies` + `metric_category` + embedding similarity, but has no concept of fiscal year. Currently low-risk because the corpus is single-year (FY2024) with multi-year comparative columns already present within each chunk — a same-metric-different-year "collision" likely still resolves to the same, correct chunk. **This will need to be revisited before any future multi-year ingestion** (e.g. adding FY2025 filings as separate documents), where it could become a genuine cross-year contamination risk.

- **`metric_category` vocabulary gaps:** no dedicated `tax_expense` category (effective tax rate questions likely fall into `general` and safely bypass, at the cost of cache benefit); no distinction between consolidated vs. segment/geographic revenue breakdowns within `revenue_sales` (both hash to the same category — a segment-revenue cache entry could theoretically be offered for a consolidated-revenue question, though the calculator's existing "prefer consolidated" rule provides a partial downstream safeguard).

- **Groq rate limits (new constraint introduced by the provider migration):** the free/dev tier enforces both per-minute limits (~30 RPM, ~6,000–14,000 TPM for 70B models) and a **100,000 tokens/day (TPD)** cap. `GROQ_MAX_RETRIES = 5` with SDK-level backoff mitigates transient 429s from the per-minute limits, but does **not** help once the daily cap is exhausted — this was hit directly during adversarial testing. A paid GroqCloud tier removes this ceiling; worth budgeting for before heavy frontend integration testing or live demos.

- **Chunk-Budget Competition (3+ Companies):** at 9 companies × 4 slots = 36 chunks, budget competition occasionally lets a prose chunk out-compete the optimal table chunk for a given company. `grade_node` correctly rejects this and triggers a rewrite, but the root cause (fixed budget) remains. Dynamic slot allocation is an open decision.

- **Error Propagation:** `grade_exhausted_warning_node` sets `error_message` internally; the FastAPI layer now surfaces this field in the query response, but the *future frontend* still needs to actually render it as a visible low-confidence warning.

- **`get_all_full_names()` duplicate:** `"Alphabet Inc."` appears twice (once for `"Alphabet"`, once for the `"Google"` alias), causing two redundant Chroma + BM25 calls on `["all"]` queries. No correctness impact, minor efficiency waste.

### Resolved Issues (for reference)

- **Same-company/different-metric cache false positives** — root-caused via direct embedding-similarity testing (0.90–0.95 cosine similarity between different-metric questions for the same company); fixed by adding `metric_category` as a hard exact-match filter rather than relying on embedding similarity alone.
- **Directory-mtime cache invalidation bug** — caught before it caused a real incident; fixed to hash the actual `chroma.sqlite3` file.
- **`MODEL_CALCULATOR` missing from the invalidation hash** — found and fixed during the Groq migration cleanup; verified with a before/after hash-change test.
- **Tesla vs. NVIDIA percentage comparisons** — fixed by the `"difference"` operation rule for comparing two percentage-based metrics.
- **Google → Alphabet mapping** — alias fully resolves "Google" queries to the correct filing.
- **R&D and SG&A abbreviations in BM25** — handled by ordered `_ABBREV_SUBS` expansion before punctuation stripping.
- **Context resolver: new-company questions returning unresolvable fragments** (e.g. "What about Tesla?" not expanding at all) — fixed via a RULE 4 prompt addition; validated across 12 adversarial cases and 3 consistency reruns.
- **Context resolver: explicit comparisons being incorrectly rewritten** — fixed via an explicit "already self-contained" passthrough rule.

---

## 16. Frontend (Deferred)

**Decision:** Streamlit (originally planned as Phase 9) was explicitly rejected. Given the production-grade ambitions of this project, a modern, custom frontend will be built separately instead — deferred pending Opus quota availability, and intentionally decoupled from the backend via the FastAPI contract in Section 11, so frontend work can proceed purely as a consumer of a stable, already-tested API.

Design intentions for the eventual frontend (carried over from the original Phase 9 plan, to be revisited at build time):
- Session sidebar (list/create/switch sessions)
- Show `resolved_question` **only when `question_was_resolved` is true** — the cheapest safety net against an undetected resolver misfire reaching the user unnoticed
- Expandable "sources used" panel driven by `chunk_sources`
- Visible low-confidence warning when `error_message` is set
- Honest-failure display when `final_answer` originates from `hallucination_exhausted_node`
- Live status indicators (e.g. via a future SSE endpoint) rather than token streaming, per the streaming-safety decision in Section 11

---

## 17. Example Queries by Type

| Type | Example Question | Expected Route |
|---|---|---|
| Factual — single company | What was Apple's total revenue in fiscal year 2024? | retrieve |
| Factual — single company | What was NVIDIA's R&D expense in 2024? | retrieve |
| Factual — risk factors | What are Tesla's main risk factors? | retrieve |
| Calculation — single company | What was Apple's gross margin percentage? | calculate |
| Calculation — growth rate | What was Amazon's revenue growth rate from 2023 to 2024? | calculate |
| Comparative — cross-company | Compare Apple and Microsoft operating income | calculate |
| Comparative — all companies | Which company had the highest R&D spend in 2024? | calculate |
| Direct — definition | What does EBITDA stand for? | direct |
| Direct — concept | What is the difference between gross profit and operating income? | direct |
| Conversational follow-up | (after Apple revenue) "What about their R&D?" → resolved to "What was Apple's R&D expense?" | retrieve |
| Conversational new-company | (after Apple margin question) "What about Tesla?" → resolved to "What was Tesla's gross margin?" | retrieve |
