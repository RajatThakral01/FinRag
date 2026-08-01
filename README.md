# FinRAG — Financial Statement Intelligence Terminal

> A production-grade Retrieval-Augmented Generation system for institutional-quality 10-K analysis, built on LangGraph, FastAPI, ChromaDB, and a custom React terminal interface.

---

## Table of Contents

1. [Overview](#overview)
2. [What It Does](#what-it-does)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Data Corpus](#data-corpus)
6. [Ingestion Pipeline](#ingestion-pipeline)
7. [RAG Graph — Node by Node](#rag-graph--node-by-node)
8. [Hybrid Retrieval (BM25 + Vector + RRF)](#hybrid-retrieval-bm25--vector--rrf)
9. [Semantic Retrieval Cache](#semantic-retrieval-cache)
10. [Multi-Turn Session Memory](#multi-turn-session-memory)
11. [REST API Reference](#rest-api-reference)
12. [Frontend — Terminal Interface](#frontend--terminal-interface)
13. [Model Configuration](#model-configuration)
14. [Project Structure](#project-structure)
15. [Running Locally](#running-locally)
16. [Environment Variables](#environment-variables)
17. [Known Limitations](#known-limitations)
18. [Example Queries](#example-queries)

---

## Overview

FinRAG is a research-grade financial intelligence terminal that lets analysts query natural-language financial questions against the **2024 Annual Reports (10-K filings)** of nine major technology companies. Every answer is grounded in exact filing excerpts, verified for hallucinations before it reaches the user, and traceable to a specific chunk of the source document.

The system is designed around three hard correctness constraints:

- **No answer leaves the pipeline unverified** — a dedicated hallucination-check node runs on every response, cache-hit or not, before any text is returned.
- **Answers are never streamed** — streaming was explicitly rejected because it would expose unverified text before the hallucination check completes. All responses are blocking JSON.
- **Context resolution is pre-graph** — follow-up questions (e.g. *"What about their R&D?"*) are rewritten into standalone questions before they enter the graph. Every graph node always sees a complete, self-contained question.

---

## What It Does

| Capability | Detail |
|---|---|
| **Natural language querying** | Ask questions in plain English about revenues, margins, cash flows, R&D, risk factors, and more |
| **Multi-company comparison** | Compare metrics across any subset of the 9 companies in a single query |
| **Conversational follow-ups** | Follow-up questions automatically resolve pronouns and implicit references to prior turns |
| **Hallucination detection** | Every figure in every answer is verified against source chunks before the response is returned |
| **Semantic answer caching** | Paraphrased repeats of prior questions skip retrieval but always re-run generation and verification |
| **Math on extracted numbers** | Calculator node extracts numeric operands from document chunks and runs deterministic Python arithmetic — LLMs never do the math |
| **Full excerpt inspection** | Every citation card links to the full raw text of the source chunk via a slide-in side panel |
| **Honest failure mode** | When the pipeline cannot ground an answer, it says so explicitly rather than hallucinating a plausible number |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              FinRAG React Terminal  (port 3000)                 │
│                                                                 │
│   Session Sidebar  │  Chat Interface  │  Chunk Preview Panel   │
└──────────────────────────┬──────────────────────────────────────┘
                           │  REST API (JSON, blocking)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI  ·  api.py  (port 8000)                    │
│                                                                 │
│  POST /sessions/{id}/query  ──►  run_session_query()           │
│  GET  /chunks/{chunk_id}    ──►  ChromaDB direct lookup        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Pre-Graph Layer       │
              │  context_resolver.py    │  ← rewrites follow-ups
              │  retrieval_cache.py     │  ← composite cache check
              └────────────┬────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 LangGraph State Machine                         │
│                                                                 │
│  [router] ──► [cache_lookup] ──┬── HIT ──► [generate/calc]    │
│                                └── MISS ─► [retrieve]          │
│                                                [grade]          │
│                                           yes ──► [generate]   │
│                                           no  ──► [rewrite]    │
│                                                      (loop)     │
│                                    [hallucination_check]        │
│                               grounded ──► END                  │
│                               ungrounded ──► retry/failure      │
└─────────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ChromaDB           BM25 Index       SQLite
  (vector search)   (sparse search)  (sessions +
  all-mpnet-base                      turns +
                                      cache)
```

---

## Tech Stack

### Backend

| Layer | Technology |
|---|---|
| **Graph Orchestration** | LangGraph `StateGraph` — stateful, cyclic, multi-node |
| **LLM Provider** | Groq API (`ChatGroq`) — 8B and 70B Llama 3 models |
| **Embeddings** | `sentence-transformers/all-mpnet-base-v2` via `langchain-huggingface` |
| **Vector Store** | ChromaDB (`langchain-chroma`) — persistent on disk |
| **Sparse Index** | BM25Okapi (`rank_bm25`) — pickled, build-once |
| **PDF Extraction** | Docling — table structure detection ON, OCR OFF |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` (prose) + custom row-aware chunker (tables) |
| **Token Counting** | `tiktoken` (cl100k_base) |
| **Session Storage** | SQLite (WAL mode) — sessions, turns, retrieval cache |
| **API Layer** | FastAPI + Uvicorn |
| **Language** | Python 3.12+ |

### Frontend

| Layer | Technology |
|---|---|
| **Framework** | React 18 + TypeScript (Vite) |
| **Styling** | Vanilla CSS Modules — zero utility-class frameworks |
| **Icons** | Lucide React |
| **HTTP Client** | Native `fetch` with custom typed wrappers |
| **State** | React Context (`SessionContext`) + component-local `useState` |

---

## Data Corpus

**3,063 chunks** across **9 technology companies**, sourced from their 2024 Annual 10-K filings:

| Company | Ticker | Chunks |
|---|---|---:|
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

Each chunk carries rich structured metadata: `company`, `ticker`, `year`, `item_number`, `section_name`, `chunk_type` (`TABLE` / `PROSE`), `table_name`, `chunk_id`, `parent_chunk_id`, `page_start`, and `block_idx`.

---

## Ingestion Pipeline

The pipeline runs once per corpus update and produces the `chroma_db/` and `bm25_index.pkl` artifacts consumed by the live system.

```
PDFs  ──►  text_extractor.py (Docling)
               │
               ▼
          extracted_text/*.md
               │
         ┌─────▼──────────────────────────────────────────────┐
         │  ingestion/                                         │
         │  ├── line_classifier.py   section / table / prose  │
         │  ├── chunker.py           prose (450 tok) + tables  │
         │  ├── metadata_tagger.py   attach all metadata       │
         │  ├── parent_linker.py     prose↔table links         │
         │  └── embed_and_store.py   ChromaDB + BM25 index     │
         └────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Table chunking** never splits a row mid-number. The column header row is prepended to every table chunk so the embedding model always sees column context alongside the data.
- **Boilerplate filtering** — chunks without a resolved `item_number` (SEC Items 1–16) are excluded before embedding, removing cover pages, TOCs, and signature blocks.
- **BM25 tokenizer** expands abbreviations before punctuation stripping (`R&D` → `research and development`, `SG&A` → `selling general and administrative`) so ampersands don't collapse into near-zero-weight single-letter tokens.

---

## RAG Graph — Node by Node

All node logic lives in `graph/nodes.py`. The graph is built in `graph/graph.py` via `build_graph()`.

### Graph State (`graph/state.py`)

```python
class GraphState(TypedDict):
    question:             str
    rewritten_question:   str
    route:                str              # "retrieve" | "calculate" | "direct"
    companies_mentioned:  List[str]
    metric_category:      str              # controlled vocabulary (see Cache section)
    retrieved_chunks:     List[str]
    chunk_sources:        List[dict]
    relevant:             str              # "yes" | "no"
    answer:               str
    grounded:             str             # "grounded" | "not_grounded"
    retry_count:          int
    final_answer:         str
    cache_hit:            bool
    error_message:        Optional[str]
    conversation_context: Optional[str]   # last 2-3 turns, for tone only — never for facts
```

### Nodes

| Node | Model | Purpose |
|---|---|---|
| `router` | `llama-3.1-8b-instant` | Classify route + extract companies + classify `metric_category` — **one merged LLM call** |
| `cache_lookup` | *(embedding only)* | Composite-key semantic cache check; sets `cache_hit` |
| `retrieve` | *(no LLM)* | Hybrid BM25 + Vector RRF search per company; skipped on cache hit |
| `grade` | `llama-3.1-8b-instant` | Per-company relevance check; skipped on cache hit |
| `rewrite` | `llama-3.1-8b-instant` | Reformulate failed queries using standard financial terminology |
| `generate` | `llama-3.3-70b-versatile` | Synthesize grounded answer from labeled context chunks |
| `calculator` | `llama-3.1-8b-instant` | Extract `{operation, values}` → Python `compute()` — **LLM never does the math** |
| `direct_answer` | `llama-3.3-70b-versatile` | Answer general finance concepts (no document lookup) |
| `hallucination_check` | `llama-3.1-8b-instant` | Verify every figure traces to a source chunk — **always runs, cache hit or miss** |
| `grade_exhausted_warning` | *(no LLM)* | Write low-confidence warning to `error_message` |
| `hallucination_exhausted` | *(no LLM)* | Write honest-failure message to `final_answer` |

### Edge Routing

```
router
  ├─ [route == "direct"] ────────────────────────────► direct_answer ──► END
  └─ [route == "retrieve" | "calculate"] ──► cache_lookup
                                                  │
                                        ┌─── HIT ─┴─── MISS ───┐
                                        ▼                       ▼
                                  generate/calc              retrieve
                                        │                    grade
                                        │              ┌─ yes: generate/calc
                                        │              ├─ no:  rewrite ──► retrieve (loop)
                                        │              └─ exhausted: grade_exhausted_warning
                                        │                              └──► generate/calc
                                        └──────────────────────┘
                                                    │
                                          hallucination_check
                                       ┌─ grounded:   END
                                       ├─ retry:      generate/calc (loop)
                                       └─ exhausted:  hallucination_exhausted ──► END
```

### Cache Bypass Matrix

| Stage | Cache Hit | Cache Miss |
|---|:---:|:---:|
| Router | ✅ runs | ✅ runs |
| Cache Lookup | ✅ runs | ✅ runs |
| Retrieve | ⏭ skipped | ✅ runs |
| Grade | ⏭ skipped | ✅ runs |
| Rewrite loop | ⏭ skipped | ✅ if grade fails |
| Generate / Calculator | ✅ always fresh | ✅ runs |
| Hallucination Check | ✅ always runs | ✅ always runs |

---

## Hybrid Retrieval (BM25 + Vector + RRF)

Pure vector similarity systematically under-ranks numeric tables in favour of narrative MD&A prose that mentions the same metric. BM25 corrects this by rewarding chunks where exact query terms are densely concentrated. Both rankings are merged per-company using Reciprocal Rank Fusion:

```
RRF_score(chunk) = 1/(60 + rank_vector) + 1/(60 + rank_bm25)
```

Chunks found by only one method still receive a score; no results are discarded because they didn't appear in both.

### Chunk Budget

| Query Scope | Vector k | BM25 top-k | Final k (after RRF) | Total |
|---|:---:|:---:|:---:|:---:|
| Single company | 5 | 20 | 5 | 5 |
| 2+ specific companies | 4/company | 20/company | 4/company | 8+ |
| All companies | 4/company | 20/company | 4/company | 36 |

### Cross-Company Safety

Every chunk fed to an LLM is wrapped in a company/section header (e.g. `=== APPLE INC. — Financial Statements ===`) before being placed in the prompt, preventing cross-company number contamination in multi-company queries.

---

## Semantic Retrieval Cache

The cache stores **retrieved chunks, not final answers**. This means `generate` and `hallucination_check` always run fresh — the cache never lets an unverified or potentially stale answer reach the user.

### Composite Cache Key

Three dimensions evaluated in strict order — the first two are exact-match hard filters:

1. **`route`** — exact match (`"retrieve"` / `"calculate"`)
2. **`companies_mentioned`** — exact sorted-set match
3. **`metric_category`** — exact match (controlled vocabulary, see below)
4. **Embedding similarity** >= `0.88` — distinguishes phrasing variation *within* the same company + route + metric

> **Why not pure embedding similarity?** Validated by direct testing: *"What was Apple's revenue?"* vs. *"What was Apple's operating income?"* scores **0.91 cosine similarity**. Different-metric questions for the same company reliably clear a 0.88–0.95 threshold on sentence-structure similarity alone, making dangerous cross-metric false-positive hits possible without the hard category filter.

### `metric_category` Controlled Vocabulary

Classified by the router node in the same merged LLM call as company extraction (no extra API call):

| Category | Covers |
|---|---|
| `revenue_sales` | Revenue, Net Sales, Top Line |
| `net_income_profit` | Net Income, Net Profit, Bottom Line |
| `operating_income` | Operating Income, EBIT, Operating Margin |
| `gross_profit` | Gross Profit, Gross Margin, COGS |
| `cash_flow` | Operating/Free/Investing/Financing Cash Flow |
| `assets_liabilities_equity` | Balance sheet items |
| `r_and_d` | R&D Expense |
| `s_g_and_a` | SG&A, Sales & Marketing, G&A |
| `eps` | Earnings Per Share (basic/diluted) |
| `business_description` | Business overview, product segments |
| `risk_factors` | Risk factors, legal proceedings, competition |
| `general` | Multi-metric, ambiguous — **automatic cache bypass** |

Queries classified as `general` bypass the cache entirely. A missed cache hit costs latency; a false hit on an ambiguous query risks serving a wrong financial figure. The system is biased toward the safer failure mode.

### Cache Invalidation

Fully automatic, hash-based — no TTL required because the 10-K corpus is static until re-ingested. The invalidation hash incorporates all six model config values, `TOP_K`, and the file-level `mtime` of both `chroma_db/chroma.sqlite3` and `bm25_index.pkl`. If the hash changes on process start, the entire cache table is cleared automatically.

---

## Multi-Turn Session Memory

### The Problem

A follow-up question like *"What about their R&D expense?"* is ambiguous to the graph in isolation. Without resolution, company extraction returns `["all"]`, triggering a 36-chunk retrieval across all 9 companies instead of the correct single-company lookup.

### Pre-Graph Context Resolution

A `resolve_context()` call runs **before the graph is invoked**. It reads the last 5 turns of session history from SQLite and rewrites the raw question into a self-contained, standalone question using Groq 8B at `temperature=0.0`.

```
User: "What about their R&D?"
   ↓
context_resolver.py  (sees last 5 turns)
   ↓
Resolved: "What was Apple's R&D expense in fiscal year 2024?"
   ↓
LangGraph (unchanged — always receives a complete question)
```

The graph itself is completely unchanged. Every node always sees a resolved question as if the user had typed it in full.

### Resolver Rules

- **Pronoun resolution** — *"their"*, *"the company"*, *"it"* → substitute the correct company from history
- **Metric-only follow-ups** — *"And net income?"* → attach the most recent company
- **New-company follow-ups** — *"What about Tesla?"* → expand using the *metric* from the prior turn, not the company
- **Self-contained questions** — comparisons or complete questions pass through verbatim, unchanged
- **Definition questions** — pass through unchanged

Both `raw_question` and `resolved_question` are stored in SQLite. The raw question is the debugging ground truth if the resolver ever misfires.

### Session Storage Schema

```sql
CREATE TABLE sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    title       TEXT,          -- auto-generated from first raw_question
    last_active TEXT NOT NULL
);

CREATE TABLE turns (
    turn_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(session_id),
    turn_number       INTEGER NOT NULL,
    raw_question      TEXT NOT NULL,    -- exactly what the user typed
    resolved_question TEXT NOT NULL,    -- what entered the graph
    route             TEXT,
    companies         TEXT,             -- JSON array
    final_answer      TEXT,
    error_message     TEXT,
    created_at        TEXT NOT NULL,
    UNIQUE(session_id, turn_number)
);
```

---

## REST API Reference

The FastAPI backend (`api.py`) runs on port `8000`. Interactive docs available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok"}` — frontend connectivity check |
| `POST` | `/sessions` | Create a new research session. Returns `{ "session_id": "uuid", "created_at": "..." }` |
| `GET` | `/sessions` | List all sessions ordered by most recent activity |
| `GET` | `/sessions/{session_id}/turns` | Full turn history for a session. `404` on unknown ID |
| `POST` | `/sessions/{session_id}/query` | Submit a financial query. Runs pre-graph resolution + LangGraph pipeline |
| `GET` | `/chunks/{chunk_id}` | Retrieve full excerpt text + metadata from ChromaDB for the side panel |

### Query Response Contract

```json
{
  "raw_question":          "What about their R&D?",
  "resolved_question":     "What was Apple's R&D expense in fiscal year 2024?",
  "question_was_resolved": true,
  "final_answer":          "Apple's R&D expense in fiscal year 2024 was $31.4 billion...",
  "cache_hit":             false,
  "chunk_sources": [
    {
      "chunk_id":     "aapl_2024_item8_table_08",
      "company":      "Apple Inc.",
      "ticker":       "AAPL",
      "year":         "2024",
      "section_name": "Financial Statements",
      "chunk_type":   "TABLE",
      "table_name":   "Consolidated Statements of Operations"
    }
  ],
  "error_message": null
}
```

`question_was_resolved` is computed server-side — the frontend never needs to perform this comparison itself.

### Chunk Endpoint Response

```json
{
  "chunk_id": "aapl_2024_item8_table_08",
  "text":     "| | 2024 | 2023 | 2022 |\n|---|---|---|---|\n| Research and development | $ 31,370 | ...",
  "metadata": {
    "company":      "Apple Inc.",
    "ticker":       "AAPL",
    "year":         "2024",
    "item_number":  "8",
    "section_name": "Financial Statements",
    "chunk_type":   "TABLE",
    "table_name":   "Consolidated Statements of Operations"
  }
}
```

---

## Frontend — Terminal Interface

The React frontend is a purpose-built research terminal, not a general-purpose chat UI. It is designed around the specific data structures and workflows of the FinRAG backend.

### Design System

| Token | Value | Usage |
|---|---|---|
| `--bg-app` | `#111214` | Page background |
| `--bg-surface` | `#1A1C20` | Cards, panels, input containers |
| `--border` | `#2A2B2F` | Default borders |
| `--border-strong` | `#33353B` | Emphasis borders |
| `--text-primary` | `#EDEDEE` | Body text, financial figures |
| `--text-secondary` | `#8B8D94` | Metadata, labels |
| `--accent` | `#8CA0C7` | Verified state borders, links, active states, primary CTA |

**Typography:** `Newsreader` (display headers) · `Inter` (body prose) · `JetBrains Mono` with `tabular-nums` (financial figures and metadata tags)

No secondary hues anywhere in the UI. All semantic states are communicated through border weight, border style (solid vs. dashed), icons, and labels — not color.

### Semantic States

| State | Visual Treatment |
|---|---|
| **Verified Answer** | Thin solid `#8CA0C7` accent left border + `Cache Verified` stamp when served from cache |
| **Low-Confidence Warning** | 3px dashed `#8B8D94` left border + outline `AlertTriangle` icon + `LOW CONFIDENCE` label |
| **Honest Failure** | 3px dashed `#8B8D94` left border + outline `AlertCircle` icon + `UNVERIFIED` label |

### Key Components

| Component | File | Purpose |
|---|---|---|
| `ChatContainer` | `chat/ChatContainer.tsx` | Main layout shell; owns session state and message history |
| `MessageBubble` | `chat/MessageBubble.tsx` | Renders a single assistant turn with semantic state border |
| `SourcesPanel` | `chat/SourcesPanel.tsx` | Citation index — one card per `chunk_source` with metadata tags |
| `ChunkPreviewPanel` | `chat/ChunkPreviewPanel.tsx` | Slide-in side panel (~38% width) showing full raw excerpt text |
| `ChatInput` | `chat/ChatInput.tsx` | Query input with loading state |
| `StatusIndicator` | `chat/StatusIndicator.tsx` | Live pipeline step display during query processing |
| `WarningBanner` | `chat/WarningBanner.tsx` | Low-confidence warning strip |
| `HonestFailureCard` | `chat/HonestFailureCard.tsx` | Honest-failure placeholder card |
| `ResolvedQuestionBadge` | `chat/ResolvedQuestionBadge.tsx` | Shown only when `question_was_resolved` is true |
| `Sidebar` | `sidebar/` | Session list, creation, and switching |

### Chunk Preview Panel

Clicking a citation card triggers `GET /chunks/{chunk_id}` and slides in a non-modal side panel from the right edge:

- **Desktop:** ~38% viewport width; the rest of the document remains visible alongside
- **Narrow screens:** full-width overlay
- **Close:** dedicated close button or `Escape` key
- **Loading state:** spinner while fetching
- **Error state:** explicit *"Could not load this excerpt"* message — never a blank or silently empty panel
- **Text rendering:** `Inter` prose — never monospace

---

## Model Configuration

All models are served via the Groq API. Configure in `config.py`:

| Constant | Model | Node | Rationale |
|---|---|---|---|
| `MODEL_ROUTER` | `llama-3.1-8b-instant` | `router` | Fast — route classification + company/metric extraction |
| `MODEL_GRADER` | `llama-3.1-8b-instant` | `grade` | 8B correctly handles binary relevance grading; 70B self-contradicted on this narrow task |
| `MODEL_GENERATOR` | `llama-3.3-70b-versatile` | `generate` | Main synthesis — needs full reasoning capability |
| `MODEL_HALLUC` | `llama-3.1-8b-instant` | `hallucination_check` | Fast binary verification |
| `MODEL_REWRITE` | `llama-3.1-8b-instant` | `rewrite` | Query reformulation |
| `MODEL_CALCULATOR` | `llama-3.1-8b-instant` | `calculator` | Numeric extraction only; deterministic Python does the actual arithmetic |

**Other key parameters:**

| Parameter | Value | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | Vector search + cache key similarity |
| `CHUNK_SIZE` | `450` tokens | Safely under the 512-token model limit |
| `CHUNK_OVERLAP` | `50` tokens | Prose only — tables use zero overlap |
| `TOP_K` | `5` | Chunks retrieved per company (single-company queries) |
| `MAX_RETRY` | `3` | Shared across rewrite and hallucination retry cycles |
| `CONTEXT_WINDOW` | `5` | Prior turns fed to the context resolver |
| `CACHE_SIMILARITY_THRESHOLD` | `0.88` | Cosine similarity floor for cache hits |
| `GROQ_MAX_RETRIES` | `3` | SDK-level retry/backoff on rate-limit 429 responses |

---

## Project Structure

```
RAG_Project/
│
├── api.py                        # FastAPI application — REST layer
├── config.py                     # All tunable parameters
├── requirements.txt
├── .env                          # GROQ_API_KEY (never committed)
│
├── graph/
│   ├── state.py                  # GraphState TypedDict
│   ├── nodes.py                  # All node functions + prompt templates
│   ├── edges.py                  # Conditional edge routing functions
│   └── graph.py                  # build_graph(), run_session_query()
│
├── tools/
│   ├── vectorstore.py            # ChromaDB singleton (lazy-loaded)
│   ├── bm25_index.py             # BM25 index build / cache / query
│   ├── calculator.py             # compute() — deterministic Python arithmetic
│   ├── output_parsers.py         # Defensive LLM output parsers
│   ├── company_names.py          # Ticker <-> full-name mapping
│   ├── session_store.py          # SQLite session + turn CRUD
│   ├── context_resolver.py       # Pre-graph follow-up question resolution
│   └── retrieval_cache.py        # Semantic retrieval cache + invalidation
│
├── ingestion/
│   ├── line_classifier.py        # Classify lines: section_header / table_row / prose / blank
│   ├── chunker.py                # Prose chunking (LangChain) + table chunking (custom)
│   ├── metadata_tagger.py        # Build tagged chunk dicts from extracted text
│   ├── parent_linker.py          # Link adjacent prose <-> table chunk pairs
│   ├── embed_and_store.py        # Embed all chunks → ChromaDB; build BM25 index
│   └── embedding_check.py        # One-off sanity check for embedding model
│
├── extracted_text/               # Docling-extracted markdown (one .md per company)
├── chroma_db/                    # Persistent ChromaDB store (3,063 chunks)
├── bm25_index.pkl                # Pickled BM25Okapi bundle (~8.8 MB)
├── session_data.db               # SQLite — sessions, turns, retrieval cache
│
├── text_extractor.py             # PDF → markdown via Docling (run once per PDF)
│
└── frontend/
    ├── src/
    │   ├── App.tsx
    │   ├── main.tsx
    │   ├── styles/
    │   │   └── global.css        # Design tokens + typography
    │   ├── api/                  # Typed fetch wrappers
    │   ├── context/              # SessionContext + SessionProvider
    │   └── components/
    │       ├── chat/             # All chat UI components
    │       ├── common/           # Shared UI primitives (HealthBadge, etc.)
    │       └── sidebar/          # Session list + management
    ├── package.json
    └── vite.config.ts
```

---

## Running Locally

### Prerequisites

- Python 3.12+
- Node.js 18+ and npm
- A [Groq API Key](https://console.groq.com/) (free tier available)

### 1 — Clone and configure

```bash
git clone <repo-url>
cd RAG_Project

# Create your .env file
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### 2 — Backend setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3 — Start the backend

```bash
PYTHONUNBUFFERED=1 ./venv/bin/uvicorn api:app --port 8000
```

The FastAPI server starts on `http://localhost:8000`. Verify it's healthy:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Interactive API docs are available at `http://localhost:8000/docs`.

### 4 — Frontend setup and start

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server starts on `http://localhost:3000`. Open it in your browser.

### 5 — (Optional) Rebuild the corpus

The `chroma_db/` and `bm25_index.pkl` files are pre-built and included. To add new PDFs or rebuild from scratch:

```bash
# Extract text from PDFs (requires Docling)
python text_extractor.py

# Re-ingest: chunk, embed, and store
python ingestion/embed_and_store.py
```

> **Note:** Rebuilding the corpus will automatically invalidate the retrieval cache on next startup via the hash-based invalidation mechanism.

---

## Environment Variables

| Variable | Required | Description |
|---|:---:|---|
| `GROQ_API_KEY` | Yes | Your Groq API key — obtain from [console.groq.com](https://console.groq.com) |

No other environment variables are required. All other configuration is in `config.py`.

---

## Known Limitations

| Limitation | Detail |
|---|---|
| **No temporal cache dimension** | The cache key has no fiscal-year field. Safe for now (corpus is FY2024-only), but would need revisiting before adding FY2025 filings |
| **Groq rate limits** | The free/dev tier enforces ~30 RPM and a 100K tokens/day cap. Heavy multi-turn sessions or all-company queries can exhaust the daily limit. A paid GroqCloud tier removes this ceiling |
| **`metric_category` vocabulary gaps** | No dedicated `tax_expense` category; effective-tax-rate queries fall into `general` and bypass the cache safely but without cache benefit |
| **Chunk budget competition** | At 9 companies x 4 slots = 36 chunks, prose can occasionally out-rank the optimal table chunk. `grade_node` catches this and triggers a rewrite, but the root cause (fixed budget) remains |
| **Alphabet / Google alias** | `"Google"` is an alias for Alphabet, which appears twice in the full-names list, causing two redundant retrieval calls on `["all"]` queries — no correctness impact, minor efficiency cost |

---

## Example Queries

| Type | Example |
|---|---|
| Factual — single company | *"What was Apple's total revenue in fiscal year 2024?"* |
| Factual — filing section | *"What are NVIDIA's main risk factors for 2024?"* |
| Calculation — margin | *"What was Apple's gross margin percentage in 2024?"* |
| Calculation — growth | *"What was Amazon's revenue growth rate from 2023 to 2024?"* |
| Comparative — two companies | *"Compare Apple and Microsoft operating income for 2024"* |
| Comparative — all companies | *"Which company had the highest R&D spend in 2024?"* |
| Conceptual — direct | *"What is the difference between gross profit and operating income?"* |
| Follow-up — pronoun | *(after Apple revenue)* → *"What about their R&D?"* |
| Follow-up — new company | *(after Apple margin question)* → *"What about Tesla?"* |

---

## License

Built for portfolio and financial research purposes.

> **Disclaimer:** This tool queries static 10-K filings for research and analytical purposes only. It is not financial advice. All data is sourced from publicly available SEC filings.
