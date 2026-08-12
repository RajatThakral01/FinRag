# FinRAG — Financial Intelligence Terminal

> A Retrieval-Augmented Generation system for querying SEC 10-K filings in natural language. Built with LangGraph, FastAPI, ChromaDB, BM25, and a React terminal interface.

---

## What It Does

FinRAG lets you ask financial questions in plain English and get answers grounded in exact SEC filing excerpts — with every figure verified before it reaches you.

- **Natural language queries** — revenues, margins, cash flows, R&D, risk factors, and more
- **Multi-company comparisons** — query any subset of the 9 supported companies in one shot
- **Conversational follow-ups** — pronouns and implicit references auto-resolve to prior context
- **Hallucination detection** — every figure is verified against source chunks before the response is returned
- **Deterministic math** — a calculator node extracts numbers from documents and runs Python arithmetic; the LLM never does the math
- **Semantic answer caching** — paraphrased repeats skip retrieval but always re-run generation and verification
- **Full source inspection** — every citation links to the full raw chunk text in a slide-in panel

---

## Data Corpus

**3,063 chunks** from **9 technology companies**, sourced from their 2024 Annual 10-K filings:

| Company | Ticker | Chunks |
|---|---|---:|
| Microsoft | MSFT | 471 |
| Meta Platforms | META | 419 |
| Tesla | TSLA | 344 |
| Adobe | ADBE | 340 |
| Apple | AAPL | 317 |
| Alphabet | GOOGL | 315 |
| NVIDIA | NVDA | 308 |
| Amazon | AMZN | 278 |
| Netflix | NFLX | 271 |

---

## Architecture

```
React Terminal (port 3000)
        │  REST API (JSON, blocking)
        ▼
FastAPI  ·  api.py  (port 8000)
        │
  Pre-Graph Layer
  ├── context_resolver.py   ← rewrites follow-ups into standalone questions
  └── retrieval_cache.py    ← composite semantic cache check
        │
  LangGraph State Machine
  ├── router       → classify route + extract companies + metric category
  ├── cache_lookup → HIT: skip retrieve/grade → generate → hallucination_check
  ├── retrieve     → hybrid BM25 + Vector (RRF fusion) per company
  ├── grade        → relevance check; fails → rewrite → retrieve (loop)
  ├── generate     → 70B Llama synthesis from labeled source chunks
  ├── calculator   → extract operands → Python compute() → 70B format
  └── hallucination_check → verify every figure traces to a source chunk
        │
  ┌─────┼──────────┐
  ▼     ▼          ▼
ChromaDB  BM25 Index  SQLite
```

---

## Tech Stack

**Backend:** Python 3.12 · FastAPI · LangGraph · Groq API (Llama 3.1 8B + 3.3 70B) · ChromaDB · BM25Okapi · `sentence-transformers/all-mpnet-base-v2` · Docling · SQLite

**Frontend:** React 18 + TypeScript (Vite) · Vanilla CSS Modules · Lucide React

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
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### 2 — Backend

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --port 8000
```

Verify: `curl http://localhost:8000/health` → `{"status":"ok"}`  
Interactive docs: `http://localhost:8000/docs`

### 3 — Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### 4 — (Optional) Rebuild corpus

The `chroma_db/` and `bm25_index.pkl` are pre-built. To add PDFs or rebuild:

```bash
python ingestion/text_extractor.py  # PDF → markdown via Docling
python ingestion/embed_and_store.py # chunk, embed, and store
```

> Rebuilding automatically invalidates the retrieval cache on next startup.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Answers are never streamed** | Streaming would expose unverified text before the hallucination check completes |
| **Context resolution is pre-graph** | Every graph node always sees a self-contained, standalone question |
| **Cache stores chunks, not answers** | `generate` and `hallucination_check` always run fresh — no stale answers |
| **Composite cache key** | `route` + `companies` + `metric_category` + `cosine ≥ 0.88` — pure embedding similarity alone gave false hits across different metrics |
| **BM25 + Vector RRF** | Pure vector search under-ranks numeric tables; BM25 corrects this by rewarding exact term density |
| **Boilerplate filtering** | Chunks without a resolved SEC Item number (cover pages, TOCs) are excluded before embedding |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Backend connectivity check |
| `POST` | `/sessions` | Create a new research session |
| `GET` | `/sessions` | List all sessions |
| `GET` | `/sessions/{id}/turns` | Full turn history |
| `POST` | `/sessions/{id}/query` | Submit a financial query |
| `PUT` | `/sessions/{id}/title` | Rename a session |
| `GET` | `/chunks/{chunk_id}` | Fetch raw chunk text + metadata |

---

## Example Queries

```
"What was Apple's total revenue in fiscal year 2024?"
"Compare Apple and Microsoft operating income for 2024"
"Which company had the highest R&D spend in 2024?"
"What was Amazon's revenue growth rate from 2023 to 2024?"
"What are NVIDIA's main risk factors?"

# Follow-ups work too:
"What was Apple's revenue?"  →  "What about their R&D?"  →  "What about Tesla?"
```

---

## Project Structure

```
RAG_Project/
├── api.py                  # FastAPI REST layer
├── config.py               # All tunable parameters
├── graph/                  # LangGraph state machine
│   ├── state.py            # GraphState TypedDict
│   ├── nodes.py            # All node functions + prompts
│   ├── edges.py            # Conditional edge routing
│   └── graph.py            # build_graph(), run_session_query()
├── tools/                  # Shared utilities
│   ├── vectorstore.py      # ChromaDB singleton
│   ├── bm25_index.py       # BM25 build / cache / query
│   ├── calculator.py       # Deterministic Python arithmetic
│   ├── context_resolver.py # Pre-graph follow-up resolution
│   ├── retrieval_cache.py  # Semantic cache + invalidation
│   └── session_store.py    # SQLite CRUD
├── ingestion/              # One-time corpus pipeline
│   ├── chunker.py
│   ├── metadata_tagger.py
│   └── embed_and_store.py
└── frontend/               # React terminal UI
    └── src/
        ├── components/chat/
        ├── components/sidebar/
        └── api/
```

---

## Environment Variables

| Variable | Required | Description |
|---|:---:|---|
| `GROQ_API_KEY` | ✅ | Obtain from [console.groq.com](https://console.groq.com) |

All other configuration lives in `config.py`.

---

> **Disclaimer:** This tool queries static 10-K filings for research purposes only. It is not financial advice.
