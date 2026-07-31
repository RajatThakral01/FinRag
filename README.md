# 📊 Financial Statement RAG Intelligence Terminal
### Multi-Company 10-K Analysis · Adaptive RAG · LangGraph · FastAPI · React + TypeScript

A **production-grade Financial Intelligence Terminal** built on 2024 Annual Report (10-K) filings from major technology corporations (Apple, Microsoft, NVIDIA, Tesla, Amazon, Meta, Alphabet, Netflix, Adobe). Analysts can query natural language financial metrics and receive verifiable, audit-ready answers backed by exact filing excerpts, automatic context resolution, hallucination detection, and interactive chunk inspection.

---

## 🏗️ Architecture & Component Stack

The system consists of a Python FastAPI backend running a stateful LangGraph RAG pipeline and a modern React/TypeScript frontend designed as a high-precision research terminal.

```
[ Financial RAG Web Client (React + TS) ]
                │
                ▼ REST API (Port 8000)
┌──────────────────────────────────────────────────────────────┐
│  FastAPI Backend (api.py) & SQLite Store (session_data.db)   │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│            LangGraph Adaptive RAG State Machine             │
│                                                              │
│  [Router Node] ──► [Retrieve Node (RRF)] ──► [Grade Node]    │
│        │                    │                    │           │
│        ▼                    ▼                    ▼           │
│   [Direct/Math]      [ChromaDB + BM25]     [Rewrite Cycle]   │
│        │                                         │           │
│        └───────────────► [Generate Node] ────────┘           │
│                                │                             │
│                                ▼                             │
│                     [Hallucination Check]                    │
│                                │                             │
│                                ▼                             │
│                 [Grounded Answer / Honest Failure]           │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Capabilities & Features

### 1. High-Precision Financial Retrieval (RRF Hybrid Search)
- **Docling Conversion:** 10-K PDFs are converted into structured markdown with table structure detection.
- **Two-Track Chunking:**
  - *Track A (Prose):* 450-token recursive splitting with 50-token overlap.
  - *Track B (Tables):* Row-aware table chunking preserving column headers on every chunk.
- **Reciprocal Rank Fusion (RRF):** Merges vector similarity search (ChromaDB `all-mpnet-base-v2`) and BM25 keyword matching for optimal recall.

### 2. Multi-Turn Session Memory & Query Context Rewriting
- **Automatic Context Resolution:** Follow-up questions (e.g. *"What about its R&D expense?"*) automatically resolve previous turn contexts using session history stored in SQLite.
- **Retrieval Cache:** Semantic caching with cosine similarity matching (`threshold = 0.88`) returns instant verified responses for repeated queries.

### 3. Institutional Monochromatic Terminal Interface
- **Palette:** Graphite dark mode (`#111214` page background, `#1A1C20` surface background, `#2A2B2F` border) with a single muted blue-gray accent (`#8CA0C7`). Zero arbitrary color-coding.
- **Typography:** `Newsreader` display serif for headers, `Inter` for body prose, and `JetBrains Mono` (`tabular-nums`) for financial figures and metadata tags.
- **Structural Semantic States:**
  - *Verified Answer:* Thin solid `#8CA0C7` accent left border + `Cache Verified` stamp.
  - *Low-Confidence Warning:* 3px dashed `#8B8D94` gray left border + outline `AlertTriangle` icon + `"LOW CONFIDENCE"` label.
  - *Honest-Failure Placeholder:* 3px dashed `#8B8D94` gray left border + outline `AlertCircle` icon + `"UNVERIFIED"` label.

### 4. Interactive Excerpt Inspection & Slide-in Full Chunk Panel
- **Audited Citation Index:** Cards highlight exact company, ticker, fiscal year, filing section, document type (`[TABLE]` vs `[PROSE]`), and monospace `chunk_id`.
- **Slide-in Preview Panel:** Clicking *"Show full chunk"* slides in a non-modal side panel (~38% viewport width) displaying the complete 10-K excerpt text in readable `Inter` prose, fetched dynamically via `GET /chunks/{chunk_id}`.

---

## 📡 REST API Reference

The backend exposes a REST API via FastAPI on port `8000`:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/sessions` | Create a new research session. Returns `{ "session_id": "uuid-string" }`. |
| `GET` | `/sessions` | List active sessions ordered by most recent activity. |
| `GET` | `/sessions/{id}/turns` | Retrieve full turn history for a session. |
| `POST` | `/sessions/{id}/query` | Submit a financial query. Runs multi-turn resolution & LangGraph RAG pipeline. |
| `GET` | `/chunks/{chunk_id}` | Retrieve full document excerpt text and metadata from ChromaDB for side panel preview. |

---

## 📝 Model Configuration (`config.py`)

Models are tuned for latency, reasoning stability, and instruction adherence via Groq API:

| Node | Model | Role |
|---|---|---|
| `MODEL_ROUTER` | `llama-3.1-8b-instant` | Classify route: `retrieve`, `calculate`, or `direct` |
| `MODEL_GRADER` | `llama-3.1-8b-instant` | Verify excerpt relevance & multi-company completeness |
| `MODEL_GENERATOR` | `llama-3.3-70b-versatile` | Synthesize comprehensive, grounded financial answers |
| `MODEL_HALLUC` | `llama-3.1-8b-instant` | Verify numerical claims against source chunks |
| `MODEL_REWRITE` | `llama-3.1-8b-instant` | Reformulate failed queries into standard financial terminology |
| `MODEL_CALCULATOR` | `llama-3.1-8b-instant` | Mathematical expression formulation |

---

## 🚀 Running the Project Locally

### 1. Prerequisites
- Python 3.12+
- Node.js 18+ and npm
- Groq API Key set in `.env` (`GROQ_API_KEY=gsk_...`)

### 2. Backend Setup & Start
```bash
# In the project root
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Launch FastAPI server on port 8000
PYTHONUNBUFFERED=1 ./venv/bin/uvicorn api:app --port 8000
```

### 3. Frontend Setup & Start
```bash
# In the frontend directory
cd frontend
npm install

# Start Vite dev server on port 3000
npm run dev
```

Open `http://localhost:3000/` in your browser to access the FinRAG Terminal.

---

## 📄 License

This repository is built for portfolio and financial research purposes.
