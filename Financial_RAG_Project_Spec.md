# Financial Intelligence RAG System — Project Specification

*Last updated: 2026-07-24 (Post-BM25 Hybrid Search Implementation)*

## 1. Project Overview
A production-grade Retrieval-Augmented Generation (RAG) pipeline built on 2024 Annual 10-K Reports for **9 technology companies**: Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), NVIDIA (NVDA), Tesla (TSLA), Meta (META), Alphabet (GOOGL), Netflix (NFLX), Adobe (ADBE).
The pipeline uses **LangGraph** for orchestration, featuring intelligent routing, relevance grading, query rewriting, calculation tools, and hallucination detection.

---

## 2. Technology Stack & Configuration
### Core Tech
- **PDF Parsing**: PyMuPDF (`fitz`) and custom regex for SEC Item detection.
- **Vector Store**: ChromaDB (`chroma_db/`).
- **Embeddings**: Local `sentence-transformers/all-mpnet-base-v2` via LangChain.
- **Sparse Index**: BM25 (`rank_bm25`), cached to `./bm25_index.pkl`.
- **LLM**: NVIDIA NIM API (`ChatNVIDIA`).
- **Graph Orchestration**: LangGraph.
- **UI Framework**: Streamlit (Pending Phase 8).

### `config.py` Values
```python
EMBEDDING_MODEL    = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE         = 450   # tokens
CHUNK_OVERLAP      = 50    # tokens
TOP_K              = 5     # chunks to retrieve (single-company)
MAX_RETRY          = 3     # max rewrite/hallucination retries

MODEL_ROUTER       = "meta/llama-3.1-8b-instruct"
MODEL_GRADER       = "meta/llama-3.1-70b-instruct" 
MODEL_GENERATOR    = "meta/llama-3.1-70b-instruct"
MODEL_HALLUC       = "meta/llama-3.1-8b-instruct"
MODEL_REWRITE      = "meta/llama-3.1-8b-instruct"
```

---

## 3. Data & Ingestion (Chunking Strategy)
The system uses a two-track chunking strategy to avoid destroying financial tables.
- **Track A (Prose)**: LangChain `RecursiveCharacterTextSplitter`. Chunk boundaries never fall mid-sentence.
- **Track B (Tables)**: Custom table-aware chunker. Extracts table header row (e.g. "2024", "2023") and prepends it to every chunk from that table. Never splits a row mid-number.

**Chunk Metadata Schema:**
```json
{
    "company":       "Apple",
    "ticker":        "AAPL",
    "year":          "2024",
    "item_number":   "Item 8",
    "section_name":  "Financial Statements",
    "chunk_type":    "table",
    "table_name":    "Income Statement",
    "chunk_id":      "apple_2024_item8_042",
    "parent_chunk_id": "apple_2024_item8_041" // Links table row to explaining prose
}
```

---

## 4. System Architecture & Node Flow

### Graph State
```python
class GraphState(TypedDict):
    question:             str
    rewritten_question:   str
    route:                str            # retrieve / calculate / direct
    companies_mentioned:  List[str]
    retrieved_chunks:     List[str]
    chunk_sources:        List[dict]
    relevant:             str            # "yes" / "no"
    answer:               str
    grounded:             str            # "grounded" / "not_grounded"
    retry_count:          int
    final_answer:         str
    error_message:        Optional[str]
```

### Conditional Edge Logic
- **Router** → `Retrieve` (if retrieve/calculate) OR `Direct Answer`.
- **Calculate Route** → Continues to `Grade` before passing to `Calculator`.
- **Grade** → `Generate` (if yes), `Rewrite` (if no & retries < 3), `warning_node` (if no & retries >= 3).
- **Rewrite** → `Retrieve` (always).
- **Hallucination Check** → `END` (if grounded), `Generate` (if not & retries < 3), `exhausted_node` (if not & retries >= 3).

*(Note: There are also two plumbing nodes, `grade_exhausted_warning_node` and `hallucination_exhausted_node`, to write error states to `state["error_message"]` when `MAX_RETRY` is hit).*

---

## 5. Retrieval Strategy (Hybrid Search)
**BM25 + Vector Search**
To prevent dense embeddings from favoring narrative text over numeric tables, the retrieval node uses both:
1. **Vector**: ChromaDB semantic search.
2. **BM25**: Sparse search over all docs (loaded from `.pkl` cache).
   - **Tokenizer rules**: Abbreviations like `R&D` and `SG&A` are expanded via regex to `research and development` and `selling general and administrative` *before* punctuation stripping.

**Reciprocal Rank Fusion (RRF)**
Results are merged per company: `RRF = 1/(60+rank_vector) + 1/(60+rank_bm25)`.

**Budget & Filtering**
- **Single-company**: 5 chunks.
- **Multi-company / "All"**: 4 chunks per company (e.g. 9 companies = 36 chunks). ChromaDB `filter={"company": ...}` is applied prior to scoring.

---

## 6. Prompt Engineering & LLM Parsers

### Prompt: Router
`System: You are a financial query router. Classify the question into exactly one of these three routes: "retrieve", "calculate", "direct". Return ONLY the single word.`

### Prompt: Company Extractor
`System: Extract company names from the question. Return ONLY a JSON array containing names from: Apple, Microsoft, Amazon, NVIDIA, Tesla, Meta, Alphabet, Netflix, Adobe. If no specific company is mentioned, return ["all"].`

### Prompt: Grade (Corrected Per-Company Logic)
`System: Given a question, a list of companies the question requires data for, and retrieved document chunks, decide if the chunks contain sufficient information to answer the question for EVERY listed company — not most of them. Reason company-by-company before your final line, then return ONLY "yes" or "no" as the last line.`

### Prompt: Hallucination Check
`System: Verify that every specific number, percentage, date, and financial figure in the answer is directly traceable to the source chunks. Return ONLY "grounded" or "not_grounded".`

### Calculator Tool Design
- *LLM Step*: Extracts `{operation, values}` JSON. Operations: `percent_change, difference, sum, average, ratio, margin, max, min`.
  - **Rules**: Percentage-vs-percentage uses `difference` (not `ratio`). Prefers consolidated company figures over segment-level figures.
- *Python Step*: `tools/calculator.py` computes the math.

### Defensive Parsers
LLM outputs are wrapped in defensive `parse_grade()`, `parse_hallucination()`, and `parse_route()` functions to handle leading/trailing conversational text and normalize to exact expected strings.

---

## 7. Streamlit UI Design (Phase 8 - Pending)
- **Status Display**: Use `st.status()` to show live execution steps (Router → Retrieve → Grade → Generate).
- **Source Disclosure**: Must include an expandable "View sources used" panel displaying chunks, company, and section labels to foster trust.
- **Honest Failures**: Do not display a hallucinated answer if verification fails. Present: `"Unable to generate a verified answer. The model could not produce a response grounded in the 10-K documents."`

---

## 8. Validated State & Open Issues
- **Single/Two-Company Queries**: Fully working (Retrieval and Calculation routes). Bug B (Tesla vs. NVIDIA percentage comparisons) is fixed, parsing correctly.
- **Router (Google Fix)**: "Google" correctly maps to "Alphabet Inc." via `SHORT_TO_FULL`.
- **Hybrid Retrieval (BM25 + Vector)**: Fully implemented and working. R&D and SG&A abbreviations expand properly.
- **Chunk-Budget Competition (3+ Companies)**: At a large scale (e.g. 9 companies), limiting a company to 4 slots occasionally allows prose to out-compete the optimal table chunk. `grade_node` rightfully rejects this, but the root cause is the budget limit. Raising `k` or dynamic slot allocation remains undecided.
- **Error Propagation**: `grade_exhausted_warning_node` logs an `error_message` internally but it doesn't currently inject into the user's `final_answer`. (Deferred).
