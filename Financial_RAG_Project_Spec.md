# Financial Intelligence RAG System — Project Specification

*Last updated: 2026-07-27 (Post-BM25 Hybrid Search — Fully Implemented)*

---

## 1. Project Overview

A production-grade Retrieval-Augmented Generation (RAG) pipeline built on **2024 Annual 10-K Reports** for **9 technology companies**: Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), NVIDIA (NVDA), Tesla (TSLA), Meta (META), Alphabet (GOOGL), Netflix (NFLX), Adobe (ADBE).

The pipeline uses **LangGraph** for orchestration with a fully stateful, cyclic graph featuring:
- Intelligent query routing (retrieve / calculate / direct)
- Per-company relevance grading
- Query rewriting with retry loop
- Hybrid retrieval (BM25 + Vector + RRF) — **fully implemented**
- Calculator node (LLM extraction + deterministic Python math)
- Hallucination detection with retry/honest-failure fallback

---

## 2. Technology Stack & Configuration

### Core Tech Stack

| Layer | Technology |
|---|---|
| PDF → Markdown | **Docling** (`text_extractor.py`) — table structure detection ON, OCR OFF |
| Section Detection | Custom Python regex (SEC Item boundaries) via `ingestion/line_classifier.py` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (prose) + custom row-aware chunker (tables) |
| Token Counting | `tiktoken` (cl100k_base encoding) |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` via `langchain-huggingface` |
| Vector Store | ChromaDB (`chroma_db/`) via `langchain-chroma` |
| Sparse Index | BM25Okapi (`rank_bm25`), pickled to `./bm25_index.pkl` |
| LLM | NVIDIA NIM API (`langchain-nvidia-ai-endpoints`, `ChatNVIDIA`) |
| Graph Orchestration | LangGraph (`StateGraph`) |
| UI Framework | Streamlit *(planned — Phase 9, not yet built)* |
| Language | Python 3.11+ |

### `config.py` — All Tunable Parameters

```python
NVIDIA_API_KEY     = os.getenv("NVIDIA_API_KEY")     # from .env
NVIDIA_BASE_URL    = os.getenv("NVIDIA_BASE_URL")     # from .env

EMBEDDING_MODEL    = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE         = 450      # tokens (safely under model's 512 token limit)
CHUNK_OVERLAP      = 50       # tokens (prose only — tables use no overlap)
TOP_K              = 5        # chunks for single-company queries
MAX_RETRY          = 3        # shared across rewrite and hallucination retry cycles

CHROMA_PATH        = "./chroma_db"
COLLECTION_NAME    = "financial_10k"
BM25_INDEX_PATH    = "./bm25_index.pkl"

MODEL_ROUTER       = "meta/llama-3.1-8b-instruct"   # fast — 3-way classification
MODEL_GRADER       = "meta/llama-3.1-70b-instruct"  # per-company reasoning requires quality
MODEL_GENERATOR    = "meta/llama-3.1-70b-instruct"  # main answer generation
MODEL_HALLUC       = "meta/llama-3.1-8b-instruct"   # binary verification
MODEL_REWRITE      = "meta/llama-3.1-8b-instruct"   # query reformulation
```

### Key Dependencies (`requirements.txt`)

```
pymupdf==1.24.0
langchain==0.2.0
langchain-core==0.2.28
langchain-community==0.2.0
langchain-nvidia-ai-endpoints==0.1.0
langchain-huggingface==0.0.3
langchain-chroma==0.1.4
chromadb==0.5.0
sentence-transformers==3.0.0
langgraph==0.1.19
streamlit==1.35.0
python-dotenv==1.0.0
tiktoken==0.7.0
rank_bm25              # BM25Okapi for sparse retrieval
```

---

## 3. Project File Structure

```
RAG_Project/
├── config.py                  # Central config — all model names, paths, k values
├── .env                       # API keys (never committed)
│
├── graph/
│   ├── state.py               # GraphState TypedDict + create_initial_state()
│   ├── nodes.py               # All 10 node functions + all prompt templates
│   ├── edges.py               # Conditional edge routing functions (4 functions)
│   └── graph.py               # build_graph() + run_query()
│
├── tools/
│   ├── vectorstore.py         # ChromaDB singleton getter (lazy-loaded)
│   ├── bm25_index.py          # BM25 index build/cache/query — full implementation
│   ├── calculator.py          # compute() — pure Python arithmetic (8 operations)
│   ├── output_parsers.py      # Defensive LLM output parsers (5 parse functions)
│   └── company_names.py       # SHORT_TO_FULL mapping + get_all_full_names()
│
├── ingestion/
│   ├── line_classifier.py     # Classify lines (section_header/table_row/prose/blank)
│   ├── chunker.py             # Prose chunking (LangChain) + table chunking (custom)
│   ├── metadata_tagger.py     # Build tagged chunk dicts from extracted text files
│   ├── parent_linker.py       # Link adjacent prose<->table chunk pairs
│   ├── embed_and_store.py     # Embed all chunks and populate ChromaDB
│   └── embedding_check.py     # One-off sanity check for embedding model
│
├── extracted_text/            # Docling-extracted markdown (one .md per company PDF)
├── chroma_db/                 # Persistent ChromaDB store (3,063 chunks, 9 companies)
├── bm25_index.pkl             # Pickled BM25Okapi bundle (~8.8 MB, build-once cache)
│
├── text_extractor.py          # PDF -> markdown via Docling (run once per PDF)
├── inspect_extract.py         # Extraction inspection utility
├── test_pipeline.py           # Full-graph integration test (run from project root)
├── test_setup.py              # Environment setup verification
└── requirements.txt
```

---

## 4. Data & Ingestion Pipeline

The ingestion pipeline is a **one-time offline process** that converts raw PDFs into a queryable vector store + BM25 index.

### Phase 1: PDF Extraction (`text_extractor.py`)

- **Tool**: Docling with table structure detection ON and OCR OFF.
- **Output**: One structured `.md` file per company in `extracted_text/` (e.g., `apple_2024.md`).
- **Why Docling**: Its built-in table structure detection produces well-formed markdown tables from 10-K financial statements — a critical quality advantage over character-based extraction.
- Each PDF is processed in isolation to avoid memory accumulation across files.

### Phase 2: Line Classification (`ingestion/line_classifier.py`)

Reads each `.md` file line by line and assigns one of four types:

| Type | Detection Rule |
|---|---|
| `section_header` | Matches `^Item \d+[A]?.` regex OR markdown `#` heading matching a known SEC item title |
| `table_row` | Starts with `\|` and contains 2+ pipe characters |
| `prose` | Everything else (non-blank, non-header, non-table) |
| `blank` | Empty line — used as block separator |

Known SEC Item titles are stored in `STANDARD_ITEM_TITLES` dict (Items 1–16). The `group_into_blocks()` function then merges consecutive same-type lines into typed blocks, tracking `section_name` and `item_number` metadata.

### Phase 3: Two-Track Chunking (`ingestion/chunker.py`)

The system uses a two-track chunking strategy to avoid destroying financial tables:

| Track | Applied To | Method |
|---|---|---|
| **Track A — Prose** | MD&A, Risk Factors, Business, Notes | LangChain `RecursiveCharacterTextSplitter` — 450 tokens, 50 overlap, `cl100k_base` encoding |
| **Track B — Tables** | Financial Statements (Item 8), quantitative sections | Custom row-aware chunker — never splits a row mid-number; prepends column header row to every chunk |

**Table chunk header format** (prepended to every table chunk):
```
Apple Inc. | 2024 | Income Statement | Columns: 2024, 2023, 2022
Net income: 97,329  96,995  102,962
...
```

This makes every table chunk self-contained — the numbers are always readable alongside their column headers.

### Phase 4: Metadata Tagging (`ingestion/metadata_tagger.py`)

Each chunk is tagged with a full metadata dict:

```json
{
    "company":         "Apple Inc.",
    "ticker":          "AAPL",
    "year":            "2024",
    "item_number":     "8",
    "section_name":    "Financial Statements and Supplementary Data",
    "chunk_type":      "table",
    "table_name":      "Income Statement",
    "chunk_id":        "aapl_2024_item8_table_042_001",
    "parent_chunk_id": "aapl_2024_item8_prose_041_002",
    "page_start":      null,
    "block_idx":       42
}
```

**Chunk ID format**: `{ticker_lower}_{year}_item{item_number}_{prose|table}_{block_idx:03d}_{chunk_idx:03d}`
Example: `aapl_2024_item8_table_042_001`

**Company slug -> full name mapping** (`COMPANY_MAP` in `metadata_tagger.py`):
```python
{
    "apple":     ("Apple Inc.", "AAPL"),
    "microsoft": ("Microsoft Corporation", "MSFT"),
    "amazon":    ("Amazon.com Inc.", "AMZN"),
    "nvidia":    ("NVIDIA Corporation", "NVDA"),
    "tesla":     ("Tesla Inc.", "TSLA"),
    "meta":      ("Meta Platforms Inc.", "META"),
    "alphabet":  ("Alphabet Inc.", "GOOGL"),
    "netflix":   ("Netflix Inc.", "NFLX"),
    "adobe":     ("Adobe Inc.", "ADBE"),
}
```

### Phase 5: Parent Linking (`ingestion/parent_linker.py`)

After all chunks are tagged, `link_parent_chunks()` scans adjacent blocks. If two consecutive blocks are in the **same section** (`section_name` + `item_number` match) but of **different types** (prose -> table or table -> prose), the first chunk of the second block gets a `parent_chunk_id` pointing to the last chunk of the first block. This links explanatory prose to its adjacent financial table.

### Phase 6: Boilerplate Filtering + Embedding (`ingestion/embed_and_store.py`)

- **Boilerplate filter**: Chunks without an `item_number` are excluded (they belong to table-of-contents, cover pages, or other preamble — not queryable content).
- **Embedding**: `HuggingFaceEmbeddings("sentence-transformers/all-mpnet-base-v2")` — runs locally.
- **Storage**: `Chroma.from_texts()` with full metadata; persisted to `./chroma_db`, collection `financial_10k`.

### ChromaDB — Current Corpus State

**3,063 chunks** across 9 companies in collection `financial_10k`:

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
    question:             str               # original user question
    rewritten_question:   str               # reformulated by rewrite_node (or "" if unused)
    route:                str               # "retrieve" / "calculate" / "direct"
    companies_mentioned:  List[str]         # extracted company names or ["all"]
    retrieved_chunks:     List[str]         # parallel list of chunk text
    chunk_sources:        List[dict]        # parallel list of chunk metadata dicts
    relevant:             str               # "yes" / "no" (set by grade_node)
    answer:               str               # intermediate answer (before hallucination check)
    grounded:             str               # "grounded" / "not_grounded"
    retry_count:          int               # shared across rewrite + hallucination retries
    final_answer:         str               # the answer exposed to the user / UI
    error_message:        Optional[str]     # low-confidence warning or honest failure msg
```

Initial state created by `create_initial_state(question)` with all fields zeroed/empty.

---

## 6. LangGraph Node Architecture

### All Nodes (`graph/nodes.py`)

| Node | Function | Model | Purpose |
|---|---|---|---|
| `router` | `router_node()` | 8B | Classify route + extract companies (2 LLM calls) |
| `retrieve` | `retrieve_node()` | — | Hybrid BM25+Vector search, RRF merge per company |
| `grade` | `grade_node()` | 70B | Per-company relevance check — must pass for EVERY company |
| `rewrite` | `rewrite_node()` | 8B | Reformulate question with standard financial terminology |
| `generate` | `generate_node()` | 70B | Synthesize answer from labeled context chunks |
| `calculator` | `calculator_node()` | 70B | Extract numbers from chunks -> Python `compute()` |
| `direct_answer` | `direct_answer_node()` | 70B | Answer general finance concepts (no document lookup) |
| `hallucination_check` | `hallucination_check_node()` | 8B | Verify every figure is traceable to source chunks |
| `grade_exhausted_warning` | `grade_exhausted_warning_node()` | — | Write low-confidence warning to `error_message` |
| `hallucination_exhausted` | `hallucination_exhausted_node()` | — | Write honest failure message to `final_answer` |

### Graph Entry Point & Edge Routing (`graph/edges.py`)

Entry point: `router` node.

```
router
  |--[route == "direct"]-----------> direct_answer --> END
  |--[route == "retrieve"/"calculate"] -> retrieve
                                            |
                                          grade
                              [yes]     [rewrite]    [exhausted]
                               |            |             |
                          generate/     retrieve    grade_exhausted_warning
                          calculator      (^)         |-> generate
                               |                      |-> calculator
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
    return "direct" if state["route"] == "direct" else "retrieve"

def route_after_grade(state) -> str:
    if state["relevant"] == "yes":
        return "calculate" if state["route"] == "calculate" else "generate"
    if state["retry_count"] < config.MAX_RETRY:
        return "rewrite"
    return "exhausted"

def route_by_calc_type(state) -> str:
    # After grade_exhausted_warning: still pick generate vs calculate
    return "calculate" if state["route"] == "calculate" else "generate"

def route_after_hallucination(state) -> str:
    if state["grounded"] == "grounded":
        return "end"
    if state["retry_count"] < config.MAX_RETRY:
        return "calculate" if state["route"] == "calculate" else "generate"
    return "exhausted"
```

---

## 7. Retrieval Strategy — Hybrid Search (Fully Implemented)

### Why Hybrid? The Pure-Vector Problem

Pure vector (embedding) similarity favors narrative text over numeric tables. In real testing, MD&A paragraphs *mentioning* a metric consistently out-ranked the actual financial table *containing* the exact numeric total — even after query rewrites. BM25 fixes this by rewarding chunks where the query's exact terms are densely concentrated.

### Architecture (`tools/bm25_index.py`)

**Design decisions:**
1. **ONE global BM25 index** — IDF is more meaningful when computed over the full corpus (all 9 companies, 3,063 chunks), not per-company.
2. **Pickle cache** — index is built once and saved to `./bm25_index.pkl` (~8.8 MB). Loaded on subsequent runs. Delete the `.pkl` to force a rebuild.
3. **Corpus pulled from Chroma `.get()`** — same chunks as the vector index, exact parity.
4. **Module-level lazy singleton** — loaded once per process lifetime (`_bundle` global). Repeated calls within a run don't reload from disk.

### Tokenizer (applied identically at build time and query time)

```python
_ABBREV_SUBS = [
    # R&D — plain and HTML-encoded (&amp;)
    (re.compile(r'r\s*&\s*amp\s*;\s*d',  re.IGNORECASE), 'research and development'),
    (re.compile(r'r\s*&\s*d',             re.IGNORECASE), 'research and development'),
    # SG&A — plain and HTML-encoded
    (re.compile(r'sg\s*&\s*amp\s*;\s*a', re.IGNORECASE), 'selling general and administrative'),
    (re.compile(r'sg\s*&\s*a',           re.IGNORECASE), 'selling general and administrative'),
    # Generic fallback: &amp; and & -> "and"
    (re.compile(r'&amp;', re.IGNORECASE), ' and '),
    (re.compile(r'&'),                    ' and '),
]
# Pipeline: lowercase -> expand abbreviations -> strip punctuation -> split
```

**Why abbreviation expansion first**: `R&D` must expand to `research and development` *before* punctuation stripping. If punctuation is stripped first, `R&D` becomes `R`, `D` — tokens with near-zero BM25 weight.

### Reciprocal Rank Fusion (RRF) Merge (`_rrf_merge()`)

```
RRF_score(chunk) = 1/(60 + rank_in_vector) + 1/(60 + rank_in_bm25)
```

- Both methods run **per company** (not across the corpus).
- A chunk surfaced by only one method still gets a score — it is not dropped.
- Chunks are deduplicated by `chunk_id`. The first retrieval method to encounter a chunk sets its text and metadata; subsequent encounters only add their RRF score contribution.

### `bm25_query()` — Company Filtering

BM25 scores ALL 3,063 chunks in the corpus, then filters to the requested company using the `metadata["company"]` field (always a populated full name). Returns a ranked list with `score`, `rank` (1-based within company), `text`, `metadata`, and `id`.

### Chunk Budget per Query Type (`retrieve_node()`)

| Query Type | Vector k | BM25 top_k | Final k (after RRF) | Total Chunks |
|---|---|---|---|---|
| Single company | 5 | 20 | 5 | 5 |
| Two+ specific companies | 4 per company | 20 per company | 4 per company | 8+ |
| All companies (`["all"]`) | 4 per company | 20 per company | 4 per company | 36 |

---

## 8. Context Formatting — Cross-Company Safety

Before any LLM call that uses retrieved chunks, `_format_context()` wraps each chunk in a company/section header:

```
=== APPLE INC. — Financial Statements and Supplementary Data ===
[chunk text]

=== MICROSOFT CORPORATION — Management's Discussion and Analysis ===
[chunk text]
```

This prevents cross-company number contamination — the LLM always knows which company each figure belongs to.

---

## 9. Prompt Engineering & LLM Nodes

### Router Prompt
Classifies into exactly one of: `"retrieve"`, `"calculate"`, `"direct"`.

**Key distinction**: `"retrieve"` covers all directly-stated figures (revenue, net income, R&D expense) — even ones that *sound* like math. `"calculate"` is reserved for arithmetic *on top of* stated figures (YoY growth rate, margin computation, cross-company comparison).

**Returns**: A single word. Parsed by `parse_route()` which scans for keywords and defaults to `"retrieve"`.

### Company Extraction Prompt
Extracts company names from the question. Handles the Google/Alphabet alias. Returns `["all"]` if no specific company is mentioned.

**Returns**: JSON array. Parsed defensively by `parse_companies()` (handles markdown code fences, falls back to `["all"]`).

### Grade Prompt
Evaluates retrieved chunks against the question. Critical rule: must verify every *listed* company has its specific figure — not just most of them.

**Returns**: Last line is `"yes"` or `"no"`. Parsed by `parse_grade()` (checks last line only).

### Rewrite Prompt
Reformulates a failing query using standard financial statement terminology (e.g., "net sales", "total revenue", "operating income"). Strictly forbidden from adding year, section, or figure assumptions not in the original question.

**Returns**: Rewritten question string.

### Generate Prompt
Answers using ONLY the provided context from 10-K filings. Must cite company and section for each figure. Explicitly forbidden from using training data for specific numbers.

### Calculator Extraction Prompt
Extracts `{operation, values}` JSON from context chunks. Contains critical rules:

- **Percentage vs dollar comparison**: use `"difference"` for two percentages (not `"ratio"`).
- **Consolidated figures rule**: always prefer consolidated company totals over segment-level figures.
- **Missing company rule**: if a company's figure is absent, include it with `value: 0` and label ending in `" (not found in retrieved chunks)"` — never silently omit it.
- **Ordering rules**: base/older/denominator listed first for `percent_change`, `difference`, `ratio`.
- **Fallback**: returns `{"operation": "insufficient_data", "values": []}` if no valid numbers found.

**Returns**: JSON object. Parsed by `parse_calculation()` using regex to extract `{...}` block (handles code fences and conversational text wrapping).

### Hallucination Check Prompt
Verifies every specific number, percentage, date, and financial figure in the answer exists in the source chunks. Reasons figure-by-figure, then writes final verdict on the last line.

**Returns**: `"grounded"` or `"not_grounded"`. Parsed by `parse_hallucination()` — checks `"not_grounded"` BEFORE `"grounded"` (since `"grounded"` is a substring of `"not_grounded"`). Defaults to `"not_grounded"` on any parse failure.

### Direct Answer Prompt
Answers general finance/accounting concept questions (definitions, explanations) using the LLM's own knowledge. If the question actually requires specific company figures, the LLM is instructed to say so rather than guess.

---

## 10. Calculator Node — Two-Step Design

Arithmetic never goes through an LLM. The `calculator_node()` uses a deliberate two-step design:

1. **LLM extraction** (`calculator_extract_prompt`): identifies `{operation, values}` JSON from chunk text.
2. **Python `compute()`** (`tools/calculator.py`): performs the actual math deterministically.

**Supported operations in `compute()`:**

| Operation | Behavior |
|---|---|
| `percent_change` | `(new - old) / old * 100` — raises ValueError if base is 0 |
| `difference` | `values[0] - values[1]` |
| `sum` | Sum of all values |
| `average` | Mean of all values |
| `ratio` | `values[0] / values[1]` — raises ValueError if denominator is 0 |
| `margin` | `(base - subtract) / base * 100` — e.g., gross margin |
| `max` | Returns `{label, value}` dict of winner (never picks a 0-placeholder) |
| `min` | Returns `{label, value}` dict of winner (never picks a 0-placeholder) |

**Missing company handling**: `max`/`min` operations exclude `0`-placeholder entries (where label contains `"(not found in retrieved chunks)"`) from the winner selection, but still surface them as caveats in the final answer.

---

## 11. Company Name Mapping (`tools/company_names.py`)

```python
SHORT_TO_FULL = {
    "Apple":     "Apple Inc.",
    "Microsoft": "Microsoft Corporation",
    "Amazon":    "Amazon.com Inc.",
    "NVIDIA":    "NVIDIA Corporation",
    "Tesla":     "Tesla Inc.",
    "Meta":      "Meta Platforms Inc.",
    "Alphabet":  "Alphabet Inc.",
    "Google":    "Alphabet Inc.",   # alias — resolves to same filing
    "Netflix":   "Netflix Inc.",
    "Adobe":     "Adobe Inc.",
}
```

`get_all_full_names()` returns `list(SHORT_TO_FULL.values())`. Note: `"Alphabet Inc."` appears twice (for both `"Alphabet"` and `"Google"` keys). For `["all"]` queries in `retrieve_node`, this causes two redundant Chroma + BM25 calls for Alphabet — no correctness impact, minor efficiency waste (see Known Issues).

---

## 12. Output Parsers (`tools/output_parsers.py`)

All LLM outputs are wrapped in defensive parsers that normalize to exact expected strings:

| Parser | Input | Output | Fallback |
|---|---|---|---|
| `parse_route()` | LLM route string | `"retrieve"/"calculate"/"direct"` | `"retrieve"` (safest) |
| `parse_companies()` | LLM JSON array | `list[str]` | `["all"]` |
| `parse_grade()` | LLM yes/no | `"yes"/"no"` | `"no"` |
| `parse_hallucination()` | LLM grounded/not | `"grounded"/"not_grounded"` | `"not_grounded"` (safer) |
| `parse_calculation()` | LLM JSON object | `{operation, values}` dict | `{"operation": "insufficient_data", "values": []}` |

---

## 13. Failure Mode Handling

| Scenario | What Happens |
|---|---|
| Irrelevant chunks retrieved | `grade_node` returns `"no"` -> `rewrite_node` reformulates -> retry (up to `MAX_RETRY=3` times) |
| Retries exhausted, still no relevant chunks | `grade_exhausted_warning_node` sets `error_message`; pipeline continues to generate/calculate with best available chunks |
| Answer contains figures not in source chunks | `hallucination_check_node` returns `"not_grounded"` -> retry generate/calculate |
| Hallucination retries exhausted | `hallucination_exhausted_node` writes honest failure message to `final_answer`; unverified answer is never exposed to the user |
| Calculation: division by zero or wrong operation | `compute()` exception caught in `calculator_node`; descriptive error string returned as answer |
| Missing company figure in multi-company calculation | Placeholder `value: 0` with `"(not found in retrieved chunks)"` label; excluded from `max`/`min` winner; surfaced as caveat in answer text |
| Unknown company alias | `SHORT_TO_FULL.get(name)` returns `None`; that company is silently skipped in retrieve loop — caught downstream by grade returning `"no"` |

---

## 14. Development Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | PDF extraction (Docling), chunking, embedding, ChromaDB | COMPLETE |
| Phase 2 | Basic RAG — Retrieve + Generate + Hallucination Check | COMPLETE |
| Phase 3 | Router node — retrieve / calculate / direct | COMPLETE |
| Phase 4 | Grade + Rewrite retry loop | COMPLETE |
| Phase 5 | Hallucination Check + retry/honest-failure fallback | COMPLETE |
| Phase 6 | Multi-company queries (all 9 companies) | COMPLETE |
| Phase 7 | Calculator — single and multi-company | COMPLETE |
| Phase 8 | Hybrid Search — BM25 + Vector via RRF | COMPLETE |
| Phase 9 | Streamlit UI | NOT STARTED |

---

## 15. Known Issues & Open Decisions

### Active Issues

- **Chunk-Budget Competition (3+ Companies)**: At 9 companies x 4 slots = 36 chunks, budget competition occasionally lets a prose chunk out-compete the optimal table chunk for a given company. `grade_node` correctly rejects this, triggering rewrites — but the root cause is the fixed budget. Dynamic slot allocation (or raising `k`) remains an open decision.

- **Error Propagation**: `grade_exhausted_warning_node` logs `error_message` internally, but it is not currently injected into `final_answer` for the user to see in the UI. (Deferred to Phase 9 UI.)

- **`get_all_full_names()` returns duplicates**: `"Alphabet Inc."` appears twice in the `SHORT_TO_FULL.values()` list (once for `"Alphabet"`, once for `"Google"` alias). For `["all"]` queries, `retrieve_node` makes two redundant Chroma + BM25 calls for Alphabet. No correctness impact, minor efficiency waste.

### Resolved Issues (for reference)

- **Tesla vs. NVIDIA percentage comparisons**: Fixed by the `"difference"` operation rule for comparing two percentage-based metrics.
- **Google -> Alphabet mapping**: `SHORT_TO_FULL["Google"] = "Alphabet Inc."` alias fully resolves "Google" queries to the correct filing.
- **R&D and SG&A abbreviations in BM25**: Handled by the ordered `_ABBREV_SUBS` expansion before punctuation stripping — `R&D` correctly expands to `research and development`.

---

## 16. Streamlit UI Design (Phase 9 — Not Started)

Planned design principles:
- **Status Display**: Use `st.status()` to show live execution steps (Router -> Retrieve -> Grade -> Generate).
- **Source Disclosure**: Expandable "View sources used" panel showing chunk text, company, section label, and chunk type — to foster user trust.
- **Honest Failures**: Never display a hallucinated or unverified answer. If `final_answer` comes from `hallucination_exhausted_node`, display: *"Unable to generate a verified answer. The model could not produce a response grounded in the 10-K documents."*
- **Low-Confidence Warning**: If `error_message` is set (grade exhausted), display a visible warning alongside the answer.

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
