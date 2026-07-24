

**PRODUCT REQUIREMENTS DOCUMENT — v3.0**
# Financial Intelligence RAG System
### Multi-Company 10-K Analysis · Adaptive RAG · LangGraph
****
**Updated with: Chunking Strategy · Environment Setup · Multi-Company Queries**
**LLM Output Parsing · NIM Model Selection · ChromaDB Filtering Design**
**PDF Naming Convention · Retry Behavior · Test Questions · UI Design**
**v3: As-Built Corrections · Retrieval-Completeness Bug Fixes · Hybrid (BM25+Vector) Search Plan**

> **A note on how to read this v3.0 update:** Sections marked **[AS-BUILT]** describe what the real codebase actually does today, confirmed against real code/output, and take precedence over any conflicting text elsewhere in this document. Sections marked **[PLANNED — NOT YET IMPLEMENTED]** describe confirmed direction that has been discussed and decided but has zero code written/run yet. Original v2.0 text that has been superseded is struck through in intent (kept in the doc for history) with a correction directly beneath it.



| Field | Detail |
| --- | --- |
| Project Name | Financial Intelligence RAG System |
| Version | 3.0 (as-built corrections + hybrid search plan) |
| Status | In Development — Phases 0–5 done & validated, Phase 6 in progress (fixes written, re-test pending), Phase 7 partially validated, Phase 8 not started |
| Data | 2024 Annual 10-K Reports — **9 Companies** (Intel removed — see Ch 01) |
| Author | Rajat Thakral |
| Changes from v1 | Chunking strategy, LangChain scope, environment setup, multi-company handling, NIM model selection, output parsing, PDF naming, retry UX, test questions, UI design |
| Changes from v2 | Removed Intel (9 companies, not 10); corrected actual file structure (`RAG_Project/`, not `financial-rag/`); corrected actual config attribute names and model assignments; corrected `grounded` field to string values `"grounded"`/`"not_grounded"`; documented Calculator's real two-step (LLM-extract + Python-compute) design and its 7 operations; documented 2 extra plumbing nodes not in the original node table; documented Grade-prompt per-company-completeness fix and Calculator operation/consolidated-figure fixes (Bugs A & B — fixes written, not yet re-validated); added Chapter 21 — Hybrid (BM25 + Vector) Search, a confirmed but unimplemented plan, since real-evidence testing showed the root cause of retrieval misses is **vector-search-only retrieval**, not the prompts (prompt fixes for Bugs A/B alone do not resolve this) |




# Contents



  Chapter 01  —  Project Overview & Companies

  Chapter 02  —  Problem Statement

  Chapter 03  —  Technology Stack (updated)

  Chapter 04  —  Environment Setup — NEW

  Chapter 05  —  System Architecture

  Chapter 06  —  Data & Ingestion — Updated Chunking Strategy

  Chapter 07  —  PDF Naming Convention — NEW

  Chapter 08  —  ChromaDB Filtering Design — NEW

  Chapter 09  —  Multi-Company Query Handling — NEW

  Chapter 10  —  Node-by-Node Design

  Chapter 11  —  LLM Output Parsing Strategy — NEW

  Chapter 12  —  Retry Limit Behavior — NEW

  Chapter 13  —  NVIDIA NIM Model Selection — NEW

  Chapter 14  —  Project File Structure

  Chapter 15  —  Development Phases & Test Questions (updated)

  Chapter 16  —  Prompt Design (updated)

  Chapter 17  —  Configuration (updated)

  Chapter 18  —  Streamlit UI Design — NEW

  Chapter 19  —  Known Risks & Mitigations

  Chapter 20  —  Resume & Portfolio Value

  Chapter 21  —  Hybrid Search: BM25 + Vector (RRF) — NEW in v3, CONFIRMED PLAN, NOT YET IMPLEMENTED



# Chapter 01 — Project Overview

This system is a production-grade Retrieval-Augmented Generation (RAG) pipeline built on 2024 Annual Report (10-K) filings from ~~10~~ **9 major technology companies**. Users ask natural language questions about financial data and receive accurate, grounded, verifiable answers. The pipeline goes beyond basic RAG with relevance grading, query rewriting, hallucination detection, and intelligent routing.

**[AS-BUILT — v3]** Intel was removed from scope per an earlier confirmed decision. The company list below is the real, current 9-company list. Every other chapter's references to "10 companies" (Ch 04.2 requirements assumptions, Ch 07 PDF naming, Ch 08–09 filtering/multi-company math, Ch 16.2 extractor prompt, Ch 17 top-K notes) should be read as 9, not 10 — see the corrected tables in those chapters below.

| # | Company | Ticker | Sector |
| --- | --- | --- | --- |
| 1 | Apple Inc. | AAPL | Consumer Electronics / Software |
| 2 | Microsoft Corporation | MSFT | Cloud / Enterprise Software |
| 3 | Amazon.com Inc. | AMZN | E-Commerce / Cloud (AWS) |
| 4 | NVIDIA Corporation | NVDA | Semiconductors / AI Hardware |
| 5 | Tesla Inc. | TSLA | Electric Vehicles / Energy |
| 6 | Meta Platforms Inc. | META | Social Media / VR |
| 7 | Alphabet Inc. | GOOGL | Search / Cloud / Advertising |
| 8 | Netflix Inc. | NFLX | Streaming / Content |
| 9 | Adobe Inc. | ADBE | Creative / Document Software |

~~10 | Intel Corporation | INTC | Semiconductors / PC Hardware~~ — **removed from scope, not built.**




# Chapter 02 — Problem Statement

10-K annual reports are among the most information-dense financial documents that exist. A single filing can be 100-200 pages of financial statements, risk disclosures, management analysis, and legal text. Manually extracting insights across 10 companies is time-consuming and error-prone.


Existing LLM solutions have two critical failure modes:

- Retrieval failure — wrong chunks returned, LLM answers from irrelevant context or general training knowledge
- Hallucination — LLM confidently states financial figures that do not exist in retrieved documents

| Question Type | Example Question |
| --- | --- |
| Factual | What was Apple's total revenue in fiscal year 2024? |
| Factual | What are NVIDIA's primary risk factors for 2024? |
| Calculation | Which company had the highest gross profit margin in 2024? |
| Calculation | What was the YoY revenue growth rate for Amazon? |
| Comparative | Compare operating income between Microsoft and Alphabet |
| Comparative | Which company invested the most in R&D in 2024? |
| Direct | What does EBITDA stand for? |




# Chapter 03 — Technology Stack

| Layer | Technology | Purpose | Why This Choice |
| --- | --- | --- | --- |
| PDF Parsing | PyMuPDF (fitz) | Extract raw text from 10-K PDFs | Fast, accurate, handles complex PDF layouts |
| Section Detection | Custom Python regex | Detect SEC Item boundaries | Standard chunkers ignore financial structure |
| Chunking | LangChain RecursiveCharacterTextSplitter | Split sections into token-sized chunks | Respects sentence/paragraph boundaries, LangGraph compatible |
| Table Chunking | Custom Python | Detect and preserve table structure | Tables need row-aware splitting, not character-based |
| Embedding | all-mpnet-base-v2 via LangChain HuggingFaceEmbeddings | Convert chunks to vectors | Local, free, 512 token limit, best quality |
| Vector Store | LangChain Chroma wrapper | Store and search embedded chunks | Local, persistent, metadata filtering built in |
| LLM | LangChain ChatNVIDIA (NVIDIA NIM API) | Router, Grader, Generator, Hallucination checker | Free credits, powerful models, LangChain native |
| Prompts | LangChain ChatPromptTemplate | Structure all LLM calls consistently | Standard interface, easy to test and swap |
| Orchestration | LangGraph | Build stateful multi-step agent graph | Native cycles, conditional edges, typed state |
| UI | Streamlit | User-facing interface | Fast to build, clean Python, easy to demo |
| Language | Python 3.11+ | Primary development language | Standard for ML/AI projects |



> **LangChain scope in this project**
> - PDF parsing, section detection, metadata tagging → custom Python (LangChain has no financial doc awareness)
> - Chunking → LangChain RecursiveCharacterTextSplitter (after sections are detected)
> - Table chunking → custom Python (LangChain splitters split mid-row)
> - Embeddings → LangChain HuggingFaceEmbeddings wrapper
> - Vector store → LangChain Chroma wrapper
> - LLM calls → LangChain ChatNVIDIA
> - Prompts → LangChain ChatPromptTemplate
> - Graph, state, nodes, cycles → LangGraph directly




# Chapter 04 — Environment Setup

## 4.1 Python and Virtual Environment

    python --version               # must be 3.11+
    python -m venv venv
    source venv/bin/activate       # Mac/Linux
    venv\Scripts\activate          # Windows

## 4.2 requirements.txt

    # PDF Parsing
    pymupdf==1.24.0
    
    # LangChain core
    langchain==0.2.0
    langchain-core==0.2.0
    langchain-community==0.2.0
    
    # LangChain integrations
    langchain-nvidia-ai-endpoints==0.1.0   # NVIDIA NIM
    langchain-huggingface==0.0.3            # HuggingFace embeddings
    langchain-chroma==0.1.0                 # ChromaDB
    
    # Vector store
    chromadb==0.5.0
    
    # Embedding model
    sentence-transformers==3.0.0
    
    # LangGraph
    langgraph==0.1.0
    
    # UI
    streamlit==1.35.0
    
    # Utilities
    python-dotenv==1.0.0
    tiktoken==0.7.0               # token counting for chunk size
    
    # [PLANNED — NOT YET IMPLEMENTED, v3] Hybrid search
    rank_bm25                     # BM25Okapi sparse retrieval — see Chapter 21

## 4.3 .env File

Create a .env file in the project root. Never commit this to GitHub.

    NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
    NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

## 4.4 config.py

Central config file that all modules import from. Change values here, nowhere else.

**[AS-BUILT — v3]** The block below is the *real* current `config.py`, confirmed against the actual file — not the original v2.0 draft. Two attribute names differ from earlier PRD chapters (see the mismatch table right after), and the Grader model is actually the 70B model despite an inline comment still saying "fast, cheap" (the comment is stale — it was upgraded from 8B at some point and never updated). Always code against the attribute names in this block.

    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    NVIDIA_API_KEY     = os.getenv("NVIDIA_API_KEY")
    NVIDIA_BASE_URL    = os.getenv("NVIDIA_BASE_URL")
    
    EMBEDDING_MODEL    = "sentence-transformers/all-mpnet-base-v2"
    CHUNK_SIZE         = 450   # tokens
    CHUNK_OVERLAP      = 50    # tokens
    TOP_K              = 5     # chunks to retrieve (single-company)
    MAX_RETRY          = 3     # max rewrite/hallucination retries
    CHROMA_PATH        = "./chroma_db"
    COLLECTION_NAME    = "financial_10k"
    
    # Model per node (see Chapter 13) — real attribute names, confirmed
    MODEL_ROUTER       = "meta/llama-3.1-8b-instruct"
    MODEL_GRADER       = "meta/llama-3.1-70b-instruct"   # NOTE: stale comment elsewhere says "fast, cheap" — this is actually the 70B model
    MODEL_GENERATOR    = "meta/llama-3.1-70b-instruct"   # NOT "MODEL_GENERATE" — see mismatch table below
    MODEL_HALLUC       = "meta/llama-3.1-8b-instruct"    # NOT "MODEL_HALLUCINATION" — see mismatch table below
    MODEL_REWRITE      = "meta/llama-3.1-8b-instruct"
    
    # [PLANNED — NOT YET IMPLEMENTED, v3] add for Hybrid Search (Chapter 21):
    # BM25_INDEX_PATH  = "./bm25_index.pkl"

**Attribute-naming mismatch — already caused two `AttributeError` bugs, do not reintroduce:**

| Earlier PRD chapters call it | Real `config.py` attribute |
| --- | --- |
| `MODEL_GENERATE` | `MODEL_GENERATOR` |
| `MODEL_HALLUCINATION` | `MODEL_HALLUC` |

**Multi-company retrieval breadth — also corrected from v2.0's original value:** the `["all"]` (cross-company) branch was originally `k=2` chunks/company (Ch 09.2 below). Real-evidence testing (Chapter 19 / Chapter 21) showed this was too narrow to reliably surface a specific line item across 9 differently-formatted 10-Ks, so it was bumped to **`k=4` chunks/company** (36 total for a 9-company query, not 18). This is a real, accepted cost/latency tradeoff, not a free fix — see Chapter 21 for how Hybrid Search interacts with this same final-k value.


# Chapter 05 — System Architecture

## 5.1 Two Phases

> **Phase 1 — Ingestion (run once)**
> - Load all 10 PDFs with PyMuPDF
> - Extract text page by page
> - Detect SEC Item boundaries with regex
> - Split prose sections with LangChain RecursiveCharacterTextSplitter
> - Split table sections with custom table-aware chunker
> - Tag every chunk with metadata: {company, ticker, year, section, chunk_id}
> - Embed all chunks with all-mpnet-base-v2 (local)
> - Store in ChromaDB



> **Phase 2 — Query (every user question)**
> - User submits natural language question
> - Router node classifies: retrieve / calculate / direct
> - Company extractor identifies which companies are mentioned
> - Retrieve node fetches top-K chunks with metadata filter
> - Grade node checks chunk relevance (yes/no)
> - If not relevant → Rewrite node reformulates → retry Retrieve
> - If relevant → Generate node produces answer
> - Hallucination check node verifies answer is grounded
> - If grounded → return answer. If not → retry or flag user



## 5.2 LangGraph State Object

    from typing import TypedDict, List, Optional
    
    class GraphState(TypedDict):
        question:             str            # original user question
        rewritten_question:   str            # reformulated if retrieval failed
        route:                str            # retrieve / calculate / direct
        companies_mentioned:  List[str]      # extracted company names
        retrieved_chunks:     List[str]      # raw chunks from ChromaDB (from vector search today; from hybrid vector+BM25 once Ch 21 ships)
        chunk_sources:        List[dict]     # metadata for each chunk
        relevant:             str            # yes / no from Grade node
        answer:               str            # generated answer
        grounded:             str            # "grounded" / "not_grounded" from Hallucination — NOT "yes"/"no", see correction below
        retry_count:          int            # prevent infinite loops
        final_answer:         str            # cleaned final output
        error_message:        Optional[str]  # set if system gives up

**[AS-BUILT — v3] `grounded` field correction:** this chapter and Chapter 10.1's edge table originally specified `yes`/`no` for the `grounded` field. The actual Hallucination Check prompt (Ch 16.6) and its parser (Ch 11.2) both specify `"grounded"`/`"not_grounded"` string values instead — that was treated as the authoritative pair (prompt/parser over this chapter's stale text), and the real code's edge functions all check `== "grounded"`, never `== "yes"`. See Ch 10.1 for the corrected edge table.

**[AS-BUILT — v3] Two extra nodes not shown in the Ch 10 node table:** `grade_exhausted_warning_node` and `hallucination_exhausted_node` also exist in the real graph. LangGraph conditional edge functions can only route to the next node — they cannot write to state — so each "retries exhausted" behavior in Ch 12.1 needed its own tiny node to actually write the warning/failure message into state before continuing. These are plumbing, not a design change to the retry logic itself.


# Chapter 06 — Data & Ingestion — Updated Chunking Strategy

## 6.1 The Problem with Naive Chunking on Financial Documents

A typical income statement table is 400-450 tokens. At a 450-token chunk size, a naive character-based splitter cuts the table in the middle. You end up with one chunk containing column headers and the first half of rows, and another containing the second half without headers. Neither chunk is useful for retrieval because a number without its column header has no meaning.


## 6.2 Two-Track Chunking Strategy

| Track | Applied To | Method | Key Rule |
| --- | --- | --- | --- |
| Track A — Prose | MD&A, Risk Factors, Business, Notes (text-heavy sections) | LangChain RecursiveCharacterTextSplitter, 450 tokens, 50 overlap | Splits on paragraphs first, then sentences — chunk boundaries must never fall mid-sentence, since a split sentence loses its meaning in both resulting chunks |
| Track B — Tables | Financial Statements (Item 8), quantitative sections | Custom table-aware chunker | Never split a table row. Keep column headers in every chunk. |



## 6.3 Table Detection

PyMuPDF provides page-level text blocks. Tables are detected by looking for lines with 3+ tab-separated or whitespace-aligned numeric columns. When a table is detected, it is extracted separately from surrounding prose.


## 6.4 Table Chunking Rules

- Extract table header row (column names like "2024", "2023", "2022")
- Group rows into chunks that stay under 450 tokens
- Prepend column headers to every chunk from that table
- Prepend company name, year, and table name to every chunk
- Never split a row — if a row pushes over 450 tokens, start a new chunk

## 6.5 Header Prepending — What It Looks Like

Instead of storing a raw table row like this:

    "97,329      96,995     102,962"

Every table chunk is stored like this:

    "Apple Inc. | Income Statement 2024 | Columns: 2024, 2023, 2022"
    "Net income: 97,329 (2024)  96,995 (2023)  102,962 (2022)"
    "Total revenue: 391,035 (2024)  383,285 (2023)  394,328 (2022)"

This makes every chunk self-contained and retrievable even without the surrounding table context.


## 6.6 Section Detection — SEC Item Regex

| SEC Item | Section Name | Chunking Track |
| --- | --- | --- |
| Item 1 | Business | Track A — Prose |
| Item 1A | Risk Factors | Track A — Prose |
| Item 7 | MD&A | Track A — Prose |
| Item 7A | Quantitative Market Risk | Track B — May contain tables |
| Item 8 | Financial Statements | Track B — Tables primary |
| Item 9A | Controls & Procedures | Track A — Prose |



## 6.7 Chunk Metadata Schema

    {
        "company":       "Apple",
        "ticker":        "AAPL",
        "year":          "2024",
        "item_number":   "Item 8",
        "section_name":  "Financial Statements",
        "chunk_type":    "table",          # or "prose"
        "table_name":    "Income Statement",# only for table chunks
        "chunk_id":      "apple_2024_item8_042",
        "page_start":    84,
        "parent_chunk_id": "apple_2024_item8_041"   # optional — see 6.8
    }


## 6.8 Parent-Child Chunk Linking

Some chunks are only fully meaningful alongside a neighboring chunk — most commonly, a Track B table chunk (raw figures) and the Track A prose chunk immediately following it that explains those figures (e.g. a "Research and Development" table row and the paragraph "The growth in R&D expense during 2024 compared to 2023 was driven primarily by increases in headcount-related expenses." that follows it in the source document).

To preserve this relationship without merging the two chunks into one (which would blur Track A/Track B chunking rules), each chunk may carry an optional `parent_chunk_id` field pointing to a related chunk that provides broader context. At retrieval time, when a chunk with a `parent_chunk_id` is returned as a top match, its linked parent chunk is also pulled into context — giving the Generate node both the number and its explanation, even though they were embedded and retrieved as separate chunks.

Linking rule: a chunk's `parent_chunk_id` is set to the `chunk_id` of the nearest preceding chunk in reading order that shares the same `section_name` and `item_number`, when that preceding chunk is of a different `chunk_type` (table linking to prose, or prose linking to the table it discusses). Not every chunk will have a `parent_chunk_id` — it is only set when this kind of cross-chunk relationship is detected during ingestion.


# Chapter 07 — PDF Naming Convention

The ingestion script maps filename to company name and ticker automatically. All PDFs must follow this exact naming convention:


| Filename | Company | Ticker |
| --- | --- | --- |
| apple_10k_2024.pdf | Apple Inc. | AAPL |
| microsoft_10k_2024.pdf | Microsoft Corporation | MSFT |
| amazon_10k_2024.pdf | Amazon.com Inc. | AMZN |
| nvidia_10k_2024.pdf | NVIDIA Corporation | NVDA |
| tesla_10k_2024.pdf | Tesla Inc. | TSLA |
| meta_10k_2024.pdf | Meta Platforms Inc. | META |
| alphabet_10k_2024.pdf | Alphabet Inc. | GOOGL |
| netflix_10k_2024.pdf | Netflix Inc. | NFLX |
| adobe_10k_2024.pdf | Adobe Inc. | ADBE |

~~intel_10k_2024.pdf | Intel Corporation | INTC~~ — **[AS-BUILT — v3] Intel removed from scope, this file is not ingested.**



The ingestion script uses a hardcoded mapping dictionary that derives company name and ticker from the filename prefix. If a PDF does not match a key in the dictionary, it is skipped with a warning.

    COMPANY_MAP = {
        "apple":     {"name": "Apple Inc.",            "ticker": "AAPL"},
        "microsoft": {"name": "Microsoft Corporation", "ticker": "MSFT"},
        "amazon":    {"name": "Amazon.com Inc.",       "ticker": "AMZN"},
        # ... etc
    }


# Chapter 08 — ChromaDB Filtering Design

## 8.1 How Metadata Filtering Works

ChromaDB supports filtering by metadata BEFORE the vector similarity search. This means for a question about Apple, you first filter to only Apple chunks, then run vector search within those. This is much better than filtering after search because:

- You get top-K results from the right company, not top-K across all companies where Apple might only appear twice
- No risk of Microsoft revenue appearing in an Apple query
- Faster search — smaller candidate set

## 8.2 Single Company Query

    results = vectorstore.similarity_search(
        query=question,
        k=5,
        filter={"company": "Apple"}   # pre-filter before vector search
    )

## 8.3 Multi-Company Query

For comparative questions ("Compare Apple and Microsoft"), run two separate retrieval calls and combine results:

    chunks_a = vectorstore.similarity_search(query, k=4, filter={"company":"Apple"})
    chunks_b = vectorstore.similarity_search(query, k=4, filter={"company":"Microsoft"})
    all_chunks = chunks_a + chunks_b   # 8 total chunks, 4 from each

Why 4 each instead of 5 each? Keep total context under control. 8 chunks at 450 tokens each = 3600 tokens of context, well within NIM model limits.


## 8.4 Section Filtering (Optional Enhancement)

For questions specifically about financials, additionally filter by section to improve precision:

    filter={"company": "Apple", "item_number": "Item 8"}  # financials only
    filter={"company": "Apple", "item_number": "Item 1A"} # risk factors only

The Router node can set a section_hint in state when the question is clearly about a specific section. The Retrieve node checks for this hint before building the filter.



# Chapter 09 — Multi-Company Query Handling

## 9.1 Company Extraction

Before retrieval, a company extractor identifies which companies the question is about. This runs as part of the Router node using a simple LLM call:

    prompt: "Extract company names from this question. Return only a JSON array
    of company names from this list: [Apple, Microsoft, Amazon, NVIDIA, Tesla,
    Meta, Alphabet, Netflix, Adobe, Intel]. If no specific company mentioned,
    return ['all']. Return JSON only."
    
    # Example outputs:
    # "What was Apple's revenue?" → ["Apple"]
    # "Compare Apple and Microsoft" → ["Apple", "Microsoft"]
    # "Which company had highest R&D?" → ["all"]

## 9.2 The "all" Case — Cross-Company Questions

When the question asks across all companies (e.g. "Which company had the highest revenue?"), you cannot retrieve from just one company. Strategy (original v2.0 draft):

- ~~Retrieve top-2 chunks from each of the 10 companies = 20 chunks total~~
- ~~This keeps total tokens manageable (20 × 450 = 9000 tokens)~~
- Pass all chunks to Generator with clear company labels in context

**[AS-BUILT — v3] Corrected real values:** it's **9 companies, top-4 chunks each = 36 chunks total** (not 10 companies / top-2 / 20). The bump from `k=2` to `k=4` per company was a deliberate fix (Bug A, Chapter 19) after real-evidence testing showed 2 chunks/company was too narrow to reliably surface one specific line item (e.g. R&D total) across 9 differently-formatted 10-Ks — narrative MD&A paragraphs mentioning a metric kept out-ranking the actual numeric table containing it, purely because vector search has no way to prefer a chunk for being *literally about* the right metric rather than just semantically close to it.

**This is a retrieval-quality problem, not a prompt problem.** The Grade-prompt rewrite (per-company completeness rule, Ch 16.3) was necessary but not sufficient — it can only judge relevance among whatever chunks retrieval actually hands it. If the correct chunk was never retrieved, no prompt fix downstream can recover it. **Chapter 21 (Hybrid Search — BM25 + Vector, confirmed plan, not yet implemented)** is the fix aimed directly at this root cause, and is expected to run alongside — not replace — the `k=4`/company breadth increase above.

## 9.3 Context Structuring for Multi-Company Answers

When passing multi-company chunks to the Generator, structure the context clearly so the LLM does not mix up companies:

    context = """
    === APPLE (AAPL) ===
    [chunk 1 content]
    [chunk 2 content]
    
    === MICROSOFT (MSFT) ===
    [chunk 1 content]
    [chunk 2 content]
    """

| Query Type | Companies Extracted | Retrieval Strategy | Chunks Retrieved |
| --- | --- | --- | --- |
| Single company | ["Apple"] | Filter by company = Apple, top-5 | 5 |
| Two companies | ["Apple","Microsoft"] | Separate retrieval per company, top-4 each | 8 |
| All companies | ["all"] | Top-4 per company × **9** companies **[corrected — was top-2 × 10]** | **36 [corrected — was 20]** |

**[PLANNED — v3, Chapter 21]** Once Hybrid Search ships, "Retrieval Strategy" for every row above becomes "vector search + BM25 search per company, RRF-merged" instead of vector search alone. The final chunk counts in the rightmost column (5 / 4-each / 4-each) are not expected to change as part of that work — only *which* chunks fill those slots.




# Chapter 10 — Node-by-Node Design

| Node | Input from State | Output to State | LLM Call | Model |
| --- | --- | --- | --- | --- |
| Router | question | route, companies_mentioned | Yes | 8B (fast) |
| Retrieve | question/rewritten_question, companies_mentioned | retrieved_chunks, chunk_sources | No (vector search today — see Ch 21 for planned hybrid vector+BM25) | — |
| Grade | question, retrieved_chunks, ~~companies_mentioned resolved to full names~~ **[AS-BUILT: now also takes companies list — Ch 16.3]** | relevant (yes/no) | Yes | ~~8B~~ **70B, see Ch 04.4 note** |
| Rewrite | question, retry_count | rewritten_question, retry_count+1 | Yes | 8B (fast) |
| Generate | question, retrieved_chunks, chunk_sources | answer | Yes | 70B (powerful) |
| Hallucination Check | answer, retrieved_chunks | grounded (**"grounded"/"not_grounded"**, not yes/no) | Yes | 8B (fast) |
| Calculator | retrieved_chunks | answer (computed result) | **[AS-BUILT] Yes — two-step: LLM extracts `{operation, values}` JSON, then Python `tools/calculator.py::compute()` does the arithmetic. See correction below — v2.0 claimed "no LLM call," that was never buildable as written.** | LLM step uses same model as Generator; math step is pure Python |
| Direct Answer | question | answer, final_answer | Yes | 70B |

**[AS-BUILT — v3] Direct Answer route correction:** Direct Answer sets `final_answer` directly inside its own node and **skips Hallucination Check entirely** — there's no `retrieved_chunks` for this route (no document was retrieved), so there is nothing to check groundedness against. The original table above didn't make this explicit.

**[AS-BUILT — v3] Calculator design correction:** v2.0's Technology Stack framing implied Calculator needed no LLM call at all. That's not buildable as written — there is no mechanism for turning unstructured chunk text into clean numbers without one. The real, deliberately-discussed design is two-step: an LLM extraction call produces `{operation, values}` as JSON, then pure Python performs the actual arithmetic (never the LLM). Calculator's answer also **does** go through Hallucination Check afterward (same as Generate) — confirmed explicitly, since the extraction step can still hallucinate a number not actually present in the chunks. Operations were also expanded to **7**, beyond whatever v2.0 implied: `percent_change`, `difference`, `sum`, `average`, `ratio`, `margin`, `max`, `min` — `margin`/`max`/`min` were added after confirming the Ch 15 test list ("highest gross margin?") needed them.



## 10.1 Conditional Edge Logic

**[AS-BUILT — v3]** Every `grounded`/`not_grounded` row below is corrected from the original `yes`/`no` wording (see Ch 5.2 correction) — the real edge functions check `== "grounded"`, never `== "yes"`. The `relevant` field genuinely does use `yes`/`no`, unchanged.

| From Node | Condition | Next Node |
| --- | --- | --- |
| Router | route == "retrieve" or "calculate" | Retrieve |
| Router | route == "direct" | Direct Answer |
| Calculate route | ~~skips Grade~~ **[AS-BUILT] still goes through Grade before Calculator — confirmed, deliberate, unchanged from a "calculate should skip grading" alternative that was considered and rejected]** | Grade |
| Grade | relevant == "yes" | Generate |
| Grade | relevant == "no" AND retry_count < MAX_RETRY | Rewrite |
| Grade | relevant == "no" AND retry_count >= MAX_RETRY | grade_exhausted_warning_node **[AS-BUILT: extra plumbing node, see Ch 5.2]** → Generate (with warning in state) |
| Rewrite | always | Retrieve |
| Hallucination Check | grounded == ~~"yes"~~ **"grounded"** | END |
| Hallucination Check | grounded == ~~"no"~~ **"not_grounded"** AND retry_count < MAX_RETRY | Generate |
| Hallucination Check | grounded == ~~"no"~~ **"not_grounded"** AND retry_count >= MAX_RETRY | hallucination_exhausted_node **[AS-BUILT: extra plumbing node]** → END (flag to user) |




# Chapter 11 — LLM Output Parsing Strategy

LLMs do not always return exactly what you prompt for. Even with strict instructions, a model might return "Yes, the chunks are relevant" instead of just "yes". You must parse defensively.


## 11.1 Grade Node — Parsing yes/no

    def parse_grade(llm_output: str) -> str:
        text = llm_output.strip().lower()
        if text.startswith("yes") or "relevant" in text:
            return "yes"
        if text.startswith("no") or "not relevant" in text:
            return "no"
        # Fallback: if ambiguous, treat as not relevant (safer)
        return "no"

## 11.2 Hallucination Check Node — Parsing grounded/not_grounded

    def parse_hallucination(llm_output: str) -> str:
        text = llm_output.strip().lower()
        if "not_grounded" in text or "not grounded" in text or "hallucin" in text:
            return "not_grounded"
        if "grounded" in text:
            return "grounded"
        # Fallback: if ambiguous, treat as not grounded (safer)
        return "not_grounded"

## 11.3 Router Node — Parsing route classification

    def parse_route(llm_output: str) -> str:
        text = llm_output.strip().lower()
        if "calculate" in text or "math" in text or "compute" in text:
            return "calculate"
        if "direct" in text or "general" in text or "definition" in text:
            return "direct"
        return "retrieve"  # default — safest fallback

## 11.4 Company Extractor — Parsing JSON array

    import json, re
    
    def parse_companies(llm_output: str) -> list:
        try:
            match = re.search(r'\[.*?\]', llm_output, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        return ["all"]  # fallback — retrieve broadly


# Chapter 12 — Retry Limit Behavior

## 12.1 When Retries Are Exhausted

| Scenario | What Happens | Message Shown to User |
| --- | --- | --- |
| Grade says "no" after 3 rewrites | System proceeds to Generate with best available chunks + warning in state | Answer generated from best available context. Retrieval confidence was low — verify with source document. |
| Hallucination check fails 3 times | System returns honest failure message, does not return the hallucinated answer | Could not generate a verified answer for this question. The model was unable to produce an answer grounded in the source documents. Try rephrasing or asking a more specific question. |



## 12.2 retry_count Behavior

- retry_count starts at 0 for every new question
- Incremented by 1 every time Rewrite node runs (rewrite retry)
- Incremented by 1 every time Hallucination Check sends back to Generate
- Shared counter — combined retries across both cycles max at MAX_RETRY (3)
- Resets to 0 when a new question is submitted


# Chapter 13 — NVIDIA NIM Model Selection

Not every node needs the same model. Router, Grader, Rewrite, and Hallucination Check do simple classification tasks. Generator needs genuine reasoning. Using a smaller model for simple tasks preserves NIM credits for where it matters.


| Node | Task Complexity | Model | Reason |
| --- | --- | --- | --- |
| Router | Low — classify into 3 categories | meta/llama-3.1-8b-instruct | Simple classification, speed matters |
| Grade | Low — yes/no decision | meta/llama-3.1-8b-instruct | Binary output, no reasoning needed |
| Rewrite | Medium — rephrase a question | meta/llama-3.1-8b-instruct | Language task, 8B handles well |
| Generate | High — synthesise answer from context | meta/llama-3.1-70b-instruct | Needs strong reasoning and accuracy |
| Hallucination Check | Low — fact check yes/no | meta/llama-3.1-8b-instruct | Verification task, not generation |
| Direct Answer | Medium — explain a concept | meta/llama-3.1-70b-instruct | User-facing answer, quality matters |



> **Credit consumption estimate per query**
> - Router: 1 call × 8B = ~200 tokens
> - Grade: 1 call × 8B = ~600 tokens (chunks in prompt)
> - Generate: 1 call × 70B = ~2000 tokens (main cost)
> - Hallucination: 1 call × 8B = ~2000 tokens (answer + chunks)
> - Total per successful query: ~4800 tokens
> - With 1000 free NIM credits: ~200 successful queries before top-up




# Chapter 14 — Project File Structure

**[AS-BUILT — v3]** This chapter originally described a `financial-rag/` layout with separate `ingestion/`, `retrieval/`, and `prompts/` directories. The real project root is **`RAG_Project/`**, and it never split into those directories — ingestion, prompts, and retrieval logic live inside `graph/nodes.py` and `tools/`, not standalone modules. The tree below is the real, confirmed structure.

    RAG_Project/                     <- project root, ALWAYS run scripts from here
    ├── config.py
    ├── test_pipeline.py             <- manual-chain + full-graph tests
    ├── test_multicompany.py         <- diagnostic script, prints raw chunks + raw LLM extraction JSON
    ├── graph/
    │   ├── state.py                 <- GraphState TypedDict, create_initial_state()
    │   ├── nodes.py                 <- ALL node functions (router, retrieve, grade, rewrite,
    │   │                                generate, hallucination check, calculator, direct answer,
    │   │                                grade_exhausted_warning_node, hallucination_exhausted_node)
    │   ├── edges.py                 <- conditional edge functions
    │   └── graph.py                 <- build_graph(), run_query()
    └── tools/
        ├── output_parsers.py        <- parse_route, parse_companies, parse_grade,
        │                                parse_hallucination, parse_calculation
        ├── calculator.py            <- compute()
        ├── vectorstore.py           <- get_vectorstore()
        └── company_names.py        <- SHORT_TO_FULL, get_all_full_names()
    
    [PLANNED — NOT YET IMPLEMENTED, v3 — Chapter 21]
    └── tools/
        └── bm25_index.py            <- pulls docs from Chroma .get(), builds/caches
                                         BM25Okapi index, returns ranked BM25 results

**Known import gotcha (real, confirmed):** scripts must be run from `RAG_Project/` root (e.g. `python -u test_pipeline.py`), never from inside `graph/`. Running from inside `graph/` causes Python to resolve `graph.state` against the sibling file `graph/graph.py` instead of the package, throwing `'graph' is not a package`. All internal imports are absolute from root (`from graph.state import ...`, `from tools.calculator import ...`).

Note the original `data/pdfs/`, `chroma_db/`, `app.py`, `.env`, `.gitignore`, and `requirements.txt` items from v2.0 are still expected at the project root once ingestion and the UI phase are complete — only the *code layout* above has been corrected; ingestion outputs and top-level config/env files are unaffected by this correction.


# Chapter 15 — Development Phases & Test Questions

**[AS-BUILT — v3] Real status as of this update:** Phases 0–5 are done and validated with real evidence (see Chapter 6 of the handoff doc for specifics — single-company retrieve/calculate, hallucination discrimination, full graph routing, both retry-exhaustion paths, and two-company retrieve/calculate are all confirmed working). **Phase 6 (Multi-Company) is in progress** — Bugs A and B below were found via real testing, fixes have been written for both, but neither has been re-run yet (see Chapter 19). Phase 7 (Calculator) is validated for single-company only; its multi-company issues are the same Bug B. **Phase 8 (Streamlit UI) has not been started at all.**

| Phase | Build | Test Questions to Verify |
| --- | --- | --- |
| Phase 1
Ingestion | pdf_loader → section_detector → chunker_prose + chunker_table → embedder → ChromaDB | Query ChromaDB directly for "Apple revenue 2024" — do correct chunks come back? Check chunk metadata is correct. Check table chunks have header prepended. |
| Phase 2
Basic RAG | Retrieve + Generate (no grading yet) | 1. What was Apple's revenue in 2024?
2. What were NVIDIA's main risk factors? |
| Phase 3
Router | Router node classifying question type | 1. "What does EBITDA mean?" → should route to direct
2. "Apple revenue?" → retrieve
3. "Highest gross margin?" → calculate |
| Phase 4
Grading + Rewrite | Grade + Rewrite + retry loop | Ask a vague question: "tell me about the money" — verify it rewrites and retries rather than generating garbage |
| Phase 5
Hallucination Check | Hallucination check node + retry/flag | Manually inject wrong answer into state, verify system catches it and returns not_grounded |
| Phase 6
Multi-Company | Company extractor + multi-filter retrieval | 1. Compare Apple and Microsoft revenue
2. Which company had highest R&D spend?
3. Compare Tesla and NVIDIA gross margins |
| Phase 7
Calculator | Calculator tool + number extraction | 1. What was Amazon's revenue growth from 2023 to 2024?
2. What is Apple's gross margin percentage? |
| Phase 8
UI | Streamlit interface with source display | Full demo: ask 5 questions, verify sources shown, verify rewrite cycle visible in UI |




# Chapter 16 — Prompt Design

## 16.1 Router Prompt

    System: You are a financial query router. Classify the question into exactly
    one of these three routes:
    "retrieve"  — needs specific facts from a 10-K document
    "calculate" — needs arithmetic performed on financial numbers
    "direct"    — general finance/accounting question, no document needed
    Return ONLY the single word. No explanation. No punctuation.
    
    Human: {question}

## 16.2 Company Extractor Prompt

    System: Extract company names from the question. Return ONLY a JSON array
    containing names from: Apple, Microsoft, Amazon, NVIDIA, Tesla, Meta,
    Alphabet, Netflix, Adobe.
    If no specific company is mentioned, return ["all"].
    Return JSON array only. No other text.

**[AS-BUILT — v3]** Intel dropped from the allowed-names list (9 companies, not 10).
    
    Human: {question}

## 16.3 Grade Prompt

    System: You are a relevance grader for financial documents.
    Given a question and retrieved document chunks, decide if the chunks
    contain sufficient information to answer the question accurately.
    Return ONLY "yes" or "no". No explanation.
    "yes" = chunks contain the specific data needed to answer
    "no"  = chunks are missing key information or are about the wrong topic
    
    Human: Question: {question}
    Chunks: {chunks}

**[AS-BUILT — v3, Bug A fix — written, NOT yet re-validated with real output]** The prompt above graded the whole chunk batch as one yes/no, so a multi-company question could pass Grade even when several companies had zero usable data (real symptom: a 9-company R&D question silently dropped 3 companies, and every remaining number was individually real, so this bug is structurally invisible to Hallucination Check — it's a completeness failure, not a grounding failure). Real, corrected prompt:

    System: You are a relevance grader for financial documents.
    Given a question, a list of companies the question requires data for,
    and retrieved document chunks, decide if the chunks contain sufficient
    information to answer the question for EVERY listed company — not most
    of them. Reason company-by-company before your final line, then return
    ONLY "yes" or "no" as the last line.
    "yes" = the specific figure is present for every listed company
    "no"  = one or more listed companies are missing the needed figure,
            or chunks are about the wrong topic
    
    Human: Question: {question}
    Companies required: {companies}
    Chunks: {chunks}

`grade_node` was updated to resolve `state["companies_mentioned"]` into full company names (via `SHORT_TO_FULL` / `get_all_full_names()`) and pass them into the prompt as `{companies}`. Still needs a real re-run of the "highest R&D spend" test case (Ch 15, Phase 6) before this is considered closed — see Chapter 19.

**Important root-cause note (v3):** this prompt fix is necessary but cannot, by itself, guarantee full coverage — Grade can only judge relevance among whatever chunks Retrieve actually handed it. If a company's real figure was never retrieved in the first place (the actual failure mode observed — see Ch 21), no amount of grading logic recovers it. The retrieval-side fix (breadth `k=2→4`, Ch 09.2) and the planned Hybrid Search (Ch 21) are what address that half of the problem.

## 16.4 Rewrite Prompt

    System: You are a query rewriter for a financial document search system.
    The original question failed to retrieve relevant chunks. Rewrite it
    using more specific financial terminology, SEC section names, or
    alternative phrasing that might match better.
    Return ONLY the rewritten question. No explanation.
    
    Human: Original question: {question}
    Rewrite attempt number: {retry_count}

## 16.5 Generate Prompt

    System: You are a financial analyst assistant. Answer the question using
    ONLY the provided context from 10-K filings. Do not use any information
    from your training data for specific numbers, dates, or financial figures.
    If the context does not contain enough information to answer fully,
    say so clearly rather than guessing.
    Cite which company and section each figure comes from.
    
    Human: Question: {question}
    Context: {context}

## 16.6 Hallucination Check Prompt

    System: You are a hallucination detector for financial answers.
    Given an answer and the source chunks it was generated from, verify
    that every specific number, percentage, date, and financial figure
    in the answer is directly traceable to the source chunks.
    Return ONLY "grounded" or "not_grounded". No explanation.
    "grounded"     = every specific claim in the answer exists in the chunks
    "not_grounded" = answer contains figures not present in the chunks
    
    Human: Answer: {answer}
    Source chunks: {chunks}

## 16.7 Calculator Extraction Prompt — [AS-BUILT, NEW in v3]

Not present in v2.0, since v2.0 assumed Calculator needed no LLM call (corrected in Ch 10). This prompt turns unstructured chunk text into `{operation, values}` JSON for `tools/calculator.py::compute()` to do the actual arithmetic on.

**Bug B fix — written, NOT yet re-validated with real output.** Symptom: comparing Tesla vs. NVIDIA gross margins crashed once (`Cannot compute margin with a base of 0`) and, on a debug re-run, computed a meaningless ratio-of-two-percentages instead of the point gap, while also using Tesla's unlabeled automotive-*segment* margin (16.9%) instead of its consolidated company-wide margin (~17.87%, computable from Tesla's own "Total gross profit" / "Total revenues"). Root cause: the extraction prompt had no rule distinguishing "compare two percentages" from "compare two dollar amounts," and no preference for consolidated/total figures over segment-level ones when a company reports both. Real, corrected prompt logic includes:

- **Operation-selection rule:** percentage-vs-percentage comparisons must use `difference` (the point gap), never `ratio`.
- **Consolidated-figure rule:** when a company reports multiple segment margins instead of one total, prefer the consolidated total (computing via `margin` from stated total revenue/total cost if needed) over any single segment's percentage.
- **Missing-company rule (from Bug A's downstream fix):** if a company's figure genuinely isn't in the retrieved chunks even after the retrieval-breadth fix, extraction should emit a placeholder (`value: 0`, label ending `" (not found in retrieved chunks)"`) instead of silently dropping it. `calculator_node` was updated to exclude these placeholders from `max`/`min` computation (so a `0` never wins by default) while still surfacing them in the final answer text as an explicit caveat.

Still needs a real re-run of the Tesla/NVIDIA margin comparison (Ch 15, Phase 6/7 test list) confirming `difference` is chosen and Tesla's consolidated ~17.87% is used — see Chapter 19.


# Chapter 17 — Configuration

| Parameter | Value | Notes |
| --- | --- | --- |
| Embedding model | sentence-transformers/all-mpnet-base-v2 | Local, 512 token limit, 420MB |
| Chunk size (prose) | 450 tokens | Safely under 512 token model limit |
| Chunk size (tables) | 450 tokens max, row-aware | Never split mid-row |
| Chunk overlap | 50 tokens | Prose only — tables use no overlap |
| Top-K retrieval | 5 (single), 4 per company (two-company or all-9) | **[AS-BUILT — v3 correction]** all-companies case is 4×9=36, not 4×2=8 or 2×10=20 — see Ch 09.2 |
| Max retries | 3 | Shared across rewrite and hallucination cycles |
| Model — Router | meta/llama-3.1-8b-instruct | Fast, cheap — simple classification |
| Model — Grader | ~~meta/llama-3.1-8b-instruct~~ **meta/llama-3.1-70b-instruct** | **[AS-BUILT — v3]** Real config uses the 70B model here; the "fast, cheap" rationale in this row is stale — see Ch 04.4 |
| Model — Generator | meta/llama-3.1-70b-instruct | Powerful — main answer quality |
| Model — Hallucination | meta/llama-3.1-8b-instruct | Fast, cheap — verification task |
| Model — Rewrite | meta/llama-3.1-8b-instruct | Fast, cheap — rephrasing task |
| LLM temperature | 0.0 for all nodes | Deterministic — no creativity in factual Q&A |
| ChromaDB path | ./chroma_db/ | Local persistent directory |
| Collection name | financial_10k | Single collection, all companies |
| **[PLANNED v3]** BM25 index path | ./bm25_index.pkl | Not yet implemented — see Chapter 21 |




# Chapter 18 — Streamlit UI Design

## 18.1 Layout

| Section | Content |
| --- | --- |
| Header | Project title, brief description, company badges |
| Question Input | Text input box, Submit button |
| Pipeline Status | Live status showing which node is currently running (Router → Retrieve → Grade → Generate...) |
| Answer Panel | The final generated answer in a styled box |
| Source Chunks Panel | Expandable section showing which chunks were used, with company and section labels |
| Debug Panel (optional) | Shows route taken, whether rewrite happened, retry count |



## 18.2 Pipeline Status Display

Use Streamlit's st.status() to show live node execution. Users can see the system thinking:

    with st.status("Running pipeline...") as status:
        st.write("Router: classifying question...")
        # run router
        st.write("Retrieve: fetching chunks from ChromaDB...")
        # run retrieval
        st.write("Grade: checking relevance...")
        # run grader
        st.write("Generate: writing answer...")
        # run generator
        status.update(label="Done", state="complete")

## 18.3 Source Display

Every answer shows which chunks were used. This is the key trust feature — users can verify the answer against sources:

    with st.expander("View sources used"):
        for chunk in state["chunk_sources"]:
            st.markdown(f"**{chunk['company']} — {chunk['section_name']}**")
            st.text(chunk["content"][:300] + "...")

## 18.4 Honest Failure Messages

When the system exhausts retries, show a clear and honest message — never show a hallucinated answer:

> **Example UI failure messages**
> - "Answer generated but confidence is low — retrieval was difficult for this question. Please verify the figures in the source document."
> - "Unable to generate a verified answer. The model could not produce a response grounded in the 10-K documents. Try a more specific question."




# Chapter 19 — Known Risks & Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| PDF extraction quality | Tables with complex formatting may not extract cleanly | Test PyMuPDF extraction on all 10 PDFs before ingestion. Flag pages where extraction looks broken. |
| Section detection misses | Some 10-Ks format Item numbers differently | Test regex on all 10 PDFs. Add fallback: if no sections detected, use Track A chunking on entire document. |
| NIM credit exhaustion | Heavy testing burns free credits fast | Use MODEL_ROUTER and MODEL_GRADER with 8B models. Test with local Ollama during development, switch to NIM for demo. |
| Embedding truncation | Chunks over 512 tokens silently cut off | Enforce 450-token hard limit. Add assertion: if chunk > 512 tokens, log warning and truncate explicitly. |
| Table row splitting | Naive splitter cuts a row mid-number | Custom table chunker enforces row boundaries. Tested separately before full ingestion. |
| Multi-company confusion | Numbers from one company appear in another company's answer | Company labels in context (=== APPLE ===). Hallucination check catches cross-company contamination. |
| Infinite retry loops | Rewrite loop never finds relevant chunks | retry_count hard limit of 3. After limit: proceed with best available or return honest failure. |
| LLM parse failures | Router returns unexpected text instead of route keyword | Defensive parsers with fallback values (default to "retrieve" for router, "no" for grader). |
| **[REALIZED — v3] Grade uses "majority passes" instead of "all companies covered" (Bug A)** | Cross-company question (highest R&D spend, 9 companies) silently dropped 3 of 9 companies from the final answer, no indication given to the user. Every number in the wrong answer was individually real, extracted from actual chunks — so Hallucination Check correctly said "grounded" and would again. **This bug is structurally invisible to Hallucination Check**, since it's a completeness failure, not a grounding failure. | Grade prompt rewritten to require per-company reasoning against an explicit `{companies}` list (Ch 16.3); retrieval breadth for the `["all"]` branch bumped `k=2→4`/company (Ch 09.2); Calculator extraction given a missing-company placeholder rule (Ch 16.7). **Fix written, NOT yet re-validated with a real re-run.** |
| **[REALIZED — v3] Calculator picks wrong operation / wrong figure for percentage comparisons (Bug B)** | Comparing Tesla vs. NVIDIA gross margins crashed once (`Cannot compute margin with a base of 0`); on re-run, computed a meaningless ratio of two percentages instead of the point gap, and used Tesla's unlabeled automotive-segment margin (16.9%) instead of its consolidated company-wide margin (~17.87%). | Extraction prompt given an explicit operation-selection rule (percentage-vs-percentage → `difference`, never `ratio`) and a consolidated-figure preference rule (Ch 16.7). **Fix written, NOT yet re-validated with a real re-run.** |
| **[REALIZED — v3, root cause] Vector-search-only retrieval misses chunks that are topically adjacent but not the actual answer** | Narrative MD&A paragraphs mentioning a metric (e.g. "R&D expenses increased $2.3B") consistently out-scored the actual numeric table containing the metric's real total, for 6 of 9 companies on the R&D cross-company question — even after the `k=2→4` breadth fix and multiple rewrite attempts. Vector search has no notion of a chunk being *literally about* the right metric vs. merely *mentioning* it — this is the deeper cause underlying Bug A, and prompt-only fixes (Grade, Calculator) cannot resolve it since the correct chunk is sometimes never retrieved in the first place. | **[PLANNED — NOT YET IMPLEMENTED, v3]** Hybrid search: run vector search (existing) and BM25/sparse search (new) per query, merge rankings via Reciprocal Rank Fusion (RRF). See **Chapter 21** for the full confirmed plan, judgment calls, and file-level task list. |




# Chapter 20 — Resume & Portfolio Value

| Skill Demonstrated | Where in This Project |
| --- | --- |
| Adaptive RAG pipeline | Full ingestion → grading → rewriting → generation → hallucination detection |
| LangGraph stateful agents | Multi-node graph with cycles, conditional edges, typed state |
| LLM prompt engineering | 6 different prompts for different tasks, all with defensive output parsing |
| Vector database design | ChromaDB with metadata filtering, multi-company retrieval strategy |
| Embedding model selection | Local sentence-transformers, chunk/model alignment, token limit awareness |
| PDF processing at scale | PyMuPDF across 10 large documents, table detection, section parsing |
| Failure mode awareness | Grading, rewriting, hallucination detection — handles unhappy paths |
| Production system thinking | Retry limits, honest error messages, deterministic math tool, config file |
| Financial domain knowledge | SEC 10-K structure, Item numbers, table-aware chunking, financial metrics |
| Model cost optimisation | Different models per node — 8B for classification, 70B for generation only |




# Chapter 21 — Hybrid Search: BM25 + Vector (RRF) — NEW in v3

**Status: CONFIRMED PLAN. Zero code has been written or run for this yet.** Everything in this chapter is "agreed direction, next thing to build," not "already done." This chapter supersedes an earlier, simpler idea (naive keyword-overlap counting used to re-rank a wider vector-search candidate pool) that was designed and discussed in chat but never saved, tested, or applied — that approach should be considered abandoned, and this hybrid design replaces it entirely rather than adding to it.

## 21.1 Why This Exists

`retrieve_node` currently uses pure vector (embedding) similarity search only. Real-evidence testing (Ch 19, Bug A root-cause row) showed this misses relevant chunks even after retrieval breadth increased (2→4 chunks/company) and multiple rewrite attempts — specifically, narrative MD&A paragraphs about a metric kept out-scoring the actual numeric table containing that metric's total value, for 6 of 9 companies on the "highest R&D spend" cross-company question. Vector search has no way to specifically prefer a chunk because its content/table is *literally* about the right metric — it only knows semantic closeness. **This is the real root cause of Bug A** — updating the Grade and Calculator prompts (Ch 16.3, 16.7) was necessary but not sufficient, since those prompts can only work with whatever chunks retrieval actually surfaces.

## 21.2 The Approach: Hybrid Search via Reciprocal Rank Fusion (RRF)

Run two independent retrieval methods per query, then merge their rankings — this is the industry-standard pattern for combining semantic and keyword-based retrieval, not something specific to this project.

- **Dense/vector search** (already built): good at semantic matches, synonyms, paraphrases.
- **Sparse/BM25 search** (to be built): good at exact term matches, rewarding chunks where the query's important words are concentrated — adjusted for chunk length and how rare those words are across the whole corpus (inverse document frequency). This is exactly the "chunk is *about* R&D vs. chunk merely *mentions* R&D" distinction vector search is missing.

**Merge method — Reciprocal Rank Fusion:**

    RRF_score(chunk) = 1/(k + rank_in_vector_results) + 1/(k + rank_in_bm25_results)

`k` conventionally = 60. RRF combines by *rank position*, not raw score — necessary because vector similarity (~0–1) and BM25 scores (unbounded, corpus-size-dependent) are on incomparable scales and can't be averaged directly. A chunk both methods rank highly gets a strong combined score; a chunk only one method surfaces still counts, just less.

## 21.3 Confirmed Judgment Calls (do not re-litigate — deliberately decided)

1. **Index scope: ONE global BM25 index across all 9 companies**, not 9 per-company indexes. BM25's IDF (word-rarity) statistic needs a reasonably large, varied corpus to be meaningful — computing "how rare is the word 'research'" against only one company's few hundred chunks is a much weaker statistical basis than computing it across the full 9-company collection. Filtering to a specific company happens *after* scoring, via metadata, mirroring how Chroma's `filter={"company": ...}` already works (Ch 08).
2. **Caching: build once, cache to disk** (proposed path: `./bm25_index.pkl`, alongside `CHROMA_PATH` in `config.py`). Load from cache if present; do not rebuild every run. **Known limitation, accepted deliberately:** this will NOT automatically pick up newly-ingested chunks if the corpus grows later (e.g. adding a 10th company) — the cache file would need to be manually deleted to force a rebuild. Judged acceptable for a fixed 9-company portfolio project scope.
3. **Corpus source: pull all documents directly out of Chroma** via its `.get()` method (returns the full stored collection without needing a similarity query), rather than rebuilding from the original ingestion script/raw source. This guarantees the BM25 index and the vector index describe exactly the same chunks, since both are sourced from the same already-ingested collection.

## 21.4 What Needs to Be Built (not yet built — task list for next session)

| File | Change | Status |
| --- | --- | --- |
| `tools/bm25_index.py` | **New file.** Pull all docs+metadata from Chroma via `.get()`; a simple tokenizer (lowercase, strip punctuation, split whitespace) applied consistently at index-build and query time; BM25 index construction via `rank_bm25`'s `BM25Okapi` class; disk caching (pickle) with load-if-exists/build-if-missing logic; a query function that scores the full corpus, filters to a given company via metadata, and returns a ranked list | Not started |
| `graph/nodes.py` | `retrieve_node` rewritten to: run vector search (existing) AND BM25 search (new) per company branch, then RRF-merge the two rankings per company, then truncate to the existing final chunk counts (4/company for `["all"]` and multi-company branches, 5 for single-company — unchanged from current values, Ch 09.2/17) | Not started — **replaces** the current pure-vector-search body of `retrieve_node`, not additive |
| `config.py` | One new line: `BM25_INDEX_PATH = "./bm25_index.pkl"` (or similar) | Not started |
| Local environment | `pip install rank_bm25` in the project venv (Ch 04.2) | Not started |

## 21.5 Open Items for Whoever Picks This Up Next

- Confirm the exact shape of documents returned by Chroma's `.get()` (field names for text content vs. metadata) before writing `tools/bm25_index.py` — don't assume, verify against the real return value first, consistent with this project's established practice (Ch 8 of the handoff doc) of confirming real code/output before writing fixes.
- After implementation, re-test using the same R&D cross-company question from Ch 19 (Bug A) as the validation case — that's the real-evidence benchmark this whole change exists to fix. Compare final per-company chunk coverage against the "only 3 of 9 companies had real totals" result already on record.
- Consider whether the same RRF final-k values (4/5 per company) are still right once retrieval quality improves, or whether they can be reduced now that relevant chunks are more reliably surfaced — not decided, worth revisiting only after real evidence from the above re-test.
- Once Hybrid Search is implemented and validated, Bugs A and B's prompt-level fixes (Ch 16.3, 16.7) should be re-tested *together with* the new retrieval path, not in isolation — a prompt fix validated against vector-only retrieval may behave differently once the candidate chunk set itself changes.

## 21.6 Implementation Correction — Tokenizer Abbreviation Normalization

**[AS-BUILT correction, found during real implementation]** The original plan (21.4) described a simple tokenizer (lowercase → strip punctuation → split). Real testing showed this is insufficient: `re.sub(r"[^\w\s]", " ", ...)` strips `&`, so `"R&D"` tokenizes to `['r', 'd']` — two single-character tokens with near-zero BM25 IDF weight — while the corpus tables spell it out as `"Research and development"` → `['research', 'and', 'development']`. Zero overlap; the target chunk scores near-zero. The same bug applies to SG&A (`['sg', 'a']` vs `['selling', 'general', 'and', 'administrative']`). Fix applied: financial abbreviations are expanded to their spelled-out forms **before** punctuation stripping, using a priority-ordered substitution list (`R&D` → `research and development`, `SG&A` → `selling general and administrative`, then a generic `&amp;`/`&` → `and` fallback), applied identically at both corpus-build time and query time. This is a vocabulary normalization, not a query-rewriting hack — the BM25 IDF is computed over the expanded tokens, which is the correct behavior. After this fix and a full index rebuild, `aapl_2024_item8_table_085_000` (Apple's Item 8 R&D row, $31,370M) ranked **Rank 3** for the query "R&D expenses" and the Tesla SG&A income-statement chunk ranked **Rank 5** for "SG&A expenses" — both previously invisible to BM25.


End of Document — v3.0



