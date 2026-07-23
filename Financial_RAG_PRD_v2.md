

**PRODUCT REQUIREMENTS DOCUMENT — v2.0**
# Financial Intelligence RAG System
### Multi-Company 10-K Analysis · Adaptive RAG · LangGraph
****
**Updated with: Chunking Strategy · Environment Setup · Multi-Company Queries**
**LLM Output Parsing · NIM Model Selection · ChromaDB Filtering Design**
**PDF Naming Convention · Retry Behavior · Test Questions · UI Design**




| Field | Detail |
| --- | --- |
| Project Name | Financial Intelligence RAG System |
| Version | 2.0 (updated) |
| Status | Pre-Development — Ready to Build |
| Data | 2024 Annual 10-K Reports — 10 Companies |
| Author | Rajat Thakral |
| Changes from v1 | Chunking strategy, LangChain scope, environment setup, multi-company handling, NIM model selection, output parsing, PDF naming, retry UX, test questions, UI design |




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



# Chapter 01 — Project Overview

This system is a production-grade Retrieval-Augmented Generation (RAG) pipeline built on 2024 Annual Report (10-K) filings from 10 major technology companies. Users ask natural language questions about financial data and receive accurate, grounded, verifiable answers. The pipeline goes beyond basic RAG with relevance grading, query rewriting, hallucination detection, and intelligent routing.


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
| 10 | Intel Corporation | INTC | Semiconductors / PC Hardware |




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

## 4.3 .env File

Create a .env file in the project root. Never commit this to GitHub.

    NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
    NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

## 4.4 config.py

Central config file that all modules import from. Change values here, nowhere else.

    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    NVIDIA_API_KEY     = os.getenv("NVIDIA_API_KEY")
    NVIDIA_BASE_URL    = os.getenv("NVIDIA_BASE_URL")
    
    EMBEDDING_MODEL    = "sentence-transformers/all-mpnet-base-v2"
    CHUNK_SIZE         = 450   # tokens
    CHUNK_OVERLAP      = 50    # tokens
    TOP_K              = 5     # chunks to retrieve
    MAX_RETRY          = 3     # max rewrite retries
    CHROMA_PATH        = "./chroma_db"
    COLLECTION_NAME    = "financial_10k"
    
    # Model per node (see Chapter 13)
    MODEL_ROUTER       = "meta/llama-3.1-8b-instruct"   # fast, cheap
    MODEL_GRADER       = "meta/llama-3.1-8b-instruct"   # fast, cheap
    MODEL_GENERATOR    = "meta/llama-3.1-70b-instruct"  # powerful
    MODEL_HALLUC       = "meta/llama-3.1-8b-instruct"   # fast, cheap
    MODEL_REWRITE      = "meta/llama-3.1-8b-instruct"   # fast, cheap


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
        retrieved_chunks:     List[str]      # raw chunks from ChromaDB
        chunk_sources:        List[dict]     # metadata for each chunk
        relevant:             str            # yes / no from Grade node
        answer:               str            # generated answer
        grounded:             str            # yes / no from Hallucination
        retry_count:          int            # prevent infinite loops
        final_answer:         str            # cleaned final output
        error_message:        Optional[str]  # set if system gives up


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
| intel_10k_2024.pdf | Intel Corporation | INTC |



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

When the question asks across all companies (e.g. "Which company had the highest revenue?"), you cannot retrieve from just one company. Strategy:

- Retrieve top-2 chunks from each of the 10 companies = 20 chunks total
- This keeps total tokens manageable (20 × 450 = 9000 tokens)
- Pass all 20 chunks to Generator with clear company labels in context

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
| All companies | ["all"] | Top-2 per company × 10 companies | 20 |




# Chapter 10 — Node-by-Node Design

| Node | Input from State | Output to State | LLM Call | Model |
| --- | --- | --- | --- | --- |
| Router | question | route, companies_mentioned | Yes | 8B (fast) |
| Retrieve | question/rewritten_question, companies_mentioned | retrieved_chunks, chunk_sources | No (vector search) | — |
| Grade | question, retrieved_chunks | relevant (yes/no) | Yes | 8B (fast) |
| Rewrite | question, retry_count | rewritten_question, retry_count+1 | Yes | 8B (fast) |
| Generate | question, retrieved_chunks, chunk_sources | answer | Yes | 70B (powerful) |
| Hallucination Check | answer, retrieved_chunks | grounded (yes/no) | Yes | 8B (fast) |
| Calculator | retrieved_chunks (numbers extracted) | answer (computed result) | No (Python math) | — |
| Direct Answer | question | answer | Yes | 70B |



## 10.1 Conditional Edge Logic

| From Node | Condition | Next Node |
| --- | --- | --- |
| Router | route == "retrieve" or "calculate" | Retrieve |
| Router | route == "direct" | Direct Answer |
| Grade | relevant == "yes" | Generate |
| Grade | relevant == "no" AND retry_count < MAX_RETRY | Rewrite |
| Grade | relevant == "no" AND retry_count >= MAX_RETRY | Generate (with warning in state) |
| Rewrite | always | Retrieve |
| Hallucination Check | grounded == "yes" | END |
| Hallucination Check | grounded == "no" AND retry_count < MAX_RETRY | Generate |
| Hallucination Check | grounded == "no" AND retry_count >= MAX_RETRY | END (flag to user) |




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

    financial-rag/
    │
    ├── data/
    │   └── pdfs/                       # 10 raw 10-K PDFs (naming: company_10k_2024.pdf)
    │
    ├── ingestion/
    │   ├── pdf_loader.py               # PyMuPDF text extraction per page
    │   ├── section_detector.py          # SEC Item regex detection
    │   ├── table_detector.py            # Identify table vs prose blocks
    │   ├── chunker_prose.py             # LangChain RecursiveCharacterTextSplitter
    │   ├── chunker_table.py             # Custom table-aware chunker
    │   ├── metadata_tagger.py           # Add company/section metadata
    │   └── ingest.py                   # Main script — run once to build ChromaDB
    │
    ├── retrieval/
    │   ├── embedder.py                 # Load all-mpnet-base-v2 via LangChain
    │   ├── retriever.py                # ChromaDB query with metadata filter
    │   └── company_extractor.py         # Extract company names from question
    │
    ├── graph/
    │   ├── state.py                    # GraphState TypedDict
    │   ├── nodes.py                    # All node functions
    │   ├── edges.py                    # Conditional edge functions
    │   └── graph.py                    # Compile LangGraph and run
    │
    ├── prompts/
    │   ├── router_prompt.py
    │   ├── grade_prompt.py
    │   ├── rewrite_prompt.py
    │   ├── generate_prompt.py
    │   └── hallucination_prompt.py
    │
    ├── tools/
    │   ├── calculator.py               # Deterministic Python math
    │   └── output_parsers.py           # parse_grade, parse_route, etc.
    │
    ├── chroma_db/                      # Auto-created on first ingest
    ├── app.py                          # Streamlit UI
    ├── config.py                       # All configuration
    ├── .env                            # API keys (never commit)
    ├── .gitignore                      # include: .env, chroma_db/, venv/
    └── requirements.txt


# Chapter 15 — Development Phases & Test Questions

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
    Alphabet, Netflix, Adobe, Intel.
    If no specific company is mentioned, return ["all"].
    Return JSON array only. No other text.
    
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


# Chapter 17 — Configuration

| Parameter | Value | Notes |
| --- | --- | --- |
| Embedding model | sentence-transformers/all-mpnet-base-v2 | Local, 512 token limit, 420MB |
| Chunk size (prose) | 450 tokens | Safely under 512 token model limit |
| Chunk size (tables) | 450 tokens max, row-aware | Never split mid-row |
| Chunk overlap | 50 tokens | Prose only — tables use no overlap |
| Top-K retrieval | 5 (single), 4 per company (multi) | Multi-company: 4×2=8 or 2×10=20 |
| Max retries | 3 | Shared across rewrite and hallucination cycles |
| Model — Router | meta/llama-3.1-8b-instruct | Fast, cheap — simple classification |
| Model — Grader | meta/llama-3.1-8b-instruct | Fast, cheap — binary output |
| Model — Generator | meta/llama-3.1-70b-instruct | Powerful — main answer quality |
| Model — Hallucination | meta/llama-3.1-8b-instruct | Fast, cheap — verification task |
| Model — Rewrite | meta/llama-3.1-8b-instruct | Fast, cheap — rephrasing task |
| LLM temperature | 0.0 for all nodes | Deterministic — no creativity in factual Q&A |
| ChromaDB path | ./chroma_db/ | Local persistent directory |
| Collection name | financial_10k | Single collection, all companies |




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




End of Document — v2.0



