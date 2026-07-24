# Financial RAG Project — Handoff Context v2

*Last updated: 2026-07-24 (post-BM25 hybrid search + diagnostic pass)*

---

## Project Overview

A RAG (Retrieval-Augmented Generation) system over 10-K filings for 9 companies:
Apple, Microsoft, Amazon, NVIDIA, Tesla, Meta, Alphabet, Netflix, Adobe.
Full graph: router → retrieve → grade → rewrite → generate/calculate → hallucination check.

---

## Current Architecture

### Retrieval — Hybrid BM25 + Vector (RRF)
- **Vector store**: Chroma (`./chroma_db/`, 3 063 chunks)
- **BM25 index**: `./bm25_index.pkl` (built once from Chroma, cached to disk)
  - Built by `tools/bm25_index.py` — tokenizer applies priority substitutions
    (`R&D` → `research and development`, `SG&A` → `selling general and administrative`)
    *before* punctuation stripping. Critical: prevents abbreviations collapsing to
    single-char tokens that score nothing against spelled-out corpus text.
  - Global index (all 9 companies); company filter applied post-scoring via metadata.
- **Merge**: Reciprocal Rank Fusion — `RRF(chunk) = 1/(60+rank_vector) + 1/(60+rank_bm25)`.
  Implemented in `_rrf_merge()` in `graph/nodes.py → retrieve_node`.
- **Chunk budget**: 5 chunks per single-company question; 4 per company for
  multi-company questions (unchanged from original design).

### Graph nodes (`graph/nodes.py`)
- `router_node` — classifies route (retrieve / calculate / direct) and extracts companies
- `retrieve_node` — hybrid BM25+vector per-company branch, RRF merge
- `grade_node` — multi-company completeness check (requires every listed company's
  specific figure to be present; YoY-change figures do NOT satisfy a "total" requirement)
- `rewrite_node` — rewrites question toward standard financial terminology on grade=no
- `generate_node` / `calculator_node` — LLM answer generation
- `hallucination_check_node` — grounding verification
- `grade_exhausted_warning_node` — fires when grade retries exhausted; writes
  `state["error_message"]` but does NOT propagate to `final_answer` (see known gap below)

---

## Validated State (as of 2026-07-24 diagnostic pass)

### Fully Working — single-company and two-company
| Question type | Route | Result |
|---|---|---|
| Single-company retrieve (e.g. Apple R&D) | retrieve | ✅ Correct, 0 retries |
| Single-company calculate (e.g. Apple gross margin) | calculate | ✅ 46.21% exact, 0 retries |
| Two-company compare/calculate (e.g. Apple vs MSFT R&D) | calculate | ✅ Both correct figures |
| Two-company calculate (Tesla vs NVIDIA gross margin) | calculate | ✅ **Bug B confirmed fixed** |

### Partially Working — 3+ company scale
| Issue | Root cause | Status |
|---|---|---|
| Chunk-budget competition | At ≥3 companies, some companies' 4-slot allocation fills with prose/adjacent tables instead of the optimal figure-bearing table chunk | Known gap — no fix decided yet |
| grade_node rejections at multi-company scale | Correct behavior: grade is rejecting because the specific total figure really isn't in the slots — it's the retrieval gap, not a grade miscalibration | By design |
| Google/Alphabet router fix | `router_node` now correctly maps "Google" → "Alphabet Inc." (see below) | Fixed today |

### Known Bugs / Gaps (logged, not yet fixed)
| # | Description | Location | Priority |
|---|---|---|---|
| Bug C (grade warning visibility) | `grade_exhausted_warning_node` writes `error_message` but it's never injected into `final_answer`. User sees partial answer with no disclosure that grading failed. | `graph/nodes.py:292`, `graph/graph.py:59-63` | Deferred |
| Chunk-budget competition | At 3+ company scale, the correct figure-bearing chunk may not make it into a company's 4-slot allocation when competing against prose chunks with similar RRF scores. Options: raise k, dynamic slot allocation. Not yet decided. | `retrieve_node` | Deferred |
| Adobe R&D coverage | Adobe's 4 slots consistently return equity/cashflow tables and R&D tax-capitalization prose — never the actual R&D expense line | Retrieval | Deferred |

---

## Router Fix — Google/Alphabet (fixed 2026-07-24)

**Bug**: `company_extraction_prompt` listed only `"Alphabet"` as the accepted name;
"Google" was absent. When a user wrote "Google", the LLM dropped it silently.

**Fix (2 files)**:
1. `tools/company_names.py` — added `"Google": "Alphabet Inc."` to `SHORT_TO_FULL`
2. `graph/nodes.py` — added alias instruction and example to `company_extraction_prompt`:
   > *"Alias: 'Google' and 'Alphabet' both refer to the same company..."*

**Verified**: Q5 ("Which of Apple, Microsoft, Amazon, and Google had the highest R&D?")
ran 3 consecutive times after the fix — `companies_mentioned` returned
`['Apple', 'Microsoft', 'Amazon', 'Google']` → `['Apple Inc.', 'Microsoft Corporation',
'Amazon.com Inc.', 'Alphabet Inc.']` all 3 times. ✓

---

## Bug B — Tesla vs NVIDIA Gross Margin (confirmed fixed 2026-07-24)

The hybrid RRF retrieval now surfaces NVIDIA's income statement chunk
(`nvda_2024_item15_table_044_000`, revenue $60,922M, GP $44,301M) in the top slots.
Q4 ("Compare Tesla and NVIDIA gross margins") returned:
- NVIDIA: 72.7% ✓  |  Tesla: 16.9% ✓
Zero retries, grounded. Previously produced wrong figures due to pure vector retrieval
choosing non-income-statement chunks.

---

## Tokenizer (critical implementation note)

`tools/bm25_index.py` tokenizer applies substitutions **before** stripping punctuation:
```python
SUBSTITUTIONS = [
    (re.compile(r'\bR&D\b', re.IGNORECASE),              'research and development'),
    (re.compile(r'\bSG&A\b', re.IGNORECASE),             'selling general and administrative'),
    (re.compile(r'\bCapEx\b', re.IGNORECASE),            'capital expenditures'),
    (re.compile(r'\bEBITDA\b', re.IGNORECASE),           'earnings before interest taxes depreciation amortization'),
    (re.compile(r'&',                                     0), 'and'),
]
```
Without this, `R&D` → `['r', 'd']` (two single-char tokens) which score nothing
against corpus text like "research and development expense". Validated in Step 2
(see `bm25_hybrid_search_addendum.md` Ch 21.6).

---

## Next Steps (undecided — scope with user before implementing)

1. **Chunk-budget competition**: Decide whether to raise per-company k (currently 4)
   or implement dynamic slot allocation based on query-company relevance scores.
   Not implementing until direction is confirmed.
2. **Bug C (grade warning visibility)**: If decided, inject `error_message` into
   `final_answer` suffix when `grade_exhausted_warning_node` fires.
3. **9-company "all" path**: Still fails end-to-end (grade rejects all 36-chunk batches
   because some companies — notably Adobe — have no R&D figure in any of their 4 slots).
   Root cause is retrieval coverage, not grade_node.

---

## Commit History Note

The commit "BM25 Implemented" (2433132) was pushed before the Google routing bug
was found and before the multi-company diagnostic pass concluded. The true state
at that commit: BM25 hybrid retrieval working, single/two-company questions correct,
Bug B fixed — but router Google/Alphabet alias was broken, and multi-company
retrieval coverage gaps were not yet characterized. The follow-up commit
(post today's fixes) accurately reflects the completed + validated state.

---

## Key Files

| File | Role |
|---|---|
| `tools/bm25_index.py` | BM25 index build, cache, query; tokenizer |
| `tools/company_names.py` | `SHORT_TO_FULL` mapping (includes Google alias) |
| `graph/nodes.py` | All node implementations + prompts |
| `graph/graph.py` | LangGraph wiring and `run_query()` |
| `graph/edges.py` | Conditional routing logic |
| `graph/state.py` | `GraphState` TypedDict |
| `Financial_RAG_PRD_v3.md` | Full product spec (Ch 21 covers hybrid search) |
| `bm25_hybrid_search_addendum.md` | Tokenizer fix documentation (Ch 21.6) |
| `bm25_index.pkl` | Cached BM25 index (3 063 chunks; rebuild by deleting this file) |
