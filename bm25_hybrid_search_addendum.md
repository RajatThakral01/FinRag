# Addendum: BM25 Hybrid Search — Confirmed Plan (Not Yet Implemented)

This is a standalone addendum to `project_handoff_context_v2.md`. It documents a plan that was fully discussed and confirmed, but **zero code has been written or run for it yet** — this session ended immediately after the judgment calls below were confirmed. Treat everything in this file as "agreed direction, next thing to build," not "already done."

---

## Why this exists

`retrieve_node` uses pure vector (embedding) similarity search. Real-evidence testing (see main handoff, Section 5) showed this misses relevant chunks even after retrieval-breadth increases (2→4 chunks/company) and multiple rewrite attempts — specifically, narrative MD&A paragraphs about a metric (e.g. "R&D expenses increased $2.3B") kept out-scoring the actual numeric table containing the metric's total value, for 6 of 9 companies on a "highest R&D spend" cross-company question. Vector search has no way to specifically prefer a chunk because its content/table is *literally* about the right metric — it only knows semantic closeness.

An earlier, simpler fix (naive keyword-overlap counting re-ranking a wider vector-search candidate pool) was designed and written in chat but **never saved, tested, or applied** — it should be considered abandoned. The current real `retrieve_node` in the codebase has only the k=4/company breadth fix (validated, see main handoff Section 5), nothing else. This BM25 plan supersedes the abandoned keyword-boost approach entirely; it is not an addition on top of it.

---

## The approach: hybrid search via Reciprocal Rank Fusion (RRF)

Run two independent retrieval methods per query, then merge their rankings — this is the industry-standard pattern for combining semantic and keyword-based retrieval, not something specific to this project.

- **Dense/vector search** (already built): good at semantic matches, synonyms, paraphrases.
- **Sparse/BM25 search** (to be built): good at exact term matches and rewards chunks where the query's important words are concentrated, adjusted for chunk length and how rare those words are across the whole corpus (inverse document frequency). This is exactly the "chunk is *about* R&D vs. chunk merely *mentions* R&D" distinction vector search was missing.

**Merge method — Reciprocal Rank Fusion:**
```
RRF_score(chunk) = 1/(k + rank_in_vector_results) + 1/(k + rank_in_bm25_results)
```
`k` conventionally = 60. RRF combines by *rank position*, not raw score — necessary because vector similarity (~0–1) and BM25 scores (unbounded, corpus-size-dependent) are on incomparable scales and can't be averaged directly. A chunk both methods rank highly gets a strong combined score; a chunk only one method surfaces still counts, just less.

---

## Confirmed judgment calls (do not re-litigate these — they were deliberately decided)

1. **Index scope: ONE global BM25 index across all 9 companies**, not 9 per-company indexes. Reasoning: BM25's IDF (word-rarity) statistic needs a reasonably large, varied corpus to be meaningful — computing "how rare is the word 'research'" against only one company's few hundred chunks is a much weaker statistical basis than computing it across the full 9-company collection. Filtering to a specific company happens *after* scoring, via metadata, mirroring how Chroma's `filter={"company": ...}` already works.
2. **Caching: build once, cache to disk** (proposed path: `./bm25_index.pkl`, alongside the existing `CHROMA_PATH` in `config.py`). Load from cache if present; do not rebuild every run. **Known limitation, accepted deliberately:** this will NOT automatically pick up newly-ingested chunks if the corpus grows later (e.g. adding a 10th company) — the cache file would need to be manually deleted to force a rebuild. Judged acceptable for a fixed 9-company portfolio project scope.
3. **Corpus source: pull all documents directly out of Chroma** via its `.get()` method (which returns the full stored collection without needing a similarity query), rather than rebuilding from the original ingestion script/raw source. This guarantees the BM25 index and the vector index describe exactly the same chunks, since they're both sourced from the same already-ingested collection.

---

## What needs to be built (not yet built — this is the task list for next session)

| File | Change | Status |
|---|---|---|
| `tools/bm25_index.py` | **New file.** Needs: a function to pull all docs+metadata from Chroma via `.get()`; a simple tokenizer (lowercase, strip punctuation, split whitespace) applied consistently at index-build and query time; BM25 index construction (via the `rank_bm25` library, `BM25Okapi` class); disk caching (pickle) with load-if-exists/build-if-missing logic; a query function that scores the full corpus, filters to a given company via metadata, and returns a ranked list | Not started |
| `graph/nodes.py` | `retrieve_node` needs to be rewritten to: run vector search (existing) AND BM25 search (new) per company branch, then RRF-merge the two rankings per company, then truncate to the existing final chunk counts (4/company for `["all"]` and multi-company branches, 5 for single-company — unchanged from current values) | Not started — **replaces** the current pure-vector-search body of `retrieve_node`, not additive |
| `config.py` | One new line: `BM25_INDEX_PATH = "./bm25_index.pkl"` (or similar) — user adds this themselves per their own config management pattern | Not started |
| Local environment | `pip install rank_bm25` in the project venv | Not started |

## Open items for whoever picks this up next

- Need to confirm the exact shape of documents returned by Chroma's `.get()` (field names for text content vs. metadata) before writing `tools/bm25_index.py` — don't assume, verify against the real return value first, consistent with this project's established practice of confirming real code/output before writing fixes.
- After implementation, re-test using the same R&D cross-company question from the main handoff (Section 5) as the validation case — that's the real-evidence benchmark this whole change exists to fix. Compare final per-company chunk coverage against the "only 3 of 9 companies had real totals" result already on record.
- Consider whether the same RRF final-k values (4/5 per company) are still right once retrieval quality improves, or whether they can be reduced now that relevant chunks are more reliably surfaced — not decided, worth revisiting only after real evidence from the above re-test.
