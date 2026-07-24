"""
STEP 4 BENCHMARK — "Which company had the highest R&D spend in 2024?"
Runs the full pipeline (real LLM calls: route → retrieve → grade → calculate
→ hallucination check) and produces per-company chunk-coverage evidence.

Two sections:
  A. Pre-LLM chunk inspection — what did retrieve_node actually return per
     company? Does each company have ≥1 chunk containing a real R&D total
     figure (a table row with "research and development" + a dollar amount)?
     This is the ground-truth retrieval audit, independent of what the LLM
     later does with those chunks.
  B. Full pipeline run — state fields (route, retry_count, relevant, grounded,
     error_message) and the final_answer. Shows whether the pipeline succeeded
     end-to-end, not just whether retrieval improved.

Run from RAG_Project/ root: python -u scratch_step4_benchmark.py
"""
import sys, os, re
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from graph.state import create_initial_state
from graph.nodes import retrieve_node, router_node
from graph.graph import run_query
from tools.company_names import get_all_full_names

QUESTION = "Which company had the highest R&D spend in 2024?"

# Pattern for a financial dollar amount: optional $, digits, commas, optional decimal
# e.g. "$31,370" or "29,915" or "$ 31,370"
DOLLAR_PATTERN = re.compile(r'\$?\s*[\d,]{4,}(?:\.\d+)?')

# "Research and development" spelled out or abbreviated — post-Docling HTML
RD_PATTERN = re.compile(
    r'research\s+and\s+development|r\s*&\s*(?:amp\s*;\s*)?d',
    re.IGNORECASE
)


def has_rd_figure(text: str) -> bool:
    """
    Returns True if the chunk text contains BOTH a spelled-out / abbreviated
    R&D mention AND a dollar-amount-style number.  This is a conservative
    proxy for 'this chunk contains an actual R&D total'.
    """
    return bool(RD_PATTERN.search(text)) and bool(DOLLAR_PATTERN.search(text))


# ---------------------------------------------------------------------------
# SECTION A — Retrieve chunks and audit per-company coverage
# ---------------------------------------------------------------------------
print("=" * 70)
print("SECTION A — Per-company retrieval coverage (pre-LLM)")
print(f"Question: {QUESTION!r}")
print("=" * 70)

# Run router first so companies_mentioned is set correctly
init_state = create_initial_state(QUESTION)
router_result = router_node(init_state)
init_state.update(router_result)

print(f"\nRoute:             {init_state['route']!r}")
print(f"Companies:         {init_state['companies_mentioned']!r}")

retrieve_result = retrieve_node(init_state)
chunks  = retrieve_result["retrieved_chunks"]
sources = retrieve_result["chunk_sources"]

print(f"\nTotal chunks retrieved: {len(chunks)}")
print(f"Expected:               36 (4 per company × 9)\n")

# Group by company
from collections import defaultdict
per_company: dict[str, list] = defaultdict(list)
for text, src in zip(chunks, sources):
    per_company[src.get("company", "UNKNOWN")].append((text, src))

ALL_COMPANIES = get_all_full_names()
coverage_summary = []

for company in sorted(ALL_COMPANIES):
    co_chunks = per_company.get(company, [])
    n = len(co_chunks)

    # Find which chunks contain an actual R&D figure
    rd_chunks = [(text, src) for text, src in co_chunks if has_rd_figure(text)]
    rd_table_chunks = [(text, src) for text, src in rd_chunks
                       if src.get("chunk_type") == "table"]

    has_real_figure = len(rd_chunks) > 0
    chunk_ids = [src.get("chunk_id", "?") for _, src in co_chunks]

    status = "✓ R&D figure present" if has_real_figure else "✗ NO R&D figure"
    if has_real_figure and not rd_table_chunks:
        status += " (prose only — no table)"

    coverage_summary.append({
        "company": company,
        "n_chunks": n,
        "has_figure": has_real_figure,
        "has_table_figure": len(rd_table_chunks) > 0,
        "rd_chunk_ids": [src.get("chunk_id") for _, src in rd_chunks],
        "all_chunk_ids": chunk_ids,
    })

    print(f"  {company}")
    print(f"    Chunks retrieved: {n}  |  {status}")
    print(f"    All chunk_ids: {chunk_ids}")
    if rd_chunks:
        for text, src in rd_chunks:
            # Show the R&D line from the chunk
            lines = text.replace("\n", " ")
            # Find the line(s) containing R&D amounts
            rd_lines = [l.strip() for l in text.split("\n")
                        if RD_PATTERN.search(l) and DOLLAR_PATTERN.search(l)]
            print(f"    R&D chunk: {src.get('chunk_id')}  "
                  f"type={src.get('chunk_type')}")
            for rl in rd_lines[:3]:
                print(f"      → {rl[:120]!r}")
    print()

# Summary counts
n_with_figure = sum(1 for c in coverage_summary if c["has_figure"])
n_with_table  = sum(1 for c in coverage_summary if c["has_table_figure"])
print(f"{'─' * 70}")
print(f"Companies with an R&D figure in retrieved chunks: {n_with_figure}/9")
print(f"Companies with an R&D TABLE chunk:               {n_with_table}/9")
print(f"Companies missing any R&D figure:                {9 - n_with_figure}/9")
print()

# Flag composition observations
print("Composition notes:")
for c in coverage_summary:
    if c["has_figure"] and not c["has_table_figure"]:
        print(f"  ⚠ {c['company']}: has R&D figure but ONLY in prose chunks "
              f"(no table) — LLM must extract from narrative text")
    if c["n_chunks"] == 0:
        print(f"  ✗ {c['company']}: zero chunks retrieved — company filter may "
              f"have failed")
    if c["n_chunks"] < 4:
        print(f"  ⚠ {c['company']}: only {c['n_chunks']}/4 expected chunks retrieved")

# ---------------------------------------------------------------------------
# SECTION B — Full pipeline run (real LLM calls)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SECTION B — Full pipeline (real LLM calls)")
print(f"Question: {QUESTION!r}")
print("=" * 70)
print("Running... (this may take 30–90 seconds for 9-company calculate path)")
print()

final_state = run_query(QUESTION)

print(f"\nroute:              {final_state.get('route')!r}")
print(f"companies_mentioned:{final_state.get('companies_mentioned')!r}")
print(f"rewritten_question: {final_state.get('rewritten_question')!r}")
print(f"relevant:           {final_state.get('relevant')!r}")
print(f"retry_count:        {final_state.get('retry_count')}")
print(f"grounded:           {final_state.get('grounded')!r}")
print(f"error_message:      {final_state.get('error_message')!r}")

print(f"\n{'─' * 70}")
print(f"final_answer:\n{final_state.get('final_answer', '(none)')}")
print(f"{'─' * 70}")

# ---------------------------------------------------------------------------
# SIDE-BY-SIDE COMPARISON vs. KNOWN-BAD BASELINE
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("COMPARISON vs. KNOWN-BAD BASELINE (pre-hybrid-search)")
print("=" * 70)
print("Baseline (pure vector search, pre-Step 3):")
print("  Only 3 of 9 companies had real R&D totals in retrieved chunks.")
print("  6 of 9 companies were missing — narrative MD&A paragraphs")
print("  out-scored their actual income statement table chunks.")
print()
print(f"After hybrid BM25+vector RRF search (this run):")
print(f"  {n_with_figure}/9 companies have at least one R&D figure chunk")
print(f"  {n_with_table}/9 companies have an R&D TABLE chunk (not just prose)")
print(f"  {9 - n_with_figure}/9 still missing")
print()
missing = [c["company"] for c in coverage_summary if not c["has_figure"]]
if missing:
    print(f"Still missing: {missing}")
    print("  → These are candidates for further retrieval investigation.")
    print("  → Do NOT touch grade/calculator prompts in this step.")
else:
    print("All 9 companies now have at least one R&D figure chunk. ✓")
