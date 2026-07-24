"""
DIAGNOSTIC PASS — 5 questions through full pipeline (new hybrid retrieval).
Diagnostic only — no code changes, just evidence gathering.

Run from RAG_Project/ root: python -u scratch_diagnostic_5q.py
"""
import sys, os, re
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from graph.graph import run_query

# ---------------------------------------------------------------------------
# Ground-truth anchors from real 10-K corpus (verified from corpus chunks)
# ---------------------------------------------------------------------------
# Apple R&D 2024:       $31,370M   (aapl_2024_item8_table_085_000)
# Apple gross margin:   $180,683M revenue $391,035M → 46.2%
# Tesla SG&A 2024:      $5,150M    (tsla_2024_item7_table_020_000)
# Tesla gross profit:   ~ $14.7B   (revenue $97.7B, COGS ~$83B)  → ~15-16%
# NVIDIA gross margin:  $44,301M / $60,922M → 72.7%
# Microsoft R&D 2024:   $29,510M   (msft_2024_item7_table_030_000)
# Meta R&D 2024:        $43,873M   (meta_2024_item7_table_024_000)
# Alphabet R&D:         ~$49-50B   (needs corpus confirm — Amazon calls it
#                                   "Technology and infrastructure", not R&D)

TESTS = [
    # -----------------------------------------------------------------------
    # 1. Single-company, retrieve route — Apple R&D
    # -----------------------------------------------------------------------
    {
        "id":       "Q1a",
        "label":    "Single-company RETRIEVE — Apple R&D expense",
        "question": "What was Apple's R&D expense in 2024?",
        "expected": "~$31,370M",
        "notes":    "Ground truth: $31,370M from aapl_2024_item8_table_085_000",
    },
    # -----------------------------------------------------------------------
    # 1b. Single-company, retrieve route — Tesla SG&A (our other validated case)
    # -----------------------------------------------------------------------
    {
        "id":       "Q1b",
        "label":    "Single-company RETRIEVE — Tesla SG&A expense",
        "question": "What was Tesla's SG&A expense in 2024?",
        "expected": "~$5,150M",
        "notes":    "Ground truth: $5,150M from tsla_2024_item7_table_020_000",
    },
    # -----------------------------------------------------------------------
    # 2. Single-company, calculate route — Apple gross margin
    # -----------------------------------------------------------------------
    {
        "id":       "Q2",
        "label":    "Single-company CALCULATE — Apple gross margin",
        "question": "What was Apple's gross margin in 2024?",
        "expected": "~46.2% (gross profit $180,683M / revenue $391,035M)",
        "notes":    "Ground truth from aapl_2024_item8_table_028_000",
    },
    # -----------------------------------------------------------------------
    # 3. Two-company, retrieve/compare route — R&D expenses
    # -----------------------------------------------------------------------
    {
        "id":       "Q3",
        "label":    "Two-company COMPARE — Apple vs Microsoft R&D",
        "question": "Compare Apple and Microsoft's R&D expenses in 2024",
        "expected": "Apple $31,370M vs Microsoft $29,510M",
        "notes":    "Both confirmed from corpus table chunks",
    },
    # -----------------------------------------------------------------------
    # 4. Two-company, calculate route — Bug B case (Tesla vs NVIDIA gross margins)
    # -----------------------------------------------------------------------
    {
        "id":       "Q4",
        "label":    "Two-company CALCULATE — Tesla vs NVIDIA gross margins (Bug B)",
        "question": "Compare Tesla and NVIDIA gross margins in 2024",
        "expected": "Tesla ~15-16%, NVIDIA ~72.7% ($44,301M / $60,922M)",
        "notes":    "Bug B from PRD — check if hybrid retrieval now feeds correct chunks",
    },
    # -----------------------------------------------------------------------
    # 5. 4-company, calculate — R&D comparison (smaller-scale "all" case)
    # -----------------------------------------------------------------------
    {
        "id":       "Q5",
        "label":    "4-company CALCULATE — highest R&D (Apple/MSFT/Amazon/Google)",
        "question": "Which of Apple, Microsoft, Amazon, and Google had the highest R&D expense in 2024?",
        "expected": "Alphabet/Google ~$49B, then Apple $31,370M, Microsoft $29,510M (Amazon calls it 'Technology and infrastructure')",
        "notes":    "Key diagnostic: does grade_node reject this at 16-chunk scale?",
    },
]


def run_test(t: dict):
    print(f"\n{'=' * 70}")
    print(f"[{t['id']}] {t['label']}")
    print(f"Question:  {t['question']!r}")
    print(f"Expected:  {t['expected']}")
    print(f"{'=' * 70}")

    state = run_query(t["question"])

    route     = state.get("route", "?")
    companies = state.get("companies_mentioned", [])
    rewritten = state.get("rewritten_question", "")
    relevant  = state.get("relevant", "?")
    retries   = state.get("retry_count", "?")
    grounded  = state.get("grounded", "?")
    answer    = state.get("final_answer", "(none)")
    chunks    = state.get("retrieved_chunks", [])
    sources   = state.get("chunk_sources", [])

    print(f"\n  route:              {route!r}")
    print(f"  companies_mentioned:{companies!r}")
    print(f"  rewritten_question: {rewritten!r}")
    print(f"  relevant:           {relevant!r}")
    print(f"  retry_count:        {retries}")
    print(f"  grounded:           {grounded!r}")

    print(f"\n  Retrieved chunks ({len(chunks)} total):")
    for i, src in enumerate(sources):
        tag = "TABLE" if src.get("chunk_type") == "table" else "prose"
        print(f"    [{i}] {tag}  {src.get('chunk_id','?')}  "
              f"| {src.get('company','?')} | {src.get('section_name','?')[:45]!r}")

    print(f"\n  final_answer:\n    {answer}")

    print(f"\n  Cross-check: {t['notes']}")

    # Simple auto-check: does the answer contain a plausible number close to expected?
    # Just flags for manual review — not a hard pass/fail
    answer_lower = answer.lower()
    if "could not" in answer_lower or "unable" in answer_lower or "sorry" in answer_lower:
        print(f"  ⚠ Pipeline FAILED to produce an answer")
    else:
        print(f"  ✓ Pipeline produced an answer — review numbers above vs expected")

    print(f"\n  {'─' * 66}")
    sys.stdout.flush()


# Run all tests sequentially
for test in TESTS:
    run_test(test)

print(f"\n{'=' * 70}")
print("DIAGNOSTIC COMPLETE")
print(f"{'=' * 70}")
