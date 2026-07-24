"""
THROWAWAY — Step 3 sanity check.
Calls retrieve_node directly (no LLM, no full graph) for:
  1. Single-company: "What was Apple's R&D expense in 2024?"
  2. Two-company:    "Compare Apple and Microsoft operating income"

Verifies:
  - No import/runtime errors from the new hybrid retrieve_node
  - Correct chunk counts (5 for single, 4+4=8 for two-company)
  - All chunk sources have non-empty company + chunk_id fields
  - Prints company/section/type of each returned chunk for eyeball check

Run from RAG_Project/ root.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from graph.state import create_initial_state
from graph.nodes import retrieve_node

TESTS = [
    {
        "label":     "Single-company: Apple R&D",
        "question":  "What was Apple's R&D expense in 2024?",
        "companies": ["Apple"],
        "expected_chunks": 5,
    },
    {
        "label":     "Two-company: Apple + Microsoft operating income",
        "question":  "Compare Apple and Microsoft operating income",
        "companies": ["Apple", "Microsoft"],
        "expected_chunks": 8,   # 4 per company
    },
]

all_passed = True

for t in TESTS:
    print(f"\n{'=' * 60}")
    print(f"TEST: {t['label']}")
    print(f"  question:  {t['question']!r}")
    print(f"  companies: {t['companies']!r}")
    print(f"{'=' * 60}")

    state = create_initial_state(t["question"])
    state["companies_mentioned"] = t["companies"]

    result = retrieve_node(state)
    chunks  = result["retrieved_chunks"]
    sources = result["chunk_sources"]

    n = len(chunks)
    expected = t["expected_chunks"]
    count_ok = (n == expected)
    parallel_ok = (len(chunks) == len(sources))

    print(f"  Chunk count:    {n}  (expected {expected})  {'✓' if count_ok else '✗ WRONG'}")
    print(f"  Parallel check: chunks={len(chunks)} sources={len(sources)}  {'✓' if parallel_ok else '✗ MISMATCH'}")

    # Check all sources have required fields
    missing_fields = []
    for i, src in enumerate(sources):
        for field in ("company", "chunk_id", "section_name", "chunk_type"):
            if not src.get(field):
                missing_fields.append(f"  chunk[{i}] missing {field!r}")
    if missing_fields:
        print(f"  ✗ Missing fields:")
        for m in missing_fields:
            print(f"    {m}")
    else:
        print(f"  All required source fields present  ✓")

    print(f"\n  Returned chunks:")
    for i, (text, src) in enumerate(zip(chunks, sources)):
        snippet = text.replace("\n", " ")[:120]
        print(f"    [{i+1}] company={src.get('company','?')!r}  type={src.get('chunk_type','?')}  "
              f"item={src.get('section_name','?')[:40]!r}")
        print(f"         chunk_id={src.get('chunk_id','?')}")
        print(f"         {snippet!r}")

    if not count_ok or not parallel_ok or missing_fields:
        all_passed = False

print(f"\n{'=' * 60}")
print(f"RESULT: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
print(f"{'=' * 60}")
