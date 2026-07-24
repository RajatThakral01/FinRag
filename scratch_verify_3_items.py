"""
THROWAWAY — Check three specific things:
1. Full text of msft_2024_item7_prose_031_000 — does it contain $29,510M R&D?
2. Confirm what grade_exhausted_warning_node wrote to state in Q5 run —
   does it show up in final_answer or only error_message?
3. Print full state keys from Q5 so we can see what's in error_message.

Run from RAG_Project/ root.
"""
import sys, os, re
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tools.vectorstore import get_vectorstore
from graph.graph import run_query

# ---------------------------------------------------------------------------
# 1. Full text of msft_2024_item7_prose_031_000
# ---------------------------------------------------------------------------
print("=" * 70)
print("1. Full text of msft_2024_item7_prose_031_000")
print("=" * 70)

vs = get_vectorstore()
result = vs.get()
chunk_text = None
for doc, meta, cid in zip(result["documents"], result["metadatas"], result["ids"]):
    if cid == "msft_2024_item7_prose_031_000":
        chunk_text = doc
        print(f"  chunk_type: {meta.get('chunk_type')}")
        print(f"  section:    {meta.get('section_name')}")
        print(f"  item:       {meta.get('item_number')}")
        print()
        print("  FULL TEXT:")
        print("  " + "\n  ".join(chunk_text.split("\n")))
        break

if not chunk_text:
    print("  ERROR: chunk not found")

# Check: does the text contain an extractable R&D dollar figure?
DOLLAR_RE = re.compile(r'\$?\s*[\d,]{4,}(?:\.\d+)?')
RD_RE = re.compile(r'research\s+and\s+development|r\s*&\s*d', re.IGNORECASE)

has_rd = bool(RD_RE.search(chunk_text or ""))
has_dollar = bool(DOLLAR_RE.search(chunk_text or ""))
print(f"\n  Contains R&D mention: {has_rd}")
print(f"  Contains dollar-pattern number: {has_dollar}")
if has_rd and has_dollar:
    # Find the specific R&D $ line
    for line in (chunk_text or "").split("\n"):
        if RD_RE.search(line) and DOLLAR_RE.search(line):
            print(f"  R&D + dollar line: {line.strip()!r}")
else:
    print("  → Does NOT contain both R&D mention AND a dollar figure in the same text")
    # Show what dollar-like numbers ARE present
    numbers = DOLLAR_RE.findall(chunk_text or "")
    print(f"  Dollar-pattern numbers found in chunk: {numbers[:10]}")

# ---------------------------------------------------------------------------
# 2. Run Q5 and print full state — including error_message and final_answer
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("2. Q5 full state — grade_exhausted_warning disclosure in final_answer?")
print("=" * 70)
print("Running Q5 through full pipeline (real LLM)...")
sys.stdout.flush()

q5 = "Which of Apple, Microsoft, Amazon, and Google had the highest R&D expense in 2024?"
state = run_query(q5)

print(f"\n  companies_mentioned: {state.get('companies_mentioned')!r}")
print(f"  route:               {state.get('route')!r}")
print(f"  relevant:            {state.get('relevant')!r}")
print(f"  retry_count:         {state.get('retry_count')}")
print(f"  grounded:            {state.get('grounded')!r}")
print()
print(f"  error_message (state.error_message):")
print(f"    {state.get('error_message')!r}")
print()
print(f"  final_answer (what user sees):")
print(f"    {state.get('final_answer')!r}")
print()

# Does final_answer contain the grade warning?
grade_warning = "Answer generated from best available context"
in_final = grade_warning in (state.get("final_answer") or "")
in_error  = grade_warning in (state.get("error_message") or "")
print(f"  Grade warning in final_answer: {in_final}")
print(f"  Grade warning in error_message: {in_error}")
if in_error and not in_final:
    print("  ✗ WARNING IS IN state.error_message BUT NOT VISIBLE IN final_answer")
    print("    → User sees a partial answer with no disclosure that grading failed")
elif in_final:
    print("  ✓ Warning IS visible to user in final_answer")
else:
    print("  Grade warning not found in either field")
