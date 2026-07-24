"""
Diagnostic script — full-pipeline test for the R&D question only.
Runs through run_query() (the real compiled graph, including the rewrite
retry loop) rather than manually chaining nodes, so we see whether
rewrite_node can actually recover missing per-company coverage.

Run from project root: python -u test_multicompany.py
"""
from graph.graph import run_query


def print_full_chunks(state):
    chunks = state.get("retrieved_chunks", [])
    sources = state.get("chunk_sources", [])

    print(f"\nnum chunks in FINAL retrieval: {len(chunks)}")
    companies_present = [c.get('company') for c in sources]
    print("companies in final chunk_sources (with counts):")
    for company in sorted(set(companies_present)):
        print(f"    {company}: {companies_present.count(company)} chunk(s)")

    print("\n--- full retrieved_chunks + chunk_sources (final attempt) ---")
    for i, (chunk, source) in enumerate(zip(chunks, sources)):
        print(f"\n[{i}] {source.get('company')} | {source.get('section_name')} | {source.get('table_name')}")
        print(f"    {chunk}")


print("=" * 70)
print("TEST 10: Full graph — 'Which company had the highest R&D spend?' (with rewrite loop)")
print("=" * 70)

final_state10 = run_query("Which company had the highest R&D spend in 2024?")

print(f"\nroute: {final_state10.get('route')}")
print(f"companies_mentioned: {final_state10.get('companies_mentioned')}")
print(f"original question: {final_state10.get('question')}")
print(f"rewritten_question: {final_state10.get('rewritten_question')}")
print(f"relevant: {final_state10.get('relevant')}")
print(f"retry_count: {final_state10.get('retry_count')}")
print(f"grounded: {final_state10.get('grounded')}")
print(f"error_message: {final_state10.get('error_message')}")

print_full_chunks(final_state10)

print(f"\nfinal_answer: {final_state10.get('final_answer')}")