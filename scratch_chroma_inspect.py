"""
THROWAWAY SCRIPT — Step 1 of Hybrid Search implementation.
Inspects the real shape of Chroma .get() for this project's collection.
Delete after confirming output.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tools.vectorstore import get_vectorstore

print("=== Loading vectorstore ===")
vs = get_vectorstore()

# Access the underlying ChromaDB collection directly
collection = vs._collection
print(f"Collection name: {collection.name}")
print(f"Collection count (total chunks): {collection.count()}")

print("\n=== Calling .get() with no args (fetch ALL) ===")
result = vs.get()  # langchain-chroma exposes .get() directly

print(f"Keys in result dict: {list(result.keys())}")

# Inspect the parallel lists
docs_list = result.get("documents", result.get("texts", None))  # handle possible key variants
metas_list = result.get("metadatas", None)
ids_list = result.get("ids", None)

print(f"\nType of 'documents' value: {type(docs_list)}")
print(f"Type of 'metadatas' value: {type(metas_list)}")
print(f"Type of 'ids' value: {type(ids_list)}")

if docs_list is not None:
    print(f"len(documents): {len(docs_list)}")
if metas_list is not None:
    print(f"len(metadatas): {len(metas_list)}")
if ids_list is not None:
    print(f"len(ids): {len(ids_list)}")

print("\n=== Sample: First metadata dict (index 0) ===")
if metas_list:
    print(f"metas_list[0] = {metas_list[0]}")

print("\n=== Sample: Second metadata dict (index 1) ===")
if metas_list and len(metas_list) > 1:
    print(f"metas_list[1] = {metas_list[1]}")

print("\n=== Sample: First document text (first 200 chars) ===")
if docs_list:
    print(repr(docs_list[0][:200]))

print("\n=== All unique metadata KEYS across ALL dicts ===")
if metas_list:
    all_keys = set()
    for m in metas_list:
        if m:
            all_keys.update(m.keys())
    print(f"Unique keys: {sorted(all_keys)}")

print("\n=== Unique values for 'company' field (first 20) ===")
if metas_list:
    company_vals = set()
    for m in metas_list:
        if m and "company" in m:
            company_vals.add(m["company"])
    print(f"Distinct 'company' values: {sorted(company_vals)}")

print("\n=== Does 'company' key exist in every metadata dict? ===")
if metas_list:
    missing = [i for i, m in enumerate(metas_list) if not m or "company" not in m]
    print(f"Indexes missing 'company' key: {missing[:10]} (total missing: {len(missing)})")

print("\n=== Cross-check: retrieve_node uses filter={'company': full_name} ===")
print("Values above should match full names like 'Apple Inc.', 'Microsoft Corporation', etc.")

print("\nDone.")
