"""
Quick corpus lookup — pull real 10-K values for the 5 diagnostic questions
so we have ground-truth anchors to cross-check the pipeline's final_answer against.
Run before the main diagnostic.
"""
import sys, os, re
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from tools.vectorstore import get_vectorstore

vs = get_vectorstore()
result = vs.get()
documents = result["documents"]
metadatas = result["metadatas"]
ids       = result["ids"]

def get_chunk(chunk_id):
    for doc, meta, cid in zip(documents, metadatas, ids):
        if cid == chunk_id:
            return doc, meta
    return None, None

# --- 1. Apple R&D 2024 (already confirmed: $31,370M) ---
# --- 2. Tesla SG&A 2024 ---
# Look for Tesla SG&A in item7 table
print("=== Tesla SG&A ===")
for doc, meta, cid in zip(documents, metadatas, ids):
    if meta.get("company") == "Tesla Inc." and meta.get("chunk_type") == "table":
        if "selling" in doc.lower() and "general" in doc.lower() and "administrative" in doc.lower():
            # Find the actual $ row
            for line in doc.split("\n"):
                if ("selling" in line.lower() or "sg&" in line.lower()) and "$" in line:
                    print(f"  {cid}: {line.strip()[:120]}")
            break

# --- 3. Apple gross margin 2024 ---
print("\n=== Apple Consolidated Statements of Operations (gross margin) ===")
doc, meta = get_chunk("aapl_2024_item8_table_028_000")
if doc:
    for line in doc.split("\n"):
        if any(kw in line.lower() for kw in ["net sales", "net revenue", "cost of sales", "gross margin", "gross profit"]):
            print(f"  {line.strip()[:120]}")

# --- 4. Tesla gross margin 2024 ---
print("\n=== Tesla Consolidated Statements of Operations (gross margin) ===")
for doc, meta, cid in zip(documents, metadatas, ids):
    if meta.get("company") == "Tesla Inc." and meta.get("chunk_type") == "table":
        if "consolidated statements of operations" in meta.get("table_name", "").lower():
            for line in doc.split("\n"):
                if any(kw in line.lower() for kw in ["revenue", "cost of", "gross profit", "gross margin"]):
                    print(f"  {cid}: {line.strip()[:120]}")
            break

# --- 5. NVIDIA gross margin 2024 ---
print("\n=== NVIDIA gross profit / margin ===")
for doc, meta, cid in zip(documents, metadatas, ids):
    if meta.get("company") == "NVIDIA Corporation" and meta.get("chunk_type") == "table":
        tname = meta.get("table_name", "").lower()
        if "statement" in tname and ("operation" in tname or "income" in tname):
            for line in doc.split("\n"):
                if any(kw in line.lower() for kw in ["revenue", "cost of", "gross profit", "gross margin"]):
                    print(f"  {cid}: {line.strip()[:120]}")
            break

# --- 6. Alphabet R&D 2024 (for Q5 cross-check) ---
print("\n=== Alphabet R&D ===")
for doc, meta, cid in zip(documents, metadatas, ids):
    if meta.get("company") == "Alphabet Inc." and meta.get("chunk_type") == "table":
        if "research" in doc.lower() and "development" in doc.lower():
            for line in doc.split("\n"):
                if "research" in line.lower() and "$" in line:
                    print(f"  {cid}: {line.strip()[:120]}")
            break

# --- 7. Amazon R&D 2024 ---
print("\n=== Amazon R&D ===")
for doc, meta, cid in zip(documents, metadatas, ids):
    if meta.get("company") == "Amazon.com Inc." and meta.get("chunk_type") == "table":
        if "research" in doc.lower() and "development" in doc.lower():
            for line in doc.split("\n"):
                if "research" in line.lower() and ("$" in line or any(c.isdigit() for c in line)):
                    print(f"  {cid}: {line.strip()[:120]}")
            break
