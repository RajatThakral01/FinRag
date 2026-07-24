"""
THROWAWAY DIAGNOSTIC — tokenization + BM25 table-chunk investigation.
Run from RAG_Project/ root. Delete after confirming output.
"""
import sys, os, re
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tools.vectorstore import get_vectorstore

# ---- replicate the exact current tokenizer from tools/bm25_index.py ----
def _tokenize_current(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [tok for tok in text.split() if tok]

# -----------------------------------------------------------------------
# 1. Show exactly what happens to "R&D"
# -----------------------------------------------------------------------
print("=" * 60)
print("1. TOKENIZER BEHAVIOUR")
print("=" * 60)

test_strings = [
    "R&D",
    "R&D expenses",
    "Research and development expense",
    "Apple research and development total",
]
for s in test_strings:
    tokens = _tokenize_current(s)
    print(f"  Input:  {s!r}")
    print(f"  Tokens: {tokens}")
    print()

# -----------------------------------------------------------------------
# 2. Find the actual Apple Item 8 R&D table chunk(s)
# -----------------------------------------------------------------------
print("=" * 60)
print("2. FINDING APPLE ITEM 8 R&D TABLE CHUNKS IN CHROMA")
print("=" * 60)

vs = get_vectorstore()
result = vs.get()
documents = result["documents"]
metadatas = result["metadatas"]
ids       = result["ids"]

# Look for Apple Item 8 table chunks mentioning research/R&D
apple_item8_rd = []
for doc, meta, cid in zip(documents, metadatas, ids):
    if meta.get("company") != "Apple Inc.":
        continue
    if meta.get("chunk_type") != "table":
        continue
    if "item 8" not in meta.get("section_name", "").lower() and meta.get("item_number") != "8":
        continue
    doc_lower = doc.lower()
    if "research" in doc_lower or "r&d" in doc_lower or "r & d" in doc_lower:
        apple_item8_rd.append((doc, meta, cid))

print(f"Found {len(apple_item8_rd)} Apple Item 8 TABLE chunks mentioning research/R&D\n")

for doc, meta, cid in apple_item8_rd:
    print(f"  chunk_id: {cid}")
    print(f"  section:  {meta.get('section_name')}")
    print(f"  table_name: {meta.get('table_name')!r}")
    tokens = _tokenize_current(doc)
    print(f"  First 400 chars of text:")
    print(f"    {doc[:400]!r}")
    print(f"  Tokenized (first 40 tokens): {tokens[:40]}")
    print()

# -----------------------------------------------------------------------
# 3. Check ALL Apple Item 8 table chunks to see if any are missing
# -----------------------------------------------------------------------
print("=" * 60)
print("3. ALL APPLE ITEM 8 TABLE CHUNKS (chunk_ids only)")
print("=" * 60)
apple_item8_all = [(meta["chunk_id"], meta.get("table_name",""), doc[:120])
                   for doc, meta, _ in zip(documents, metadatas, ids)
                   if meta.get("company") == "Apple Inc."
                   and meta.get("chunk_type") == "table"
                   and (meta.get("item_number") == "8"
                        or "item 8" in meta.get("section_name","").lower())]
print(f"Total Apple Item 8 TABLE chunks: {len(apple_item8_all)}")
for cid, tname, snippet in apple_item8_all:
    print(f"  {cid}  table_name={tname!r}")
    print(f"    snippet: {snippet!r}")
    print()

# -----------------------------------------------------------------------
# 4. Alternate-phrasing BM25 test (load cached index)
# -----------------------------------------------------------------------
print("=" * 60)
print("4. BM25 TOP-8 WITH ALTERNATE PHRASINGS")
print("=" * 60)

from tools.bm25_index import bm25_query

alt_queries = [
    "Research and development expense",
    "Apple research and development total",
    "R&D expenses",
]

for q in alt_queries:
    print(f"\n  Query: {q!r}  → tokens: {_tokenize_current(q)}")
    results = bm25_query(q, "Apple Inc.", top_k=8)
    for r in results:
        meta = r["metadata"]
        print(f"    Rank {r['rank']} score={r['score']:.4f} | {meta['chunk_id']}"
              f" | type={meta['chunk_type']} | item={meta.get('item_number','?')}"
              f" | table_name={meta.get('table_name','')!r}")
        snippet = r["text"].replace("\n"," ")[:200]
        print(f"           {snippet!r}")
