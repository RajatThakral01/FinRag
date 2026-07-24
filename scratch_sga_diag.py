"""
THROWAWAY DIAGNOSTIC — SG&A corpus check + regex ordering verification.
Run from RAG_Project/ root. Delete after confirming output.
"""
import sys, os, re
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tools.vectorstore import get_vectorstore

# -----------------------------------------------------------------------
# 1. Proposed new tokenizer — verify regex ordering produces correct output
# -----------------------------------------------------------------------
def _tokenize_new(text: str) -> list[str]:
    """Proposed fix — abbreviation normalization BEFORE punctuation strip."""
    text = text.lower()
    # Specific patterns BEFORE generic & fallback — order is critical
    text = re.sub(r'r\s*&\s*amp\s*;\s*d', 'research and development', text)  # HTML R&amp;D
    text = re.sub(r'r\s*&\s*d',           'research and development', text)   # plain R&D
    text = re.sub(r'sg\s*&\s*amp\s*;\s*a','selling general and administrative', text)  # HTML SG&amp;A
    text = re.sub(r'sg\s*&\s*a',          'selling general and administrative', text)  # plain SG&A
    text = re.sub(r'&amp;',               ' and ', text)   # remaining HTML &amp;
    text = re.sub(r'&',                   ' and ', text)   # remaining plain &
    text = re.sub(r'[^\w\s]',            ' ',     text)   # strip remaining punctuation
    return [tok for tok in text.split() if tok]

def _tokenize_old(text: str) -> list[str]:
    """Original broken tokenizer — for comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [tok for tok in text.split() if tok]

print("=" * 60)
print("1. REGEX ORDERING VERIFICATION — R&D")
print("=" * 60)
rd_inputs = [
    "R&D",
    "R&D expenses",
    "R&amp;D",           # Docling HTML-encoded form
    "R & D expenses",
]
for s in rd_inputs:
    old = _tokenize_old(s)
    new = _tokenize_new(s)
    match = new[:3] == ['research', 'and', 'development'] or new[:4] == ['research', 'and', 'development', 'expenses']
    print(f"  Input:   {s!r}")
    print(f"  OLD tokens: {old}")
    print(f"  NEW tokens: {new}  {'✓ CORRECT' if 'research' in new and 'development' in new else '✗ WRONG'}")
    print()

# Verify 'r and d' (wrong) does NOT appear when generic & fires before specific pattern
print("  Critical order check — generic & first (WRONG order):")
wrong = "r&d expenses".replace("&", " and ")
print(f"  If & replaced first: {_tokenize_old(wrong)!r}  ← should NOT be this")
print()

print("=" * 60)
print("2. REGEX ORDERING VERIFICATION — SG&A")
print("=" * 60)
sga_inputs = [
    "SG&A",
    "SG&A expenses",
    "SG&amp;A",           # possible HTML form
    "SG & A expenses",
]
for s in sga_inputs:
    old = _tokenize_old(s)
    new = _tokenize_new(s)
    print(f"  Input:   {s!r}")
    print(f"  OLD tokens: {old}")
    print(f"  NEW tokens: {new}  {'✓ CORRECT' if 'selling' in new and 'administrative' in new else '✗ WRONG'}")
    print()

# Verify the corpus tokens from "Selling, general and administrative"
corpus_phrase = "Selling, general and administrative"
print(f"  Corpus phrase: {corpus_phrase!r}")
print(f"  Corpus tokens (NEW): {_tokenize_new(corpus_phrase)}")
print(f"  SG&A tokens   (NEW): {_tokenize_new('SG&A')}")
match_check = _tokenize_new('SG&A') == _tokenize_new('Selling general and administrative')
print(f"  Token match (SG&A vs corpus phrase): {'✓ YES' if match_check else '✗ NO'}")
print()

# -----------------------------------------------------------------------
# 2. Search corpus for SG&A mentions
# -----------------------------------------------------------------------
print("=" * 60)
print("3. CORPUS SG&A SEARCH")
print("=" * 60)
vs = get_vectorstore()
result = vs.get()
documents = result["documents"]
metadatas = result["metadatas"]
ids       = result["ids"]

# Find chunks containing SG&A (abbreviated) across all companies
sga_abbreviated = []
for doc, meta, cid in zip(documents, metadatas, ids):
    doc_lower = doc.lower()
    if 'sg&a' in doc_lower or 'sg&amp;a' in doc_lower or 'sg & a' in doc_lower:
        sga_abbreviated.append((doc, meta, cid))

print(f"Chunks containing abbreviated 'SG&A' (any company): {len(sga_abbreviated)}")
for doc, meta, cid in sga_abbreviated[:5]:
    print(f"  {cid}  company={meta['company']}  type={meta['chunk_type']}")
    print(f"    snippet: {doc[:200]!r}")
    print()

# Find chunks containing "Selling, general and administrative" — spelled out
sga_spelled = []
for doc, meta, cid in zip(documents, metadatas, ids):
    if 'selling, general and administrative' in doc.lower() or \
       'selling general and administrative' in doc.lower():
        sga_spelled.append((doc, meta, cid))

print(f"Chunks containing spelled-out 'Selling, general and administrative': {len(sga_spelled)}")
companies_with_sga = sorted(set(meta['company'] for _, meta, _ in sga_spelled))
print(f"Companies: {companies_with_sga}")

# Show one example per company (table chunks preferred)
shown = set()
for doc, meta, cid in sga_spelled:
    co = meta['company']
    if co in shown:
        continue
    if meta['chunk_type'] == 'table':  # prefer table chunks
        shown.add(co)
        print(f"\n  {cid}  company={co}  type={meta['chunk_type']}")
        print(f"  table_name: {meta.get('table_name','')!r}")
        print(f"  snippet: {doc[:300]!r}")

# Also check: are any SG&A table chunks in Item 8?
print("\n\n  Item 8 TABLE chunks with SG&A (spelled out):")
for doc, meta, cid in sga_spelled:
    if meta.get('chunk_type') == 'table' and (
        meta.get('item_number') == '8' or 'item 8' in meta.get('section_name','').lower()):
        print(f"    {cid}  company={meta['company']}  table_name={meta.get('table_name','')!r}")
        print(f"      snippet: {doc[:200]!r}")

print("\nDone.")
