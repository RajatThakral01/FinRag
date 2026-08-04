"""
tool/retrieval/bm25_index.py
-------------------
Builds and caches a BM25Okapi index over the full Chroma corpus (all 9
companies, 3,063 chunks), then exposes a query function that scores every
chunk in the corpus and filters to a specific company via metadata.

Design decisions (Chapter 21, judgment calls):
  1. ONE global index — IDF is more meaningful computed over the full corpus.
  2. Pickle cache — build once, load on every subsequent run.
  3. Corpus pulled directly from Chroma .get() — same chunks as vector index.
  4. Tokenizer version guard — if _TOKENIZER_VERSION changes, the pickle is
     automatically invalidated and rebuilt so index-time and query-time
     tokenization always stay in sync.

The parallel arrays (documents, metadatas, ids) from Chroma .get() are stored
alongside the BM25Okapi object inside the pickle, so rank positions map
directly back to the correct chunk and its metadata without any re-fetch.
"""

import os
import re
import pickle
from rank_bm25 import BM25Okapi
import config
from tools.retrieval.vectorstore import get_vectorstore

# ---------------------------------------------------------------------------
# Tokenizer version guard — bump this whenever _ABBREV_SUBS or _tokenize
# logic changes.  _load_or_build_index() compares the stored version to this
# constant and rebuilds the index from scratch if they differ.
# ---------------------------------------------------------------------------
_TOKENIZER_VERSION = "v2"  # v1 → v2: added net-sales/revenue synonym expansion


# ---------------------------------------------------------------------------
# Tokenizer — applied identically at index-build time and query time
# ---------------------------------------------------------------------------

# Abbreviation expansions applied in this order (most-specific first).
# Order is CRITICAL: specific patterns must fire before generic fallbacks.
# Also handles Docling's HTML-encoded & → &amp; (e.g. "R&amp;D" in tables).
_ABBREV_SUBS = [
    # R&D — plain and HTML-encoded
    (re.compile(r'r\s*&\s*amp\s*;\s*d',  re.IGNORECASE), 'research and development'),
    (re.compile(r'r\s*&\s*d',             re.IGNORECASE), 'research and development'),
    # SG&A — plain and HTML-encoded
    (re.compile(r'sg\s*&\s*amp\s*;\s*a', re.IGNORECASE), 'selling general and administrative'),
    (re.compile(r'sg\s*&\s*a',           re.IGNORECASE), 'selling general and administrative'),
    # "net product sales" / "net service sales" / "net sales" → add "revenue" token.
    # Amazon's 10-K uses these phrases instead of the generic word "revenue".
    # Expanding to include "revenue" ensures BM25 matches revenue queries
    # against Amazon chunks that use the "sales" terminology.
    # Applied BEFORE the & fallback to avoid interfering with any & in the phrase.
    (re.compile(r'net\s+(?:product\s+|service\s+|subscription\s+)?sales', re.IGNORECASE),
     'revenue net sales'),
    # Generic fallback: remaining &amp; and & → "and"
    (re.compile(r'&amp;',                re.IGNORECASE), ' and '),
    (re.compile(r'&'),                                    ' and '),
]


def _tokenize(text: str) -> list[str]:
    """
    Lowercase → expand financial abbreviations → strip remaining punctuation
    → split on whitespace.

    Abbreviation normalization is applied BEFORE punctuation stripping so
    that 'R&D' expands to 'research and development' (matching the spelled-out
    form in corpus tables) rather than collapsing to the single-char tokens
    'r' and 'd' which have near-zero BM25 weight.

    The same function is called for every document at build time and for
    every query at query time — consistency is critical for BM25 IDF.
    """
    text = text.lower()
    for pattern, replacement in _ABBREV_SUBS:
        text = pattern.sub(replacement, text)
    text = re.sub(r'[^\w\s]', ' ', text)   # strip remaining punctuation
    return [tok for tok in text.split() if tok]


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def _build_index_from_chroma() -> dict:
    """
    Pull all documents, metadatas, and ids from Chroma, tokenize, build
    BM25Okapi, and return a bundle dict that will be pickled to disk.

    The bundle keeps documents/metadatas/ids as aligned parallel lists —
    position i in each list corresponds to the same chunk, matching how
    Chroma .get() returns them. This lets BM25 rank indices map directly
    to the right chunk without any re-fetching.
    """
    print("[bm25_index] Pulling corpus from Chroma .get() ...")
    vs = get_vectorstore()
    result = vs.get()

    documents: list[str] = result["documents"]
    metadatas: list[dict] = result["metadatas"]
    ids: list[str] = result["ids"]

    total = len(documents)
    print(f"[bm25_index] Corpus size: {total} chunks")

    # Tokenize every document (same tokenizer as query time)
    print("[bm25_index] Tokenizing corpus ...")
    tokenized_corpus = [_tokenize(doc) for doc in documents]

    # Build BM25Okapi
    print("[bm25_index] Building BM25Okapi index ...")
    bm25 = BM25Okapi(tokenized_corpus)
    print("[bm25_index] Index built.")

    return {
        "tokenizer_version": _TOKENIZER_VERSION,  # version guard for pickle validation
        "bm25": bm25,
        "documents": documents,   # list[str], parallel to metadatas and ids
        "metadatas": metadatas,   # list[dict], parallel — each has 'company' key
        "ids": ids,               # list[str], parallel chunk_id values
    }


def _load_or_build_index() -> dict:
    """
    Load the pickled bundle if the cache file exists AND its tokenizer version
    matches _TOKENIZER_VERSION; build (and save) it otherwise.

    Version mismatch triggers an automatic rebuild — this prevents silent
    BM25 degradation when the tokenizer changes (index-time and query-time
    tokenization must be identical for IDF to be meaningful).
    """
    cache_path = config.BM25_INDEX_PATH

    if os.path.exists(cache_path):
        print(f"[bm25_index] Loading cached index from {cache_path} ...")
        with open(cache_path, "rb") as f:
            bundle = pickle.load(f)

        cached_version = bundle.get("tokenizer_version", "v1")  # pre-versioning bundles → v1
        if cached_version == _TOKENIZER_VERSION:
            print(f"[bm25_index] Cache loaded ({len(bundle['documents'])} chunks). "
                  f"Tokenizer version: {cached_version}")
            return bundle
        else:
            print(f"[bm25_index] Tokenizer version mismatch "
                  f"(cached={cached_version!r}, current={_TOKENIZER_VERSION!r}). "
                  f"Rebuilding index ...")

    # Cache miss or version mismatch — build from scratch
    bundle = _build_index_from_chroma()

    print(f"[bm25_index] Saving index to {cache_path} ...")
    with open(cache_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[bm25_index] Index saved.")

    return bundle


# Module-level lazy cache — stays loaded for the process lifetime so
# repeated calls to bm25_query() within the same run don't reload from disk.
_bundle: dict | None = None


def _get_bundle() -> dict:
    global _bundle
    if _bundle is None:
        _bundle = _load_or_build_index()
    return _bundle


# ---------------------------------------------------------------------------
# Public query function
# ---------------------------------------------------------------------------

def bm25_query(query: str, company: str, top_k: int = 20) -> list[dict]:
    """
    Score all chunks in the global BM25 index for `query`, then filter to
    only chunks belonging to `company` (matched against the 'company' metadata
    field, which always contains a populated full name — confirmed in Step 1).

    Returns a ranked list of dicts, best-scoring first:
        [
            {
                "text": str,       # raw chunk text
                "metadata": dict,  # full metadata dict (company, ticker, ...)
                "id": str,         # chunk_id
                "score": float,    # raw BM25 score (for debugging/logging only)
                "rank": int,       # 1-based rank within this company's results
            },
            ...
        ]

    `top_k` is a soft upper bound — if fewer than top_k chunks exist for
    the given company, all of them are returned.

    NOTE: rank positions here are within the COMPANY-FILTERED results, not
    across the full corpus — this is intentional. The RRF merge in
    retrieve_node will combine these per-company BM25 ranks with per-company
    vector ranks.
    """
    bundle = _get_bundle()
    bm25: BM25Okapi = bundle["bm25"]
    documents: list[str] = bundle["documents"]
    metadatas: list[dict] = bundle["metadatas"]
    ids: list[str] = bundle["ids"]

    # Tokenize query identically to how documents were tokenized at build time
    tokenized_query = _tokenize(query)

    # Score ALL chunks in the corpus (BM25 returns one score per corpus position)
    all_scores: list[float] = bm25.get_scores(tokenized_query).tolist()

    # Build (score, index) pairs, filter to requested company, then sort desc
    company_results = []
    for idx, (score, meta) in enumerate(zip(all_scores, metadatas)):
        # meta["company"] is always a non-empty string (confirmed in Step 1)
        if meta.get("company") == company:
            company_results.append((score, idx))

    company_results.sort(key=lambda x: x[0], reverse=True)

    # Take top_k and build the return list
    ranked = []
    for rank, (score, idx) in enumerate(company_results[:top_k], start=1):
        ranked.append({
            "text": documents[idx],
            "metadata": metadatas[idx],
            "id": ids[idx],
            "score": score,
            "rank": rank,
        })

    return ranked


# ---------------------------------------------------------------------------
# Standalone test — run from RAG_Project/ root
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TESTS = [
        ("R&D expenses",   "Apple Inc.",  8),
        ("SG&A expenses",  "Tesla Inc.",  8),
    ]

    for test_query, test_company, top_n in TESTS:
        print(f"\n{'=' * 60}")
        print(f"Query:    {test_query!r}  →  tokens: {_tokenize(test_query)}")
        print(f"Company:  {test_company!r}")
        print(f"Top-{top_n} results:")
        print(f"{'=' * 60}")

        results = bm25_query(test_query, test_company, top_k=top_n)

        for r in results:
            meta = r["metadata"]
            print(f"  Rank {r['rank']} | score={r['score']:.4f} | {meta['chunk_id']}")
            print(f"         type={meta['chunk_type']}  item={meta.get('item_number','?')}"
                  f"  table_name={meta.get('table_name','')!r}")
            snippet = r["text"].replace("\n", " ")[:280]
            print(f"         {snippet!r}")
            print()
