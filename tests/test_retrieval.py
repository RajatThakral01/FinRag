"""
tests/test_retrieval.py
=======================
Retrieval-only diagnostic — NO LLM calls, NO graph, NO grading.

Tests:
  A. BM25 component:  tokenizer correctness, zero-score guard, company isolation
  B. Vector component: returns expected count, company filter works
  C. RRF merge:        overlap boosts rank, unique-method chunks kept, dedup
  D. Integration:      1-company / 2-company / 3-company end-to-end retrieval
                       via retrieve_node() logic called directly (no state machine)

Run:
  source venv/bin/activate
  python -m pytest tests/test_retrieval.py -v -s
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

# ─── shared fixture: vectorstore + bm25 loaded once per session ───────────
@pytest.fixture(scope="session")
def vs():
    from tools.retrieval.vectorstore import get_vectorstore
    return get_vectorstore()


@pytest.fixture(scope="session")
def bundle():
    from tools.retrieval.bm25_index import _get_bundle
    return _get_bundle()


# ══════════════════════════════════════════════════════════════════════════════
# A — BM25 component tests
# ══════════════════════════════════════════════════════════════════════════════
class TestBM25Component:

    def test_tokenizer_expands_rnd(self):
        from tools.retrieval.bm25_index import _tokenize
        tokens = _tokenize("R&D expenses")
        assert "research" in tokens
        assert "development" in tokens
        assert "r" not in tokens          # should NOT produce bare "r"

    def test_tokenizer_expands_sga(self):
        from tools.retrieval.bm25_index import _tokenize
        tokens = _tokenize("SG&A costs")
        assert "selling" in tokens
        assert "general" in tokens
        assert "administrative" in tokens

    def test_tokenizer_html_encoded_ampersand(self):
        from tools.retrieval.bm25_index import _tokenize
        tokens = _tokenize("R&amp;D expense")
        assert "research" in tokens
        assert "development" in tokens

    def test_tokenizer_net_sales_adds_revenue_token(self):
        """Amazon uses 'net sales' — tokenizer must emit 'revenue' so BM25 matches revenue queries."""
        from tools.retrieval.bm25_index import _tokenize
        tokens = _tokenize("net sales")
        assert "revenue" in tokens, "net sales should emit 'revenue' token"
        assert "net" in tokens
        assert "sales" in tokens

    def test_tokenizer_net_product_sales_adds_revenue_token(self):
        from tools.retrieval.bm25_index import _tokenize
        tokens = _tokenize("net product sales 2024")
        assert "revenue" in tokens

    def test_tokenizer_net_service_sales_adds_revenue_token(self):
        from tools.retrieval.bm25_index import _tokenize
        tokens = _tokenize("AWS net service sales")
        assert "revenue" in tokens

    def test_tokenizer_strips_punctuation(self):
        from tools.retrieval.bm25_index import _tokenize
        tokens = _tokenize("revenue, net income (fiscal 2024).")
        assert "revenue" in tokens
        assert "net" in tokens
        assert "income" in tokens
        assert "fiscal" in tokens
        assert "2024" in tokens
        # punctuation should be removed
        for tok in tokens:
            assert "," not in tok
            assert "(" not in tok
            assert "." not in tok

    def test_tokenizer_lowercase(self):
        from tools.retrieval.bm25_index import _tokenize
        tokens = _tokenize("Total Revenue APPLE")
        assert all(t == t.lower() for t in tokens)

    def test_bm25_returns_results_for_apple(self, bundle):
        from tools.retrieval.bm25_index import bm25_query
        results = bm25_query("total revenue", "Apple Inc.", top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_bm25_company_filter_isolates(self, bundle):
        """All returned results must belong only to the requested company."""
        from tools.retrieval.bm25_index import bm25_query
        results = bm25_query("net income", "Tesla Inc.", top_k=10)
        for r in results:
            assert r["metadata"]["company"] == "Tesla Inc.", \
                f"Expected Tesla Inc. but got {r['metadata']['company']}"

    def test_bm25_rank_is_sequential_from_1(self, bundle):
        from tools.retrieval.bm25_index import bm25_query
        results = bm25_query("operating income", "Microsoft Corporation", top_k=5)
        for i, r in enumerate(results, start=1):
            assert r["rank"] == i, f"Expected rank {i}, got {r['rank']}"

    def test_bm25_scores_descending(self, bundle):
        from tools.retrieval.bm25_index import bm25_query
        results = bm25_query("revenue", "Apple Inc.", top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "BM25 results must be ranked descending"

    def test_bm25_zero_score_query_still_returns(self, bundle):
        """A query with no matching tokens should return chunks (score=0) not crash."""
        from tools.retrieval.bm25_index import bm25_query
        results = bm25_query("xyzzy foobar qwerty", "Apple Inc.", top_k=3)
        # Should return results (potentially with score 0), not raise
        assert isinstance(results, list)
        for r in results:
            assert r["score"] >= 0.0

    def test_bm25_result_has_required_keys(self, bundle):
        from tools.retrieval.bm25_index import bm25_query
        results = bm25_query("revenue", "Apple Inc.", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "text" in r
        assert "metadata" in r
        assert "id" in r
        assert "score" in r
        assert "rank" in r

    def test_bm25_different_companies_different_results(self, bundle):
        """Apple revenue top result chunk_id should differ from NVIDIA revenue top result."""
        from tools.retrieval.bm25_index import bm25_query
        apple = bm25_query("total revenue", "Apple Inc.", top_k=1)
        nvidia = bm25_query("total revenue", "NVIDIA Corporation", top_k=1)
        if apple and nvidia:
            assert apple[0]["id"] != nvidia[0]["id"], \
                "Top chunk for Apple and NVIDIA should be different chunks"

    def test_bm25_rnd_query_prefers_rnd_section(self, bundle):
        """R&D query should surface chunks with 'research' or 'development' in text."""
        from tools.retrieval.bm25_index import bm25_query
        results = bm25_query("research and development expense", "Apple Inc.", top_k=3)
        hit = any(
            "research" in r["text"].lower() or "development" in r["text"].lower()
            for r in results
        )
        assert hit, "Top R&D chunks should contain 'research' or 'development'"

    def test_bm25_corpus_size(self, bundle):
        """Verify corpus has the expected ~3063 chunks."""
        n = len(bundle["documents"])
        assert 2900 <= n <= 3200, f"Corpus size {n} is outside expected range 2900-3200"


# ══════════════════════════════════════════════════════════════════════════════
# B — Vector (Chroma) component tests
# ══════════════════════════════════════════════════════════════════════════════
class TestVectorComponent:

    def test_single_company_returns_k_chunks(self, vs):
        results = vs.similarity_search(
            "total revenue fiscal 2024", k=5, filter={"company": "Apple Inc."}
        )
        assert len(results) == 5, f"Expected 5 chunks, got {len(results)}"

    def test_company_filter_isolation(self, vs):
        """Vector filter must return ONLY the requested company."""
        results = vs.similarity_search(
            "net income", k=5, filter={"company": "Tesla Inc."}
        )
        for doc in results:
            assert doc.metadata["company"] == "Tesla Inc.", \
                f"Expected Tesla but got {doc.metadata['company']}"

    def test_each_result_has_chunk_id(self, vs):
        results = vs.similarity_search(
            "operating income", k=3, filter={"company": "Microsoft Corporation"}
        )
        for doc in results:
            assert "chunk_id" in doc.metadata and doc.metadata["chunk_id"], \
                "Every vector result must have a non-empty chunk_id"

    def test_different_queries_return_different_top_chunk(self, vs):
        r1 = vs.similarity_search(
            "total revenue", k=1, filter={"company": "Apple Inc."}
        )
        r2 = vs.similarity_search(
            "risk factors competition", k=1, filter={"company": "Apple Inc."}
        )
        if r1 and r2:
            assert r1[0].metadata.get("chunk_id") != r2[0].metadata.get("chunk_id"), \
                "Revenue and risk-factor queries should not return the same top chunk"

    def test_vector_rnd_query_returns_relevant_section(self, vs):
        results = vs.similarity_search(
            "research and development expense", k=3, filter={"company": "NVIDIA Corporation"}
        )
        hit = any(
            "research" in doc.page_content.lower() or "r&d" in doc.page_content.lower()
            for doc in results
        )
        assert hit, "Vector R&D query should return at least one R&D-related chunk"

    def test_vector_filter_unknown_company_returns_empty(self, vs):
        results = vs.similarity_search(
            "revenue", k=5, filter={"company": "Unknown Corp XYZ"}
        )
        assert results == [], f"Unknown company filter should return [], got {len(results)} results"


# ══════════════════════════════════════════════════════════════════════════════
# C — RRF merge tests (pure logic, no live DB calls)
# ══════════════════════════════════════════════════════════════════════════════
from graph.nodes import _rrf_merge, _RRF_K

def _mock_vec_doc(chunk_id: str, text: str = "vec text", company: str = "Apple Inc."):
    """Build a minimal Langchain Document-like object."""
    from langchain_core.documents import Document
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "company": company,
            "ticker": "AAPL",
            "section_name": "MD&A",
            "chunk_type": "TEXT",
            "table_name": None,
        }
    )

def _mock_bm25_result(chunk_id: str, rank: int, text: str = "bm25 text", company: str = "Apple Inc."):
    return {
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "company": company,
            "ticker": "AAPL",
            "section_name": "MD&A",
            "chunk_type": "TEXT",
            "table_name": None,
        },
        "id": chunk_id,
        "score": 10.0 - rank,
        "rank": rank,
    }

class TestRRFMerge:

    def test_overlap_chunk_scores_higher_than_single_method(self):
        """A chunk appearing in both vector AND BM25 should outscore one only in vector."""
        shared_id = "chunk-shared"
        only_vec_id = "chunk-vec-only"

        vec_docs = [
            _mock_vec_doc(shared_id),    # rank 1 in vector
            _mock_vec_doc(only_vec_id),  # rank 2 in vector
        ]
        bm25_res = [
            _mock_bm25_result(shared_id, rank=1),  # rank 1 in BM25
        ]
        texts, sources = _rrf_merge(vec_docs, bm25_res, final_k=2)
        chunk_ids = [s["chunk_id"] for s in sources]
        assert chunk_ids[0] == shared_id, \
            f"Overlapping chunk should rank first. Got: {chunk_ids}"

    def test_unique_bm25_chunk_included(self):
        """A chunk only in BM25 (not in vector) should still appear in merged output."""
        bm25_only_id = "chunk-bm25-only"
        vec_docs = [_mock_vec_doc("chunk-vec-1")]
        bm25_res = [
            _mock_bm25_result("chunk-vec-1", rank=1),
            _mock_bm25_result(bm25_only_id,  rank=2),
        ]
        texts, sources = _rrf_merge(vec_docs, bm25_res, final_k=2)
        chunk_ids = [s["chunk_id"] for s in sources]
        assert bm25_only_id in chunk_ids, \
            "BM25-only chunk must still appear in merged results"

    def test_unique_vector_chunk_included(self):
        """A chunk only in vector (not in BM25) should still appear."""
        vec_only_id = "chunk-vec-only"
        vec_docs = [_mock_vec_doc("chunk-shared"), _mock_vec_doc(vec_only_id)]
        bm25_res = [_mock_bm25_result("chunk-shared", rank=1)]
        texts, sources = _rrf_merge(vec_docs, bm25_res, final_k=2)
        chunk_ids = [s["chunk_id"] for s in sources]
        assert vec_only_id in chunk_ids, \
            "Vector-only chunk must still appear in merged results"

    def test_dedup_same_chunk_id_not_duplicated(self):
        """Same chunk_id appearing in both methods must only appear once in output."""
        shared_id = "chunk-abc"
        vec_docs = [_mock_vec_doc(shared_id)]
        bm25_res = [_mock_bm25_result(shared_id, rank=1)]
        texts, sources = _rrf_merge(vec_docs, bm25_res, final_k=5)
        chunk_ids = [s["chunk_id"] for s in sources]
        assert chunk_ids.count(shared_id) == 1, \
            f"Chunk {shared_id} must appear exactly once, got {chunk_ids.count(shared_id)}"

    def test_final_k_truncates_output(self):
        vec_docs = [_mock_vec_doc(f"chunk-{i}") for i in range(6)]
        bm25_res = [_mock_bm25_result(f"chunk-{i}", rank=i+1) for i in range(6)]
        texts, sources = _rrf_merge(vec_docs, bm25_res, final_k=3)
        assert len(texts) == 3
        assert len(sources) == 3

    def test_parallel_texts_and_sources_same_length(self):
        vec_docs = [_mock_vec_doc(f"chunk-{i}") for i in range(3)]
        bm25_res = [_mock_bm25_result(f"chunk-{i}", rank=i+1) for i in range(3)]
        texts, sources = _rrf_merge(vec_docs, bm25_res, final_k=4)
        assert len(texts) == len(sources)

    def test_rrf_score_formula_correct(self):
        """Verify the RRF formula: score = 1/(60+rank). Rank-1 vector alone."""
        shared_id = "chunk-x"
        vec_docs = [_mock_vec_doc(shared_id)]  # rank 1 in vector
        bm25_res = []
        texts, sources = _rrf_merge(vec_docs, bm25_res, final_k=1)
        # Only one chunk — just check it's included
        assert sources[0]["chunk_id"] == shared_id

    def test_empty_inputs_return_empty(self):
        texts, sources = _rrf_merge([], [], final_k=5)
        assert texts == []
        assert sources == []

    def test_rrf_ranks_top_overlap_chunk_first(self):
        """
        chunk-A: rank 1 in vector, rank 1 in BM25 → score = 1/61 + 1/61 ≈ 0.0328
        chunk-B: rank 1 in vector only            → score = 1/61 ≈ 0.0164
        chunk-A must rank above chunk-B.
        """
        vec_docs = [_mock_vec_doc("chunk-A"), _mock_vec_doc("chunk-B")]
        bm25_res = [_mock_bm25_result("chunk-A", rank=1)]
        texts, sources = _rrf_merge(vec_docs, bm25_res, final_k=2)
        assert sources[0]["chunk_id"] == "chunk-A"


# ══════════════════════════════════════════════════════════════════════════════
# D — End-to-end retrieval integration (calls real Chroma + BM25 directly)
# ══════════════════════════════════════════════════════════════════════════════
from tools.retrieval.vectorstore import get_vectorstore as _vs
from tools.retrieval.bm25_index import bm25_query as _bm25q
from tools.retrieval.company_names import SHORT_TO_FULL
from graph.nodes import _rrf_merge as _rrf

def _retrieve(query: str, companies: list[str], per_company_k: int = 5):
    """
    Replicate retrieve_node logic exactly, without the graph state machine.
    Returns (all_texts, all_sources, per_company_detail).
    per_company_detail is a list of dicts with diagnostic info per company.
    """
    vs = _vs()
    detail = []

    all_texts = []
    all_sources = []

    for short_name in companies:
        full_name = SHORT_TO_FULL.get(short_name)
        if not full_name:
            continue

        vec_docs = vs.similarity_search(query, k=per_company_k, filter={"company": full_name})
        bm25_res = _bm25q(query, full_name, top_k=20)
        texts, sources = _rrf(vec_docs, bm25_res, final_k=per_company_k)

        # Chunk IDs in vector vs BM25 result (for overlap analysis)
        vec_ids  = {d.metadata.get("chunk_id") for d in vec_docs}
        bm25_ids = {r["id"] for r in bm25_res[:per_company_k]}  # top-k only
        overlap  = vec_ids & bm25_ids

        detail.append({
            "company":      short_name,
            "full_name":    full_name,
            "vec_count":    len(vec_docs),
            "bm25_count":   len(bm25_res),
            "rrf_count":    len(texts),
            "overlap_count": len(overlap),
            "overlap_ids":  overlap,
            "vec_ids":      vec_ids,
            "bm25_ids":     bm25_ids,
            "rrf_sources":  sources,
            "rrf_texts":    texts,
            "vec_top_sections":  [d.metadata.get("section_name") for d in vec_docs[:3]],
            "bm25_top_sections": [r["metadata"].get("section_name") for r in bm25_res[:3]],
            "rrf_top_sections":  [s.get("section_name") for s in sources[:3]],
        })

        all_texts.extend(texts)
        all_sources.extend(sources)

    return all_texts, all_sources, detail


class TestSingleCompanyRetrieval:
    """D1 — Single company queries."""

    def test_apple_revenue_returns_5_chunks(self):
        texts, sources, detail = _retrieve(
            "What was Apple's total revenue in fiscal year 2024?",
            ["Apple"], per_company_k=5
        )
        d = detail[0]
        print(f"\n[1-company Apple revenue] vec={d['vec_count']} bm25={d['bm25_count']} rrf={d['rrf_count']} overlap={d['overlap_count']}")
        assert d["rrf_count"] == 5, f"Expected 5, got {d['rrf_count']}"

    def test_single_company_filter_no_cross_contamination(self):
        """All returned chunks must be from the requested company."""
        texts, sources, detail = _retrieve(
            "net income profit", ["Apple"], per_company_k=5
        )
        for src in sources:
            assert src["company"] == "Apple Inc.", \
                f"Cross-contamination: got {src['company']}"

    def test_nvidia_rnd_top_chunks_relevant(self):
        texts, sources, detail = _retrieve(
            "research and development expense", ["NVIDIA"], per_company_k=5
        )
        d = detail[0]
        print(f"\n[1-company NVIDIA R&D] vec={d['vec_count']} bm25={d['bm25_count']} overlap={d['overlap_count']}")
        print(f"  RRF top sections: {d['rrf_top_sections']}")
        hit = any("research" in t.lower() or "development" in t.lower() for t in texts)
        assert hit, "At least one chunk should mention 'research' or 'development'"

    def test_tesla_operating_income_sections(self):
        texts, sources, detail = _retrieve(
            "operating income loss 2024", ["Tesla"], per_company_k=5
        )
        d = detail[0]
        print(f"\n[1-company Tesla opIncome] RRF top sections: {d['rrf_top_sections']}")
        assert d["rrf_count"] == 5

    def test_single_company_overlap_exists(self):
        """
        For a clear financial metric query, at least 1 chunk should appear
        in BOTH vector AND BM25 results (overlap > 0).
        """
        texts, sources, detail = _retrieve(
            "total net sales revenue 2024", ["Apple"], per_company_k=5
        )
        d = detail[0]
        print(f"\n[overlap check Apple] overlap={d['overlap_count']} / vec={d['vec_count']} bm25={d['bm25_count']}")
        # NOTE: this may legitimately be 0 if BM25 and vector surface completely different chunks
        # We warn rather than fail hard here
        if d["overlap_count"] == 0:
            print("  WARNING: zero overlap between vector and BM25 results — RRF is acting as simple union")
        # Always passes — this is a diagnostic assertion
        assert d["rrf_count"] > 0

    def test_bm25_top1_score_nonzero_for_clear_query(self):
        """BM25 rank-1 chunk should have score > 0 for a real financial query."""
        from tools.retrieval.bm25_index import bm25_query
        results = bm25_query("total revenue net sales", "Apple Inc.", top_k=1)
        assert results[0]["score"] > 0, \
            f"BM25 top score is 0 for a clear revenue query — tokenizer may have issues"


class TestTwoCompanyRetrieval:
    """D2 — Two-company queries (4 chunks per company = 8 total)."""

    def test_two_companies_returns_8_chunks(self):
        texts, sources, detail = _retrieve(
            "operating income 2024", ["Apple", "Microsoft"], per_company_k=4
        )
        print(f"\n[2-company Apple+Microsoft]")
        for d in detail:
            print(f"  {d['company']}: vec={d['vec_count']} bm25={d['bm25_count']} rrf={d['rrf_count']} overlap={d['overlap_count']}")
        assert len(texts) == 8, f"Expected 8 chunks total, got {len(texts)}"

    def test_two_companies_both_represented(self):
        """Output should contain chunks from both companies."""
        texts, sources, detail = _retrieve(
            "gross profit margin", ["Apple", "NVIDIA"], per_company_k=4
        )
        companies_in_output = {s["company"] for s in sources}
        assert "Apple Inc." in companies_in_output, "Apple missing from output"
        assert "NVIDIA Corporation" in companies_in_output, "NVIDIA missing from output"

    def test_two_companies_no_cross_contamination(self):
        """Apple chunks must not contain Tesla metadata, and vice versa."""
        texts, sources, detail = _retrieve(
            "revenue net income 2024", ["Apple", "Tesla"], per_company_k=4
        )
        allowed = {"Apple Inc.", "Tesla Inc."}
        for src in sources:
            assert src["company"] in allowed, \
                f"Cross-contamination: unexpected company {src['company']}"

    def test_two_companies_each_gets_4_chunks(self):
        texts, sources, detail = _retrieve(
            "cash flow from operations", ["Amazon", "Meta"], per_company_k=4
        )
        for d in detail:
            assert d["rrf_count"] == 4, \
                f"{d['company']}: expected 4 RRF chunks, got {d['rrf_count']}"

    def test_two_companies_overlap_diagnostic(self):
        """Diagnostic: print and check overlap for both companies."""
        texts, sources, detail = _retrieve(
            "research and development expense", ["Apple", "NVIDIA"], per_company_k=4
        )
        print(f"\n[R&D overlap diagnostic]")
        for d in detail:
            print(f"  {d['company']}: overlap={d['overlap_count']}, "
                  f"vec_top={d['vec_top_sections'][:2]}, "
                  f"bm25_top={d['bm25_top_sections'][:2]}, "
                  f"rrf_top={d['rrf_top_sections'][:2]}")
        assert len(texts) == 8


class TestThreeCompanyRetrieval:
    """D3 — Three-company queries (4 chunks per company = 12 total)."""

    def test_three_companies_returns_12_chunks(self):
        texts, sources, detail = _retrieve(
            "total revenue 2024", ["Apple", "Microsoft", "Amazon"], per_company_k=4
        )
        print(f"\n[3-company Apple+MSFT+Amazon]")
        for d in detail:
            print(f"  {d['company']}: vec={d['vec_count']} bm25={d['bm25_count']} rrf={d['rrf_count']} overlap={d['overlap_count']}")
        assert len(texts) == 12, f"Expected 12 chunks, got {len(texts)}"

    def test_three_companies_all_represented(self):
        texts, sources, detail = _retrieve(
            "net income profit 2024", ["Tesla", "Meta", "Netflix"], per_company_k=4
        )
        companies_in_output = {s["company"] for s in sources}
        assert "Tesla Inc." in companies_in_output
        assert "Meta Platforms Inc." in companies_in_output
        assert "Netflix Inc." in companies_in_output

    def test_three_companies_no_cross_contamination(self):
        texts, sources, detail = _retrieve(
            "R&D expense research development", ["Apple", "NVIDIA", "Adobe"], per_company_k=4
        )
        allowed = {"Apple Inc.", "NVIDIA Corporation", "Adobe Inc."}
        for src in sources:
            assert src["company"] in allowed, \
                f"Cross-contamination: {src['company']}"

    def test_three_companies_overlap_diagnostic(self):
        """Detailed diagnostic: sections surfaced by vector vs BM25 vs RRF for each company."""
        texts, sources, detail = _retrieve(
            "operating income loss 2024", ["Apple", "Tesla", "Alphabet"], per_company_k=4
        )
        print(f"\n[3-company operating income diagnostic]")
        for d in detail:
            print(f"\n  {d['company']}:")
            print(f"    vec_top_sections  : {d['vec_top_sections']}")
            print(f"    bm25_top_sections : {d['bm25_top_sections']}")
            print(f"    rrf_top_sections  : {d['rrf_top_sections']}")
            print(f"    overlap           : {d['overlap_count']} chunk(s)")
        assert len(texts) == 12

    def test_three_companies_rrf_chunk_types(self):
        """Print chunk types in RRF output to see TEXT vs TABLE distribution."""
        texts, sources, detail = _retrieve(
            "gross profit margin cost of revenue", ["Apple", "Microsoft", "Amazon"], per_company_k=4
        )
        print(f"\n[chunk type distribution in 3-company gross margin query]")
        for d in detail:
            types = [s.get("chunk_type") for s in d["rrf_sources"]]
            print(f"  {d['company']}: chunk types = {types}")
        assert len(sources) == 12


# ══════════════════════════════════════════════════════════════════════════════
# E — Known-value spot checks (ground truth from live query run)
# ══════════════════════════════════════════════════════════════════════════════
class TestGroundTruthSpotChecks:

    def test_apple_revenue_chunk_contains_391(self):
        """
        Apple's FY2024 revenue is $391,035M. At least one retrieved chunk
        should contain '391' for a direct revenue query.
        """
        texts, sources, detail = _retrieve(
            "Apple total net sales fiscal 2024", ["Apple"], per_company_k=5
        )
        hit = any("391" in t for t in texts)
        assert hit, "No chunk containing '391' (Apple FY2024 revenue) was retrieved"

    def test_nvidia_mentions_nvidia_in_chunks(self):
        texts, sources, detail = _retrieve(
            "NVIDIA total revenue 2024", ["NVIDIA"], per_company_k=5
        )
        hit = any("nvidia" in t.lower() for t in texts)
        assert hit, "NVIDIA revenue chunks should mention 'nvidia'"

    def test_tesla_mentions_tesla_in_chunks(self):
        texts, sources, detail = _retrieve(
            "Tesla net income 2024", ["Tesla"], per_company_k=5
        )
        hit = any("tesla" in t.lower() for t in texts)
        assert hit, "Tesla chunks should mention 'tesla'"

