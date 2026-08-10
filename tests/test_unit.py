"""
tests/test_unit.py
Phase 1 — Pure unit tests. Zero external deps (no Groq, no Chroma, no network).
"""
import pytest
import sys
import os
import struct
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ============================================================
# 1A — calculator.compute()
# ============================================================
from tools.retrieval.calculator import compute

class TestCalculator:
    def test_percent_change_positive(self):
        vals = [{"label": "old", "value": 100}, {"label": "new", "value": 150}]
        assert compute("percent_change", vals) == 50.0

    def test_percent_change_negative(self):
        vals = [{"label": "old", "value": 150}, {"label": "new", "value": 100}]
        assert compute("percent_change", vals) == -33.33

    def test_percent_change_zero_base(self):
        vals = [{"label": "old", "value": 0}, {"label": "new", "value": 100}]
        with pytest.raises(ValueError, match="base of 0"):
            compute("percent_change", vals)

    def test_difference(self):
        vals = [{"label": "a", "value": 391}, {"label": "b", "value": 383}]
        assert compute("difference", vals) == 8

    def test_sum(self):
        vals = [{"label": "a", "value": 10}, {"label": "b", "value": 20}, {"label": "c", "value": 30}]
        assert compute("sum", vals) == 60.0

    def test_average(self):
        vals = [{"label": "a", "value": 10}, {"label": "b", "value": 20}, {"label": "c", "value": 30}]
        assert compute("average", vals) == 20.0

    def test_ratio(self):
        vals = [{"label": "num", "value": 391}, {"label": "den", "value": 2}]
        assert compute("ratio", vals) == 195.5

    def test_ratio_zero_denominator(self):
        vals = [{"label": "num", "value": 100}, {"label": "den", "value": 0}]
        with pytest.raises(ValueError, match="denominator of 0"):
            compute("ratio", vals)

    def test_margin(self):
        vals = [{"label": "revenue", "value": 100}, {"label": "cogs", "value": 70}]
        assert compute("margin", vals) == 30.0

    def test_margin_zero_base(self):
        vals = [{"label": "revenue", "value": 0}, {"label": "cogs", "value": 10}]
        with pytest.raises(ValueError, match="base of 0"):
            compute("margin", vals)

    def test_max_returns_dict_with_highest(self):
        vals = [
            {"label": "A", "value": 100},
            {"label": "B", "value": 500},
            {"label": "C", "value": 250},
        ]
        result = compute("max", vals)
        assert isinstance(result, dict)
        assert result["label"] == "B"
        assert result["value"] == 500

    def test_min_returns_dict_with_lowest(self):
        vals = [
            {"label": "A", "value": 100},
            {"label": "B", "value": 500},
            {"label": "C", "value": 250},
        ]
        result = compute("min", vals)
        assert isinstance(result, dict)
        assert result["label"] == "A"
        assert result["value"] == 100

    def test_unsupported_operation(self):
        vals = [{"label": "a", "value": 5}, {"label": "b", "value": 3}]
        with pytest.raises(ValueError, match="Unsupported operation"):
            compute("multiply", vals)


# ============================================================
# 1B — output_parsers
# ============================================================
from tools.retrieval.output_parsers import (
    parse_hallucination, parse_grade, parse_route,
    parse_query_analysis, parse_calculation
)

class TestParseHallucination:
    def test_grounded(self):
        assert parse_hallucination("The answer is grounded.") == "grounded"

    def test_not_grounded(self):
        assert parse_hallucination("not_grounded") == "not_grounded"

    def test_substring_trap_not_grounded_wins(self):
        # 'not_grounded' contains 'grounded' as substring —
        # code must check not_grounded FIRST
        assert parse_hallucination("this is not_grounded in any way") == "not_grounded"

    def test_garbled_defaults_to_not_grounded(self):
        assert parse_hallucination("error processing request") == "not_grounded"

    def test_multiline_checks_last_line(self):
        text = "Some long reasoning.\nThe answer seems right.\ngrounded"
        assert parse_hallucination(text) == "grounded"

    def test_multiline_not_grounded_last(self):
        text = "Some long reasoning.\nThe answer seems off.\nnot_grounded"
        assert parse_hallucination(text) == "not_grounded"


class TestParseGrade:
    def test_yes(self):
        assert parse_grade("yes, this is relevant") == "yes"

    def test_no(self):
        assert parse_grade("no, this is irrelevant") == "no"

    def test_uppercase_yes(self):
        assert parse_grade("YES") == "yes"

    def test_multiline_yes_last(self):
        assert parse_grade("Reasoning about the question.\nyes") == "yes"

    def test_garbled_defaults_to_no(self):
        assert parse_grade("unable to determine") == "no"

    def test_no_is_safe_default_for_empty(self):
        assert parse_grade("") == "no"


class TestParseRoute:
    def test_calculate(self):
        assert parse_route("This requires calculate math") == "calculate"

    def test_compute_keyword(self):
        assert parse_route("You need to compute the value") == "calculate"

    def test_math_keyword(self):
        assert parse_route("Use math to get the answer") == "calculate"

    def test_direct(self):
        assert parse_route("This is a direct definition") == "direct"

    def test_general_keyword(self):
        assert parse_route("This is general knowledge") == "direct"

    def test_definition_keyword(self):
        assert parse_route("definition of amortization") == "direct"

    def test_retrieve_default(self):
        assert parse_route("The answer is in the filing somewhere") == "retrieve"

    def test_empty_defaults_to_retrieve(self):
        assert parse_route("") == "retrieve"


class TestParseQueryAnalysis:
    def test_clean_json(self):
        result = parse_query_analysis('{"companies": ["Apple"], "metric_category": "revenue_sales"}')
        assert result == {"companies": ["Apple"], "metric_category": "revenue_sales"}

    def test_code_fence_stripped(self):
        text = '```json\n{"companies": ["NVIDIA"], "metric_category": "r_and_d"}\n```'
        result = parse_query_analysis(text)
        assert result["companies"] == ["NVIDIA"]
        assert result["metric_category"] == "r_and_d"

    def test_malformed_json_fallback(self):
        result = parse_query_analysis("{bad json}")
        assert result == {"companies": ["all"], "metric_category": "general"}

    def test_missing_companies_key_fallback(self):
        result = parse_query_analysis('{"metric_category": "revenue_sales"}')
        assert result == {"companies": ["all"], "metric_category": "general"}

    def test_missing_metric_key_fallback(self):
        result = parse_query_analysis('{"companies": ["Apple"]}')
        assert result == {"companies": ["all"], "metric_category": "general"}

    def test_non_dict_fallback(self):
        result = parse_query_analysis('"just a string"')
        assert result == {"companies": ["all"], "metric_category": "general"}


class TestParseCalculation:
    def test_clean_json(self):
        text = '{"operation": "sum", "values": [{"label": "a", "value": 10}]}'
        result = parse_calculation(text)
        assert result["operation"] == "sum"
        assert len(result["values"]) == 1

    def test_json_after_prose(self):
        text = 'Some explanation here. {"operation": "percent_change", "values": [{"label": "x", "value": 1}]}'
        result = parse_calculation(text)
        assert result["operation"] == "percent_change"

    def test_no_json_returns_insufficient_data(self):
        result = parse_calculation("I cannot find the numbers.")
        assert result == {"operation": "insufficient_data", "values": []}

    def test_json_missing_operation_key(self):
        # Missing 'operation' key → insufficient_data fallback
        result = parse_calculation('{"values": [{"label": "a", "value": 1}]}')
        assert result == {"operation": "insufficient_data", "values": []}

    def test_json_missing_values_key(self):
        result = parse_calculation('{"operation": "sum"}')
        assert result == {"operation": "insufficient_data", "values": []}


# ============================================================
# 1C — Edge routing functions
# ============================================================
from graph.edges import (
    route_after_router, route_after_cache, route_after_grade,
    route_by_calc_type, route_after_hallucination
)
import config as _config

def make_state(**kwargs):
    """Build a minimal GraphState dict for edge testing."""
    defaults = {
        "question": "test",
        "rewritten_question": "",
        "route": "retrieve",
        "companies_mentioned": ["Apple"],
        "retrieved_chunks": [],
        "chunk_sources": [],
        "relevant": "no",
        "answer": "",
        "grounded": "not_grounded",
        "retry_count": 0,
        "final_answer": "",
        "error_message": None,
        "cache_hit": False,
        "conversation_context": None,
        "metric_category": "revenue_sales",
    }
    defaults.update(kwargs)
    return defaults

class TestEdgeRouting:
    # route_after_router
    def test_router_direct(self):
        assert route_after_router(make_state(route="direct")) == "direct"

    def test_router_retrieve_goes_to_cache(self):
        assert route_after_router(make_state(route="retrieve")) == "cache_lookup"

    def test_router_calculate_goes_to_cache(self):
        assert route_after_router(make_state(route="calculate")) == "cache_lookup"

    # route_after_cache
    def test_cache_hit_retrieve_route_goes_to_generate(self):
        assert route_after_cache(make_state(cache_hit=True, route="retrieve")) == "generate"

    def test_cache_hit_calculate_route_goes_to_calculate(self):
        assert route_after_cache(make_state(cache_hit=True, route="calculate")) == "calculate"

    def test_cache_miss_goes_to_retrieve(self):
        assert route_after_cache(make_state(cache_hit=False)) == "retrieve"

    # route_after_grade
    def test_grade_relevant_retrieve_goes_to_generate(self):
        assert route_after_grade(make_state(relevant="yes", route="retrieve")) == "generate"

    def test_grade_relevant_calculate_goes_to_calculate(self):
        assert route_after_grade(make_state(relevant="yes", route="calculate")) == "calculate"

    def test_grade_not_relevant_retry_available(self):
        assert route_after_grade(make_state(relevant="no", retry_count=0)) == "rewrite"

    def test_grade_not_relevant_retry_at_limit(self):
        assert route_after_grade(make_state(relevant="no", retry_count=_config.MAX_RETRY)) == "exhausted"

    def test_grade_not_relevant_one_before_limit(self):
        assert route_after_grade(make_state(relevant="no", retry_count=_config.MAX_RETRY - 1)) == "rewrite"

    # route_by_calc_type
    def test_calc_type_calculate(self):
        assert route_by_calc_type(make_state(route="calculate")) == "calculate"

    def test_calc_type_retrieve_goes_to_generate(self):
        assert route_by_calc_type(make_state(route="retrieve")) == "generate"

    # route_after_hallucination
    def test_halluc_grounded_goes_to_end(self):
        assert route_after_hallucination(make_state(grounded="grounded")) == "end"

    def test_halluc_not_grounded_retry_retrieve(self):
        state = make_state(grounded="not_grounded", retry_count=0, route="retrieve")
        assert route_after_hallucination(state) == "generate"

    def test_halluc_not_grounded_retry_calculate(self):
        state = make_state(grounded="not_grounded", retry_count=0, route="calculate")
        assert route_after_hallucination(state) == "calculate"

    def test_halluc_not_grounded_exhausted(self):
        state = make_state(grounded="not_grounded", retry_count=_config.MAX_RETRY)
        assert route_after_hallucination(state) == "exhausted"


# ============================================================
# 1D — session_store CRUD (temp SQLite)
# ============================================================
import uuid as _uuid

class TestSessionStore:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "SESSION_DB_PATH", str(tmp_path / "ss_test.db"))
        import tools.session.session_store as ss
        ss._init_db()
        self.ss = ss

    def test_create_returns_uuid(self):
        sid = self.ss.create_session()
        _uuid.UUID(sid)  # raises ValueError if not a valid UUID

    def test_get_after_create(self):
        sid = self.ss.create_session()
        session = self.ss.get_session(sid)
        assert session is not None
        assert session["session_id"] == sid
        assert "created_at" in session
        assert "last_active" in session

    def test_get_nonexistent_returns_none(self):
        assert self.ss.get_session("nonexistent-id") is None

    def test_list_sessions_ordering(self):
        import time
        s1 = self.ss.create_session()
        time.sleep(0.01)
        s2 = self.ss.create_session()
        sessions = self.ss.list_sessions()
        ids = [s["session_id"] for s in sessions]
        # s2 is more recent, should appear first
        assert ids.index(s2) < ids.index(s1)

    def test_add_turn_increments_turn_number(self):
        sid = self.ss.create_session()
        for i in range(3):
            self.ss.add_turn(sid, f"Q{i}", f"Q{i}", "retrieve", ["Apple"], f"A{i}")
        history = self.ss.get_history(sid, last_n=10)
        assert [t["turn_number"] for t in history] == [1, 2, 3]

    def test_first_turn_sets_title(self):
        sid = self.ss.create_session()
        self.ss.add_turn(sid, "What was Apple's revenue?", "What was Apple's revenue?", "retrieve", ["Apple"], "A lot.")
        session = self.ss.get_session(sid)
        assert session["title"] == "What was Apple's revenue?"

    def test_first_turn_title_truncated_at_50(self):
        sid = self.ss.create_session()
        long_q = "A" * 100
        self.ss.add_turn(sid, long_q, long_q, "retrieve", [], "answer")
        session = self.ss.get_session(sid)
        assert len(session["title"]) == 50

    def test_second_turn_updates_last_active(self):
        import time
        sid = self.ss.create_session()
        self.ss.add_turn(sid, "Q1", "Q1", "retrieve", [], "A1")
        t_after_first = self.ss.get_session(sid)["last_active"]
        time.sleep(0.02)
        self.ss.add_turn(sid, "Q2", "Q2", "retrieve", [], "A2")
        t_after_second = self.ss.get_session(sid)["last_active"]
        assert t_after_second > t_after_first

    def test_get_history_oldest_first(self):
        sid = self.ss.create_session()
        for i in range(3):
            self.ss.add_turn(sid, f"Q{i}", f"Q{i}", "retrieve", [], f"A{i}")
        history = self.ss.get_history(sid)
        assert history[0]["raw_question"] == "Q0"
        assert history[-1]["raw_question"] == "Q2"

    def test_get_history_last_n_cap(self):
        sid = self.ss.create_session()
        for i in range(10):
            self.ss.add_turn(sid, f"Q{i}", f"Q{i}", "retrieve", [], f"A{i}")
        history = self.ss.get_history(sid, last_n=3)
        assert len(history) == 3
        # Should be the last 3 (oldest first within the cap)
        assert history[-1]["raw_question"] == "Q9"

    def test_get_history_companies_parsed_as_list(self):
        sid = self.ss.create_session()
        self.ss.add_turn(sid, "Q", "Q", "retrieve", ["Apple", "Tesla"], "A")
        history = self.ss.get_history(sid)
        assert isinstance(history[0]["companies"], list)
        assert "Apple" in history[0]["companies"]
        assert "Tesla" in history[0]["companies"]

    def test_delete_session_removes_record(self):
        sid = self.ss.create_session()
        self.ss.add_turn(sid, "Q", "Q", "retrieve", [], "A")
        self.ss.delete_session(sid)
        assert self.ss.get_session(sid) is None


# ============================================================
# 1E — retrieval_cache math helpers (no SQLite, no Groq)
# ============================================================
from tools.session.retrieval_cache import _pack_vector, _unpack_vector, _cosine_similarity

class TestCacheHelpers:
    def test_pack_unpack_roundtrip(self):
        original = [0.1, 0.5, -0.3, 0.9, 1.0]
        packed = _pack_vector(original)
        unpacked = _unpack_vector(packed)
        # float32 precision — compare within tolerance
        assert np.allclose(unpacked, original, atol=1e-5)

    def test_cosine_similarity_identical(self):
        v = np.array([1.0, 0.5, -0.3], dtype=np.float32)
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(_cosine_similarity(v1, v2)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.5], dtype=np.float32)
        assert _cosine_similarity(v1, v2) == 0.0

    def test_cosine_similarity_opposite(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0], dtype=np.float32)
        assert abs(_cosine_similarity(v1, v2) - (-1.0)) < 1e-6


# ============================================================
# 1F — company_names
# ============================================================
from tools.retrieval.company_names import SHORT_TO_FULL, get_all_full_names

class TestCompanyNames:
    def test_apple_full_name(self):
        assert SHORT_TO_FULL["Apple"] == "Apple Inc."

    def test_microsoft_full_name(self):
        assert SHORT_TO_FULL["Microsoft"] == "Microsoft Corporation"

    def test_nvidia_full_name(self):
        assert SHORT_TO_FULL["NVIDIA"] == "NVIDIA Corporation"

    def test_tesla_full_name(self):
        assert SHORT_TO_FULL["Tesla"] == "Tesla Inc."

    def test_google_alias_maps_to_alphabet(self):
        assert SHORT_TO_FULL["Google"] == "Alphabet Inc."

    def test_alphabet_maps_to_alphabet(self):
        assert SHORT_TO_FULL["Alphabet"] == "Alphabet Inc."

    def test_all_full_names_is_list(self):
        names = get_all_full_names()
        assert isinstance(names, list)

    def test_all_full_names_contains_apple(self):
        assert "Apple Inc." in get_all_full_names()

    def test_alphabet_appears_twice_due_to_alias(self):
        # Google and Alphabet both map to Alphabet Inc. — appears twice in values
        names = get_all_full_names()
        assert names.count("Alphabet Inc.") == 2
