"""
test_resolver_consistency.py
-----------------------------
Runs the full 12-case adversarial resolver suite N times and reports
per-case pass rate across all runs.

Each case has an explicit grading function — not human eyeball review.
Graders are conservative: they check the MINIMUM correctness requirements
(right company, right metric, no false comparison injection, etc.) rather
than exact string matching, because LLM phrasing varies.

Usage:
    python -u test_resolver_consistency.py [--runs N]   (default: 3)

Output:
    - Per-run results table (raw resolved strings)
    - Per-case consistency summary (pass rate across N runs)
    - Inconsistent cases with all observed output variants
"""

import sys
import os
import time
import argparse
import re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools.context_resolver import resolve_context

# ---------------------------------------------------------------------------
# Test case definitions (same as test_resolver.py)
# ---------------------------------------------------------------------------

def _turn(n, q, a, companies=None):
    return {
        "turn_number": n,
        "raw_question": q,
        "final_answer": a,
        "companies": companies or [],
        "companies_json": "[]",
    }


TESTS = [
    {
        "id": "P1",
        "label": "Passthrough (no history)",
        "history": [],
        "question": "What was Apple's total revenue in fiscal year 2024?",
    },
    {
        "id": "P2",
        "label": "Passthrough (with history, different company)",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales for fiscal 2024 were $391.0 billion.",
                  ["Apple"]),
        ],
        "question": "What was Microsoft's cloud revenue in 2024?",
    },
    {
        "id": "R1",
        "label": "Pronoun 'their' → single company",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales for fiscal 2024 were $391.0 billion.",
                  ["Apple"]),
        ],
        "question": "What about their R&D expense?",
    },
    {
        "id": "R2",
        "label": "Pronoun 'the company' → multi-turn NVIDIA",
        "history": [
            _turn(1, "What was NVIDIA's revenue in 2024?",
                  "NVIDIA's total revenue for fiscal 2024 was approximately $60.9 billion.",
                  ["NVIDIA"]),
            _turn(2, "What was NVIDIA's gross margin?",
                  "NVIDIA achieved a gross margin of approximately 74.6% in fiscal 2024.",
                  ["NVIDIA"]),
        ],
        "question": "How much did the company spend on R&D?",
    },
    {
        "id": "M1",
        "label": "Metric-only follow-up (no company named)",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales for fiscal 2024 were $391.0 billion.",
                  ["Apple"]),
        ],
        "question": "And net income?",
    },
    {
        "id": "M2",
        "label": "Metric-only, two-turn history",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales for fiscal 2024 were $391.0 billion.",
                  ["Apple"]),
            _turn(2, "What about their gross profit?",
                  "Apple's gross profit for fiscal 2024 was approximately $180.7 billion.",
                  ["Apple"]),
        ],
        "question": "What about operating income?",
    },
    {
        "id": "N1",
        "label": "New company 'What about Tesla?' after Apple session",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales for fiscal 2024 were $391.0 billion.",
                  ["Apple"]),
            _turn(2, "What about their gross margin?",
                  "Apple's gross margin for fiscal 2024 was approximately 46.2%.",
                  ["Apple"]),
        ],
        "question": "What about Tesla?",
    },
    {
        "id": "N2",
        "label": "New company 'What about Tesla?' after Apple+Microsoft session",
        "history": [
            _turn(1, "Compare Apple and Microsoft's gross margins",
                  "Apple gross margin: 46.2%. Microsoft gross margin: 70.1%.",
                  ["Apple", "Microsoft"]),
        ],
        "question": "What about Tesla?",
    },
    {
        "id": "C1",
        "label": "Explicit comparison — already self-contained (must NOT be rewritten)",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales were $391.0 billion.",
                  ["Apple"]),
        ],
        "question": "How does Tesla's revenue compare to Apple's in 2024?",
    },
    {
        "id": "B1",
        "label": "'both their' → two companies in history",
        "history": [
            _turn(1, "Compare Apple and Microsoft's revenue in 2024",
                  "Apple: $391B. Microsoft: $245B.",
                  ["Apple", "Microsoft"]),
        ],
        "question": "How do both their gross margins compare?",
    },
    {
        "id": "D1",
        "label": "Definition question (direct route — must be unchanged)",
        "history": [
            _turn(1, "What was Apple's gross margin in 2024?",
                  "Apple's gross margin was approximately 46.2%.",
                  ["Apple"]),
        ],
        "question": "What does gross margin mean?",
    },
    {
        "id": "Y1",
        "label": "Implicit year shift ('what about 2023?')",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales for fiscal 2024 were $391.0 billion.",
                  ["Apple"]),
        ],
        "question": "What about 2023?",
    },
]


# ---------------------------------------------------------------------------
# Graders: return (passed: bool, reason: str)
# ---------------------------------------------------------------------------
# All graders operate on the RESOLVED string and the original QUESTION.
# They are conservative — check minimum requirements, not exact phrasing.
# A grader should NOT fail for benign phrasing variations.

def _has(text, *words):
    """Check all words appear in text (case-insensitive)."""
    t = text.lower()
    return all(w.lower() in t for w in words)

def _lacks_false_comparison(text, prior_companies):
    """True if text does NOT frame Tesla as a comparison against prior companies."""
    t = text.lower()
    # Patterns that indicate false comparison injection
    comparison_words = ["compare", " vs ", " versus ", "compared to", "how does"]
    if not any(w in t for w in comparison_words):
        return True, "no comparison language"
    # If comparison words present, verify it's not comparing Tesla to prior companies
    for co in prior_companies:
        if co.lower() in t and "tesla" in t:
            return False, f"false comparison: Tesla + {co} both appear with comparison language"
    return True, "comparison language present but not cross-company"


GRADERS = {
    "P1": lambda q, r: (
        (r.strip() == q.strip(), "exact match required")
        if r.strip() == q.strip()
        else (False, f"changed: {r!r}")
    ),

    "P2": lambda q, r: (
        (r.strip() == q.strip(), "exact match required")
        if r.strip() == q.strip()
        else (False, f"changed: {r!r}")
    ),

    "R1": lambda q, r: (
        (_has(r, "apple") and (_has(r, "r&d") or _has(r, "research")),
         "must contain Apple + R&D/research")
    ),

    "R2": lambda q, r: (
        (_has(r, "nvidia") and
         (_has(r, "r&d") or _has(r, "research") or _has(r, "spend") or
          _has(r, "research and development") or _has(r, "expense")),
         "must contain NVIDIA + R&D/research/spend/expense")
    ),

    "M1": lambda q, r: (
        (_has(r, "apple") and _has(r, "net income"),
         "must contain Apple + net income")
    ),

    "M2": lambda q, r: (
        (_has(r, "apple") and _has(r, "operating income"),
         "must contain Apple + operating income")
    ),

    "N1": lambda q, r: (
        # Must contain Tesla, must not inject false comparison with Apple
        (_has(r, "tesla") and _lacks_false_comparison(r, ["Apple"])[0],
         "must contain Tesla, no false Apple comparison")
    ),

    "N2": lambda q, r: (
        # Must contain Tesla, must not inject false comparison with Apple or Microsoft
        (_has(r, "tesla") and _lacks_false_comparison(r, ["Apple", "Microsoft"])[0],
         "must contain Tesla, no false Apple/Microsoft comparison")
    ),

    "C1": lambda q, r: (
        # CRITICAL: must be unchanged — this prevents route flip retrieve→calculate
        (r.strip() == q.strip(), "must be UNCHANGED — comparison question already self-contained")
        if r.strip() == q.strip()
        else (False, f"REWRITTEN (breaks calculate route): {r!r}")
    ),

    "B1": lambda q, r: (
        (_has(r, "apple") and _has(r, "microsoft") and
         (_has(r, "margin") or _has(r, "gross")),
         "must contain Apple, Microsoft, and margin/gross")
    ),

    "D1": lambda q, r: (
        (r.strip() == q.strip(), "must be UNCHANGED — definition question")
        if r.strip() == q.strip()
        else (False, f"changed: {r!r}")
    ),

    "Y1": lambda q, r: (
        (_has(r, "apple") and
         (_has(r, "revenue") or _has(r, "sales") or _has(r, "net sales")) and
         "2023" in r,
         "must contain Apple + revenue/sales + 2023")
    ),
}


def grade(test_id, question, resolved):
    grader = GRADERS.get(test_id)
    if grader is None:
        return False, "no grader defined"
    result = grader(question, resolved)
    # Handle tuple return vs bool
    if isinstance(result, tuple):
        passed, reason = result
    else:
        passed = result
        reason = ""
    return bool(passed), reason


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_suite(run_number, delay_between_calls=1.0):
    """Run all 12 cases once. Returns list of (test_id, resolved, passed, reason)."""
    results = []
    print(f"\n{'='*64}")
    print(f"  RUN {run_number}")
    print(f"{'='*64}")

    for i, case in enumerate(TESTS):
        if i > 0:
            time.sleep(delay_between_calls)

        test_id = case["id"]
        question = case["question"]
        history = case["history"]

        try:
            resolved = resolve_context(question, history)
        except Exception as exc:
            resolved = f"ERROR: {exc}"

        passed, reason = grade(test_id, question, resolved)
        changed = resolved.strip() != question.strip()
        change_marker = "CHANGED" if changed else "unchanged"
        pass_marker = "✅" if passed else "❌"

        print(f"  {pass_marker} {test_id:4s} [{change_marker:9s}]  {resolved!r}")
        if not passed:
            print(f"        ↳ FAIL reason: {reason}")

        results.append((test_id, resolved, passed, reason))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of full suite runs (default: 3)")
    args = parser.parse_args()

    import config
    print(f"\nCONTEXT RESOLVER — Consistency Test")
    print(f"Model : {config.MODEL_REWRITE}")
    print(f"Runs  : {args.runs}")
    print(f"Cases : {len(TESTS)}")

    # all_results[test_id] = list of (resolved, passed) across runs
    all_results = defaultdict(list)

    for run_num in range(1, args.runs + 1):
        run_results = run_suite(run_num, delay_between_calls=0.5)
        for test_id, resolved, passed, reason in run_results:
            all_results[test_id].append((resolved, passed))

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print(f"\n{'='*64}")
    print(f"  CONSISTENCY SUMMARY ({args.runs} runs)")
    print(f"{'='*64}")
    print(f"  {'ID':4s}  {'Pass Rate':10s}  {'Consistent?':12s}  {'Label'}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*12}  {'-'*30}")

    inconsistent_cases = []

    for case in TESTS:
        tid = case["id"]
        runs_data = all_results[tid]
        pass_count = sum(1 for _, p in runs_data if p)
        pass_rate = f"{pass_count}/{args.runs}"

        resolved_variants = [r for r, _ in runs_data]
        unique_outputs = list(dict.fromkeys(resolved_variants))  # preserve order, dedupe
        is_consistent = len(unique_outputs) == 1
        consistency_label = "✅ consistent" if is_consistent else "⚠️  VARIES"

        print(f"  {tid:4s}  {pass_rate:10s}  {consistency_label:12s}  {case['label']}")

        if not is_consistent or pass_count < args.runs:
            inconsistent_cases.append({
                "id": tid,
                "label": case["label"],
                "pass_rate": pass_rate,
                "variants": unique_outputs,
                "all_results": runs_data,
            })

    # -------------------------------------------------------------------------
    # Variant detail for inconsistent/failing cases
    # -------------------------------------------------------------------------
    if inconsistent_cases:
        print(f"\n{'='*64}")
        print(f"  INCONSISTENT / FAILING CASES — DETAIL")
        print(f"{'='*64}")
        for c in inconsistent_cases:
            print(f"\n  {c['id']}: {c['label']}")
            print(f"  Pass rate: {c['pass_rate']}")
            print(f"  Unique output variants ({len(c['variants'])}):")
            for i, v in enumerate(c["variants"], 1):
                count = sum(1 for r, _ in c["all_results"] if r == v)
                passed_count = sum(1 for r, p in c["all_results"] if r == v and p)
                print(f"    Variant {i} [{count}x seen, {passed_count}/{count} passed]: {v!r}")
    else:
        print(f"\n  All cases consistent and passing across {args.runs} runs. ✅")

    print(f"\n{'='*64}")
    total_passes = sum(
        1 for case in TESTS
        for _, passed in all_results[case["id"]]
        if passed
    )
    total_possible = len(TESTS) * args.runs
    print(f"  TOTAL: {total_passes}/{total_possible} across all runs")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
