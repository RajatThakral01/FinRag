"""
test_resolver.py — Adversarial test set for context_resolver.resolve_context()

Run from project root:
    python -u test_resolver.py

Tests the resolver in ISOLATION — does NOT invoke the graph, retrieve chunks,
or call any LLM other than the resolver itself (MODEL_REWRITE / 8B).

Each test case defines:
  - description:  what the test is checking
  - history:      simulated prior turns (list of dicts matching session_store schema)
  - question:     raw input to the resolver
  - expect_note:  what we EXPECT to see (not asserted — we're observing LLM behavior,
                  not running a deterministic unit test)
  - risk:         what failure looks like if the resolver gets it wrong

The 12 cases cover:
  [P] Passthrough (standalone questions that must NOT be rewritten)
  [R] Pronoun resolution
  [M] Implicit metric (metric-only follow-up)
  [N] New-company shift (the trickiest case — ambiguity between "new topic" vs "compare")
  [C] Explicit comparison (should pass through as-is, already self-contained)
  [B] Both-their (two companies in history)
  [D] Definition / direct route (should be unchanged)
  [Y] Implicit year reference
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.context_resolver import resolve_context

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _turn(n, q, a, companies=None):
    """Build a minimal turn dict matching the session_store schema."""
    return {
        "turn_number": n,
        "raw_question": q,
        "final_answer": a,
        "companies": companies or [],
        "companies_json": "[]",
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TESTS = [

    # ------------------------------------------------------------------ [P1]
    {
        "id": "P1",
        "description": "[PASSTHROUGH] Already self-contained — must return UNCHANGED",
        "history": [],
        "question": "What was Apple's total revenue in fiscal year 2024?",
        "expect_note": "Return identical to input — no changes whatsoever",
        "risk":       "Any rewrite here degrades downstream prompt precision",
    },

    # ------------------------------------------------------------------ [P2]
    {
        "id": "P2",
        "description": "[PASSTHROUGH] Standalone question WITH history — must still be unchanged",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales for fiscal 2024 were $391.0 billion.",
                  ["Apple"]),
        ],
        "question": "What was Microsoft's cloud revenue in 2024?",
        "expect_note": "Different company, fully specified — return unchanged. "
                       "Resolver must not add 'compare to Apple'",
        "risk":       "False comparison injection",
    },

    # ------------------------------------------------------------------ [R1]
    {
        "id": "R1",
        "description": "[PRONOUN] 'their' → single company in history",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales for fiscal 2024 were $391.0 billion.",
                  ["Apple"]),
        ],
        "question": "What about their R&D expense?",
        "expect_note": "Should resolve to: 'What was Apple's R&D expense?' or equivalent. "
                       "Must contain 'Apple' and 'R&D'",
        "risk":       "Returns 'their R&D' unchanged → graph company extractor returns ['all'] "
                       "→ 9-company retrieval → wrong answer",
    },

    # ------------------------------------------------------------------ [R2]
    {
        "id": "R2",
        "description": "[PRONOUN] 'the company' reference with multi-turn history",
        "history": [
            _turn(1, "What was NVIDIA's revenue in 2024?",
                  "NVIDIA's total revenue for fiscal 2024 was approximately $60.9 billion.",
                  ["NVIDIA"]),
            _turn(2, "What was NVIDIA's gross margin?",
                  "NVIDIA achieved a gross margin of approximately 74.6% in fiscal 2024.",
                  ["NVIDIA"]),
        ],
        "question": "How much did the company spend on R&D?",
        "expect_note": "Should resolve to 'How much did NVIDIA spend on R&D?' — "
                       "most recent company (NVIDIA) from history",
        "risk":       "Returns unchanged → 'the company' passes to router → company extractor "
                       "confused → ['all'] retrieval",
    },

    # ------------------------------------------------------------------ [M1]
    {
        "id": "M1",
        "description": "[METRIC] Implicit metric — no company named in follow-up",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales for fiscal 2024 were $391.0 billion.",
                  ["Apple"]),
        ],
        "question": "And net income?",
        "expect_note": "Should resolve to: 'What was Apple's net income?' or "
                       "'What was Apple's net income in 2024?' (year optional)",
        "risk":       "Returns 'And net income?' unchanged → router/extractor returns ['all']",
    },

    # ------------------------------------------------------------------ [M2]
    {
        "id": "M2",
        "description": "[METRIC] 'What about operating income?' after two-turn Apple session",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales for fiscal 2024 were $391.0 billion.",
                  ["Apple"]),
            _turn(2, "What about their gross profit?",
                  "Apple's gross profit for fiscal 2024 was approximately $180.7 billion.",
                  ["Apple"]),
        ],
        "question": "What about operating income?",
        "expect_note": "Should resolve to Apple's operating income — company carries through",
        "risk":       "Picks wrong company or returns with ['all'] context",
    },

    # ------------------------------------------------------------------ [N1]  ← THE KEY CASE
    {
        "id": "N1",
        "description": "[NEW COMPANY] 'What about Tesla?' after Apple session "
                       "— ambiguity: new topic vs compare?",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales for fiscal 2024 were $391.0 billion.",
                  ["Apple"]),
            _turn(2, "What about their gross margin?",
                  "Apple's gross margin for fiscal 2024 was approximately 46.2%.",
                  ["Apple"]),
        ],
        "question": "What about Tesla?",
        "expect_note": "SHOULD produce a standalone Tesla question — e.g. 'What was "
                       "Tesla's revenue in 2024?' or just 'What are Tesla's financials?'. "
                       "Must NOT say 'Compare Tesla to Apple'. No comparison signal in input.",
        "risk":       "Injecting 'Compare Tesla to Apple' is a false comparison that changes "
                       "the route (retrieve→calculate), the companies list (['Apple','Tesla']), "
                       "the grading criteria, and the final answer structure",
    },

    # ------------------------------------------------------------------ [N2]
    {
        "id": "N2",
        "description": "[NEW COMPANY] 'What about Tesla?' but after a CALCULATE query "
                       "comparing Apple and Microsoft",
        "history": [
            _turn(1, "Compare Apple and Microsoft's gross margins",
                  "Apple gross margin: 46.2%. Microsoft gross margin: 70.1%.",
                  ["Apple", "Microsoft"]),
        ],
        "question": "What about Tesla?",
        "expect_note": "Should still treat as a NEW standalone Tesla question — NOT 'Compare "
                       "Tesla to Apple and Microsoft'. The input has no comparison signal.",
        "risk":       "Introducing a 3-way comparison the user never asked for",
    },

    # ------------------------------------------------------------------ [C1]
    {
        "id": "C1",
        "description": "[EXPLICIT COMPARE] Explicit comparison — already self-contained",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales were $391.0 billion.",
                  ["Apple"]),
        ],
        "question": "How does Tesla's revenue compare to Apple's in 2024?",
        "expect_note": "Already self-contained — return UNCHANGED. Both companies named "
                       "explicitly, comparison signal present ('compare').",
        "risk":       "Unnecessary rewrite might drop one company or change phrasing",
    },

    # ------------------------------------------------------------------ [B1]
    {
        "id": "B1",
        "description": "[BOTH-THEIR] 'both their margins' after two-company session",
        "history": [
            _turn(1, "Compare Apple and Microsoft's revenue in 2024",
                  "Apple: $391B. Microsoft: $245B.",
                  ["Apple", "Microsoft"]),
        ],
        "question": "How do both their gross margins compare?",
        "expect_note": "Should resolve to: 'How do Apple's and Microsoft's gross margins "
                       "compare?' or 'Compare Apple and Microsoft gross margins'",
        "risk":       "Leaving 'both their' unresolved → company extractor returns ['all'] "
                       "→ 9-company retrieval instead of 2",
    },

    # ------------------------------------------------------------------ [D1]
    {
        "id": "D1",
        "description": "[DIRECT/DEFINITION] General concept question — must pass through unchanged",
        "history": [
            _turn(1, "What was Apple's gross margin in 2024?",
                  "Apple's gross margin was approximately 46.2%.",
                  ["Apple"]),
        ],
        "question": "What does gross margin mean?",
        "expect_note": "Definition question — no pronouns, no company reference. "
                       "Must return UNCHANGED. This is a 'direct' route question.",
        "risk":       "Resolver adds 'for Apple' → routes to retrieve instead of direct",
    },

    # ------------------------------------------------------------------ [Y1]
    {
        "id": "Y1",
        "description": "[YEAR] Implicit year — 'what about 2023?' after a 2024 question",
        "history": [
            _turn(1, "What was Apple's revenue in 2024?",
                  "Apple's total net sales for fiscal 2024 were $391.0 billion.",
                  ["Apple"]),
        ],
        "question": "What about 2023?",
        "expect_note": "Should resolve to: 'What was Apple's revenue in 2023?' "
                       "Metric (revenue) carries from history. Year changes to 2023.",
        "risk":       "Returns '2023' unchanged → router/extractor has no company or metric → "
                       "likely routes to retrieve with ['all'] and gets garbage",
    },
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

PASS_INDICATOR = {
    "P": "🟦 PASSTHROUGH",
    "R": "🟧 PRONOUN",
    "M": "🟨 METRIC",
    "N": "🔴 NEW-COMPANY",   # highlighted because it's the highest-risk category
    "C": "🟦 EXPLICIT-CMP",
    "B": "🟧 BOTH-THEIR",
    "D": "🟦 DEFINITION",
    "Y": "🟨 YEAR",
}


def run_tests():
    print("=" * 72)
    print("CONTEXT RESOLVER — Adversarial Test Suite")
    print(f"Model: {__import__('config').MODEL_REWRITE}")
    print("=" * 72)
    print()

    for i, case in enumerate(TESTS, 1):
        test_type = case["id"][0]
        type_label = PASS_INDICATOR.get(test_type, "⬜ UNKNOWN")

        print(f"─── Test {case['id']}: {type_label} ──────────────────────────────")
        print(f"Description : {case['description']}")

        if case["history"]:
            print(f"History     : {len(case['history'])} turn(s)")
            for t in case["history"]:
                companies_tag = f" [{', '.join(t['companies'])}]" if t["companies"] else ""
                print(f"  Q{t['turn_number']}: {t['raw_question']!r}{companies_tag}")
                answer_preview = (t["final_answer"] or "")[:80]
                print(f"  A{t['turn_number']}: {answer_preview!r}")
        else:
            print("History     : (empty — first question in session)")

        print(f"Input Q     : {case['question']!r}")

        try:
            resolved = resolve_context(case["question"], case["history"])
            changed = resolved.strip() != case["question"].strip()
            change_marker = "⚠ CHANGED" if changed else "✓ unchanged"
            print(f"Resolved    : {resolved!r}  [{change_marker}]")
        except Exception as exc:
            print(f"Resolved    : ERROR — {exc}")
            resolved = None

        print(f"Expected    : {case['expect_note']}")
        print(f"Failure risk: {case['risk']}")
        print()

    print("=" * 72)
    print("Done. Review each 'Resolved' line against 'Expected' manually.")
    print("Pay particular attention to N1 and N2 (new-company shift cases).")
    print("=" * 72)


if __name__ == "__main__":
    run_tests()
