"""
tools/context_resolver.py
--------------------------
Context resolution for multi-turn financial Q&A sessions.

Public API:
    resolve_context(question: str, history: list[dict]) -> str

Converts a raw follow-up question (which may contain pronouns, implicit
company references, or topic-shift signals) into a fully self-contained
standalone question that the existing graph can process without any session
context.

Key design constraints:
  - If the input is already self-contained, return it UNCHANGED.
    The resolver must never "improve" standalone questions — the downstream
    prompts in nodes.py are tuned for clean, direct questions.
  - Uses MODEL_REWRITE (8B) — same class as rewrite_node (reformulation,
    not reasoning). The 70B model is not needed here.
  - History is truncated to CONTEXT_WINDOW turns and answers are capped at
    300 chars to prevent prompt bloat on long sessions.
  - Defensive prefix stripping handles models that prepend "Resolved: "
    despite instructions not to.

Adversarial scenarios this prompt is designed to handle correctly:
  1. Clear pronoun:   "What about their R&D?" → "What was Apple's R&D expense?"
  2. Implicit metric: "And net income?"        → "What was Apple's net income?"
  3. New company shift — no comparison signal:
     "What about Tesla?" after Apple Q         → "What was Tesla's [metric]?"
     NOT "Compare Tesla to Apple" (ambiguity rule: simpler interpretation wins)
  4. Explicit comparison:
     "How does Tesla compare to Apple on revenue?" → returned unchanged
  5. Both-their after two companies:
     "How do both their margins compare?" → "Compare Apple and Microsoft gross margins"
  6. Standalone passthrough: "What was Apple's revenue in 2024?" → unchanged
  7. Definition passthrough: "What does EBITDA mean?" → unchanged
"""

import config
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

context_resolve_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a context resolver for a financial Q&A system. The system has "
     "10-K filings for 9 tech companies: Apple, Microsoft, Amazon, NVIDIA, "
     "Tesla, Meta, Alphabet (Google), Netflix, Adobe.\n\n"
     "You receive a conversation history (prior Q&A turns) and a follow-up "
     "question from the user. Rewrite the follow-up into a COMPLETE, "
     "SELF-CONTAINED question that requires NO prior context to understand.\n\n"

     "RULES — apply in this exact order:\n\n"

     "RULE 1 — PASSTHROUGH (most important): If the question already names "
     "a specific company, mentions a specific metric with no ambiguous pronoun, "
     "and makes sense as a standalone question — return it WORD FOR WORD, "
     "unchanged. Do not rephrase, paraphrase, or 'clean up' a self-contained "
     "question. Even minor rewording can change what the downstream system "
     "retrieves. "
     "CRITICAL SPECIAL CASE: if the question already contains a comparison "
     "signal ('compare', 'vs', 'versus', 'compared to', 'how does X compare') "
     "AND names both companies or entities explicitly — return it WORD FOR "
     "WORD, no exceptions. A self-resolved comparison question must NEVER be "
     "rephrased, even to 'clean up' the wording.\n\n"

     "RULE 2 — PRONOUN RESOLUTION: Resolve pronouns ('their', 'its', 'the "
     "company', 'that company') to the most recently mentioned company in the "
     "history. If two companies were discussed, 'their' or 'both their' refers "
     "to both; resolve by naming both explicitly.\n\n"

     "RULE 3 — IMPLICIT METRIC (metric-only follow-up): If the question asks "
     "about a metric without naming a company (e.g. 'What about net income?', "
     "'And operating income?'), and the most recent turn named a company, "
     "prepend that company: 'What was [company]'s [metric]?'\n\n"

     "RULE 4 — NEW COMPANY, NO COMPARISON SIGNAL: If the question introduces "
     "a DIFFERENT company from the previous turn (e.g. 'What about Tesla?' "
     "after an Apple discussion), treat it as a NEW standalone question about "
     "that company UNLESS the question contains an explicit comparison word "
     "('compare', 'vs', 'versus', 'compared to', 'how does X compare'). "
     "Do NOT assume comparison. "
     "If the new-company question is incomplete or vague (e.g. just 'What "
     "about Tesla?' with no metric), expand it into a full standalone question "
     "using the SAME metric from the most recent prior turn: "
     "'What was Tesla's [prior metric]?' This is better than returning a bare "
     "fragment that the downstream system cannot retrieve against.\n\n"

     "RULE 5 — EXPLICIT COMPARISON: If the question explicitly says to compare "
     "two companies and names both, resolve any pronouns and return the "
     "comparison question with both companies named explicitly.\n\n"

     "RULE 6 — NO INVENTED CONTEXT: Never add years, fiscal periods, section "
     "names, note numbers, or figures that appear in NEITHER the question NOR "
     "the history. If the original question had no year, the resolved question "
     "has no year.\n\n"

     "OUTPUT: Return ONLY the resolved question — one sentence or phrase. "
     "No explanation, no prefix like 'Resolved:', no alternatives, no quotes."),

    ("human",
     "Conversation history:\n{history}\n\n"
     "Follow-up question: {question}\n\n"
     "Resolved question:")
])


# ---------------------------------------------------------------------------
# History formatter
# ---------------------------------------------------------------------------

_ANSWER_TRUNCATE_CHARS = 300
_ANSWER_TRUNCATION_SUFFIX = "... [truncated]"


def _format_history(turns: list[dict]) -> str:
    """
    Format turn history as a readable string for the resolver prompt.

    Answers are truncated at _ANSWER_TRUNCATE_CHARS to prevent prompt bloat
    on sessions with long generated answers. The company list from each turn
    is appended to the question line so the resolver can identify which company
    was discussed even if the question phrasing didn't name it explicitly.
    """
    if not turns:
        return "(no prior conversation)"

    lines = []
    for t in turns:
        n = t["turn_number"]
        q = t["raw_question"]
        companies = t.get("companies") or []
        company_tag = f" [companies: {', '.join(companies)}]" if companies else ""
        lines.append(f"Q{n}: {q}{company_tag}")

        answer = t.get("final_answer") or ""
        if answer:
            if len(answer) > _ANSWER_TRUNCATE_CHARS:
                answer = answer[:_ANSWER_TRUNCATE_CHARS] + _ANSWER_TRUNCATION_SUFFIX
            lines.append(f"A{n}: {answer}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Prefixes the model sometimes prepends despite instructions — stripped defensively
_STRIP_PREFIXES = (
    "resolved question:",
    "rewritten question:",
    "standalone question:",
    "answer:",
    "result:",
)


def resolve_context(question: str, history: list[dict]) -> str:
    """
    Resolve a follow-up question into a self-contained standalone question.

    Args:
        question: The raw question the user typed (may contain pronouns, etc.)
        history:  Prior turns from get_history(), oldest-first. Each dict must
                  have: turn_number, raw_question, final_answer, companies.

    Returns:
        A self-contained question string. If no resolution was needed (the
        question is already standalone), the original question is returned
        unchanged (enforced by RULE 1 in the prompt).

    Fallback: if the LLM returns an empty string or a known bad output, the
    original question is returned unchanged so the pipeline always has
    something to work with.
    """
    if not history:
        # No history → nothing to resolve against → pass through unchanged
        return question

    llm = ChatNVIDIA(
        model=config.MODEL_REWRITE,
        api_key=config.NVIDIA_API_KEY,
        base_url=config.NVIDIA_BASE_URL,
        temperature=0.0,
    )
    chain = context_resolve_prompt | llm

    history_str = _format_history(history)
    response = chain.invoke({
        "history": history_str,
        "question": question,
    })

    resolved = response.content.strip()

    # Defensive: strip any prefix the model added despite instructions
    lower = resolved.lower()
    for prefix in _STRIP_PREFIXES:
        if lower.startswith(prefix):
            resolved = resolved[len(prefix):].strip()
            break

    # Final fallback: if resolution produced an empty string, return original
    return resolved if resolved else question
