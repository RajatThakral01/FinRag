import json , re

def parse_hallucination(llm_output: str) -> str:
    """
    Checks only the last line (same defensive pattern as parse_grade).
    Checks 'not_grounded' before 'grounded' — 'grounded' is a substring
    of 'not_grounded', so checking in the wrong order would misclassify
    every hallucinated answer as grounded.
    Defaults to 'not_grounded': safer to treat an unparseable response
    as unverified than to risk certifying a hallucinated answer as final.
    """
    lines = llm_output.strip().splitlines()
    last_line = lines[-1].strip().lower() if lines else ""
    if "not_grounded" in last_line:
        return "not_grounded"
    elif "grounded" in last_line:
        return "grounded"
    return "not_grounded"

def parse_grade(llm_output: str) -> str:
    lines = llm_output.strip().splitlines()
    last_line = lines[-1].strip().lower() if lines else ""
    if "yes" in last_line:
        return "yes"
    return "no"

def parse_query_analysis(llm_output: str) -> dict:
    text = llm_output.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"companies": ["all"], "metric_category": "general"}

    if not isinstance(parsed, dict) or "companies" not in parsed or "metric_category" not in parsed:
        return {"companies": ["all"], "metric_category": "general"}

    return parsed

def parse_route(llm_output: str) -> str:
    text = llm_output.strip().lower()
    if "calculate" in text or "math" in text or "compute" in text:
        return "calculate"
    if "direct" in text or "general" in text or "definition" in text:
        return "direct"
    return "retrieve"  # default — safest fallback

def parse_calculation(llm_output: str) -> dict:
    """
    Defensive JSON parsing, same pattern as parse_companies (PRD 11.4) —
    models sometimes wrap JSON in code fences or add commentary despite
    instructions not to. Falls back to insufficient_data: safe because it
    routes to the "can't calculate" message rather than crashing or
    silently computing on garbage.
    """
    try:
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if "operation" in parsed and "values" in parsed:
                return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {"operation": "insufficient_data", "values": []}