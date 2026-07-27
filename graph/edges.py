import config


def route_after_router(state) -> str:
    return "direct" if state["route"] == "direct" else "cache_lookup"

def route_after_cache(state) -> str:
    if state.get("cache_hit"):
        return "calculate" if state["route"] == "calculate" else "generate"
    return "retrieve"


def route_after_grade(state) -> str:
    if state["relevant"] == "yes":
        return "calculate" if state["route"] == "calculate" else "generate"
    if state["retry_count"] < config.MAX_RETRY:
        return "rewrite"
    return "exhausted"


def route_by_calc_type(state) -> str:
    """Used after grade_exhausted_warning_node to pick the right answer node."""
    return "calculate" if state["route"] == "calculate" else "generate"


def route_after_hallucination(state) -> str:
    if state["grounded"] == "grounded":
        return "end"
    if state["retry_count"] < config.MAX_RETRY:
        return "calculate" if state["route"] == "calculate" else "generate"
    return "exhausted"