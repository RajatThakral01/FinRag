from langgraph.graph import StateGraph, END

from graph.state import GraphState, create_initial_state
from graph.nodes import (
    router_node,
    retrieve_node,
    grade_node,
    rewrite_node,
    generate_node,
    calculator_node,
    direct_answer_node,
    hallucination_check_node,
    grade_exhausted_warning_node,
    hallucination_exhausted_node,
    cache_lookup_node,
)
from graph.edges import (
    route_after_router,
    route_after_grade,
    route_by_calc_type,
    route_after_hallucination,
    route_after_cache,
)


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("cache_lookup", cache_lookup_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("hallucination_check", hallucination_check_node)
    graph.add_node("grade_exhausted_warning", grade_exhausted_warning_node)
    graph.add_node("hallucination_exhausted", hallucination_exhausted_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"direct": "direct_answer", "cache_lookup": "cache_lookup"},
    )

    graph.add_conditional_edges(
        "cache_lookup",
        route_after_cache,
        {"retrieve": "retrieve", "calculate": "calculator", "generate": "generate"},
    )

    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "calculate": "calculator",
            "rewrite": "rewrite",
            "exhausted": "grade_exhausted_warning",
        },
    )

    graph.add_conditional_edges(
        "grade_exhausted_warning",
        route_by_calc_type,
        {"generate": "generate", "calculate": "calculator"},
    )

    graph.add_edge("rewrite", "retrieve")

    graph.add_edge("generate", "hallucination_check")
    graph.add_edge("calculator", "hallucination_check")

    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination,
        {
            "end": END,
            "generate": "generate",
            "calculate": "calculator",
            "exhausted": "hallucination_exhausted",
        },
    )

    graph.add_edge("hallucination_exhausted", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


def run_query(question: str, conversation_context: str = None) -> GraphState:
    app = build_graph()
    initial_state = create_initial_state(question)
    initial_state["conversation_context"] = conversation_context
    return app.invoke(initial_state)


def run_session_query(session_id: str, raw_question: str) -> tuple[GraphState, str]:
    """
    Run a query within a named session, with conversation-context resolution.

    Flow:
      1. Load last CONTEXT_WINDOW turns from SQLite session store
      2. Resolve raw_question → resolved_question (handles pronouns, implicit
         company references, topic shifts)
      3. Run the existing graph on resolved_question (graph unchanged)
      4. Persist the turn (raw + resolved + graph outputs) to session store
      5. Return (final_state, resolved_question)

    The caller should compare raw_question != resolved_question to decide
    whether to show the resolved version in the UI (Option B: only show
    when they differ, so the user can catch misresolutions).

    The resolved_question is EXACTLY what was passed to create_initial_state()
    and entered the graph — not a summarized or re-processed version.
    """
    import config
    from tools.session_store import get_history, add_turn
    from tools.context_resolver import resolve_context, _format_history

    history = get_history(session_id, last_n=config.CONTEXT_WINDOW)
    resolved_question = resolve_context(raw_question, history)
    
    conversation_context = _format_history(history) if history else None

    final_state = run_query(resolved_question, conversation_context=conversation_context)

    add_turn(
        session_id=session_id,
        raw_question=raw_question,
        resolved_question=resolved_question,
        route=final_state.get("route", ""),
        companies=final_state.get("companies_mentioned", []),
        final_answer=final_state.get("final_answer", ""),
        error_message=final_state.get("error_message"),
    )

    return final_state, resolved_question