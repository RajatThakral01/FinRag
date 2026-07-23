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
)
from graph.edges import (
    route_after_router,
    route_after_grade,
    route_by_calc_type,
    route_after_hallucination,
)


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
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
        {"direct": "direct_answer", "retrieve": "retrieve"},
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


def run_query(question: str) -> GraphState:
    app = build_graph()
    initial_state = create_initial_state(question)
    return app.invoke(initial_state)