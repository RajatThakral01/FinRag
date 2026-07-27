from typing import TypedDict, List, Optional


class GraphState(TypedDict):
    question: str
    rewritten_question: str
    route: str
    companies_mentioned: List[str]
    retrieved_chunks: List[str]
    chunk_sources: List[dict]
    relevant: str
    answer: str
    grounded: str
    retry_count: int
    final_answer: str
    error_message: Optional[str]
    cache_hit: bool
    conversation_context: Optional[str]


def create_initial_state(question: str) -> GraphState:
    return {
        "question": question,
        "rewritten_question": "",
        "route": "",
        "companies_mentioned": [],
        "retrieved_chunks": [],
        "chunk_sources": [],
        "relevant": "",
        "answer": "",
        "grounded": "",
        "retry_count": 0,
        "final_answer": "",
        "error_message": None,
        "cache_hit": False,
        "conversation_context": None,
    }


if __name__ == "__main__":
    state = create_initial_state("What was Apple's total revenue in fiscal year 2024?")
    for key, value in state.items():
        print(f"{key}: {value!r}")