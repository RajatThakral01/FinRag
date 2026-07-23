from graph.state import create_initial_state
from graph.nodes import (
    router_node, retrieve_node, grade_node, generate_node,
    calculator_node, hallucination_check_node,
)
from graph.graph import run_query


def run_node_chain(question, nodes):
    state = create_initial_state(question)
    for name, node_fn in nodes:
        update = node_fn(state)
        state.update(update)
        print(f"--- after {name} ---")
        for k, v in update.items():
            print(f"  {k}: {v}")
    return state


print("=" * 60)
print("TEST 1: generate_node (manual chain, retrieve-route question)")
print("=" * 60)
state1 = run_node_chain(
    "What was Apple's revenue in 2024?",
    [
        ("router", router_node),
        ("retrieve", retrieve_node),
        ("grade", grade_node),
        ("generate", generate_node),
    ],
)

print("\n" + "=" * 60)
print("TEST 2: calculator_node (manual chain, calculate-route question)")
print("=" * 60)
state2 = run_node_chain(
    "What was Apple's gross margin in 2024?",
    [
        ("router", router_node),
        ("retrieve", retrieve_node),
        ("grade", grade_node),
        ("calculator", calculator_node),
    ],
)

print("\n" + "=" * 60)
print("TEST 3a: hallucination_check_node — real answer, should say grounded")
print("=" * 60)
result_grounded = hallucination_check_node(state1)
print(result_grounded)

print("\n" + "=" * 60)
print("TEST 3b: hallucination_check_node — fabricated answer, should catch it")
print("=" * 60)
# Same real chunks as state1, but a deliberately wrong/invented answer —
# PRD Phase 5's explicit test case: does it actually catch a bad answer,
# not just approve whatever it's handed
fake_state = dict(state1)
fake_state["answer"] = (
    "Apple's revenue in 2024 was $850 billion, driven primarily by a "
    "40% increase in iPhone sales in the metaverse division."
)
result_not_grounded = hallucination_check_node(fake_state)
print(result_not_grounded)

print("\n" + "=" * 60)
print("TEST 4: full compiled pipeline via run_query()")
print("=" * 60)

test_questions = [
    "What was Apple's revenue in 2024?",       # expect route=retrieve
    "What was Apple's gross margin in 2024?",  # expect route=calculate
    "What does EBITDA mean?",                  # expect route=direct
]

for q in test_questions:
    print(f"\n--- Question: {q} ---")
    final_state = run_query(q)
    print(f"route: {final_state.get('route')}")
    print(f"relevant: {final_state.get('relevant')}")
    print(f"grounded: {final_state.get('grounded')}")
    print(f"retry_count: {final_state.get('retry_count')}")
    print(f"error_message: {final_state.get('error_message')}")
    print(f"final_answer: {final_state.get('final_answer')}")
    
    
import config
from graph.edges import route_after_grade, route_after_hallucination, route_by_calc_type
from graph.nodes import grade_exhausted_warning_node, hallucination_exhausted_node

print("=" * 60)
print("TEST 5: grade_exhausted path")
print("=" * 60)

# Simulate: Grade has said 'no' 3 times already, retry budget is gone
state5 = create_initial_state("What is Netflix's fictional Mars colony revenue?")
state5["relevant"] = "no"
state5["retry_count"] = config.MAX_RETRY
state5["route"] = "retrieve"

edge_result = route_after_grade(state5)
print(f"route_after_grade result: {edge_result}  (expect 'exhausted')")

warning_update = grade_exhausted_warning_node(state5)
print(f"grade_exhausted_warning_node output: {warning_update}")
state5.update(warning_update)

next_node = route_by_calc_type(state5)
print(f"route_by_calc_type result: {next_node}  (expect 'generate', since route='retrieve')")

print("\n" + "=" * 60)
print("TEST 5b: same state, route='calculate' instead")
print("=" * 60)
state5b = dict(state5)
state5b["route"] = "calculate"
next_node_b = route_by_calc_type(state5b)
print(f"route_by_calc_type result: {next_node_b}  (expect 'calculate')")


print("\n" + "=" * 60)
print("TEST 6: hallucination_exhausted path")
print("=" * 60)

state6 = create_initial_state("What was Apple's revenue in 2024?")
state6["answer"] = "Apple's revenue was some made up number."
state6["grounded"] = "not_grounded"
state6["retry_count"] = config.MAX_RETRY

edge_result6 = route_after_hallucination(state6)
print(f"route_after_hallucination result: {edge_result6}  (expect 'exhausted')")

exhausted_update = hallucination_exhausted_node(state6)
print(f"hallucination_exhausted_node output: {exhausted_update}")

print("=" * 60)
print("TEST 7: Multi-company retrieve — 'Compare Apple and Microsoft revenue'")
print("=" * 60)
final_state7 = run_query("Compare Apple and Microsoft revenue in 2024")
print(f"route: {final_state7.get('route')}")
print(f"companies_mentioned: {final_state7.get('companies_mentioned')}")
print(f"relevant: {final_state7.get('relevant')}")
print(f"grounded: {final_state7.get('grounded')}")
print(f"retry_count: {final_state7.get('retry_count')}")
print(f"num chunks retrieved: {len(final_state7.get('retrieved_chunks', []))}")
companies7 = set(c['company'] for c in final_state7.get('chunk_sources', []))
print(f"companies actually present in chunk_sources: {companies7}")
print(f"final_answer: {final_state7.get('final_answer')}")

print("\n" + "=" * 60)
print("TEST 8: All-companies calculate — 'Which company had highest R&D spend?'")
print("=" * 60)
final_state8 = run_query("Which company had the highest R&D spend in 2024?")
print(f"route: {final_state8.get('route')}")
print(f"companies_mentioned: {final_state8.get('companies_mentioned')}")
print(f"relevant: {final_state8.get('relevant')}")
print(f"grounded: {final_state8.get('grounded')}")
print(f"retry_count: {final_state8.get('retry_count')}")
print(f"num chunks retrieved: {len(final_state8.get('retrieved_chunks', []))}")
companies8 = set(c['company'] for c in final_state8.get('chunk_sources', []))
print(f"companies actually present in chunk_sources: {companies8}")
print(f"final_answer: {final_state8.get('final_answer')}")

print("\n" + "=" * 60)
print("TEST 9: Multi-company calculate — 'Compare Tesla and NVIDIA gross margins'")
print("=" * 60)
final_state9 = run_query("Compare Tesla and NVIDIA gross margins in 2024")
print(f"route: {final_state9.get('route')}")
print(f"companies_mentioned: {final_state9.get('companies_mentioned')}")
print(f"relevant: {final_state9.get('relevant')}")
print(f"grounded: {final_state9.get('grounded')}")
print(f"retry_count: {final_state9.get('retry_count')}")
print(f"num chunks retrieved: {len(final_state9.get('retrieved_chunks', []))}")
companies9 = set(c['company'] for c in final_state9.get('chunk_sources', []))
print(f"companies actually present in chunk_sources: {companies9}")
print(f"final_answer: {final_state9.get('final_answer')}")