import uuid
import time
import concurrent.futures
from graph.graph import run_session_query
from tools.session_store import create_session
from tools.retrieval_cache import clear_cache
import config


pairs = [
    ("What was Microsoft's revenue in 2024?", "What was Microsoft's net income in 2024?"),
    ("What was Alphabet's operating income in 2024?", "What was Alphabet's net income in 2024?"),
    ("What was Apple's total revenue?", "What was Apple's net income?"),
    ("What was Amazon's revenue in 2024?", "What was Amazon's net income in 2024?"),
]

bypass_cases = [
    "Compare Apple's revenue and net income",
    "What are the main risks for Meta?"
]

def run_query_with_timeout(session_id, q, timeout=120):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_session_query, session_id, q)
        state, _ = future.result(timeout=timeout)
        return state

for q1, q2 in pairs:
    session_id = create_session()
    print(f"\n--- Testing Pair (Session: {session_id}) ---")
    
    print(f"[1/2] Asking Q1: '{q1}'...")
    state1 = run_query_with_timeout(session_id, q1)
    print(f"      -> Category: {state1.get('metric_category')} | Cache Hit: {state1.get('cache_hit')}")
    
    print(f"[2/2] Asking Q2: '{q2}'...")
    state2 = run_query_with_timeout(session_id, q2)
    print(f"      -> Category: {state2.get('metric_category')} | Cache Hit: {state2.get('cache_hit')}")

for q in bypass_cases:
    session_id = create_session()
    print(f"\n--- Testing Bypass/Other (Session: {session_id}) ---")
    
    print(f"[1/2] Asking (First Time): '{q}'...")
    state = run_query_with_timeout(session_id, q)
    print(f"      -> Category: {state.get('metric_category')} | Cache Hit: {state.get('cache_hit')}")
    
    print(f"[2/2] Asking (Second Time): '{q}'...")
    state2 = run_query_with_timeout(session_id, q)
    print(f"      -> Category: {state2.get('metric_category')} | Cache Hit: {state2.get('cache_hit')}")
