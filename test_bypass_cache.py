import uuid
import concurrent.futures
from graph.graph import run_session_query
from tools.session_store import create_session

bypass_cases = [
    "Compare Apple's revenue and net income",
    "What are the main risks for Meta?"
]

def run_query_with_timeout(session_id, q, timeout=120):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_session_query, session_id, q)
        return future.result(timeout=timeout)[0]

for q in bypass_cases:
    session_id = create_session()
    print(f"\n--- Testing Bypass/Other (Session: {session_id}) ---")
    
    print(f"[1/2] Asking (First Time): '{q}'...")
    state = run_query_with_timeout(session_id, q)
    print(f"      -> Category: {state.get('metric_category')} | Cache Hit: {state.get('cache_hit')}")
    
    print(f"[2/2] Asking (Second Time): '{q}'...")
    state2 = run_query_with_timeout(session_id, q)
    print(f"      -> Category: {state2.get('metric_category')} | Cache Hit: {state2.get('cache_hit')}")
