import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

print("--- SMOKE TEST ---")

# Wait for server
time.sleep(2)

# Create session
resp = requests.post(f"{BASE_URL}/sessions")
session_id = resp.json()["session_id"]
print(f"Created session: {session_id}")

def ask(query):
    print(f"\n[User]: {query}")
    payload = {"question": query}
    resp = requests.post(f"{BASE_URL}/sessions/{session_id}/query", json=payload)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return
    data = resp.json()
    print(json.dumps({
        "resolved_question": data.get("resolved_question"),
        "question_was_resolved": data.get("question_was_resolved"),
        "cache_hit": data.get("cache_hit"),
        "final_answer": data.get("final_answer")
    }, indent=2))

ask("What was Apple's total revenue in 2024?")
ask("What about its net income?")
ask("How does that compare to Microsoft's net income?")

