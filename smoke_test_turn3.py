import requests
import json

BASE_URL = "http://127.0.0.1:8000"
session_id = "337c450b-1489-4088-91bd-e40086029692"

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

print("--- SMOKE TEST TURN 3 ---")
ask("How does that compare to Microsoft's net income?")

