import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

print("--- CACHE TEST ---")

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
        "cache_hit": data.get("cache_hit")
    }, indent=2))

ask("What is Apple's total revenue?")
ask("What is Apple's net income?")
ask("Can you tell me Apple's total revenue?")

