import requests
import json

BASE_URL = "http://127.0.0.1:8000"

resp = requests.post(f"{BASE_URL}/sessions")
session_id = resp.json()["session_id"]

query_resp = requests.post(f"{BASE_URL}/sessions/{session_id}/query", json={"question": "What was Apple's total revenue in 2024?"})
data = query_resp.json()

print("RAW RESPONSE CHUNK_SOURCES:")
print(json.dumps(data.get("chunk_sources"), indent=2))
