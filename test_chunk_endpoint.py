import requests
import json

BASE_URL = "http://127.0.0.1:8000"

resp = requests.get(f"{BASE_URL}/chunks/aapl_2024_item8_table_038_000")
print("STATUS CODE:", resp.status_code)
print("RESPONSE JSON:")
print(json.dumps(resp.json(), indent=2))
