import os
from dotenv import load_dotenv
load_dotenv()
from langchain_nvidia_ai_endpoints import ChatNVIDIA
client = ChatNVIDIA()
models = client.available_models
print("Available Models:")
for m in models:
    print(f"- {m.id}")
