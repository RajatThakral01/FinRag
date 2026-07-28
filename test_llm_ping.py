import os
from dotenv import load_dotenv
load_dotenv()
from langchain_nvidia_ai_endpoints import ChatNVIDIA

try:
    llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
    res = llm.invoke("Say hello")
    print("llama-3.3-70b-instruct:", res.content)
except Exception as e:
    print("3.3 70b failed:", e)

try:
    llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct")
    res = llm.invoke("Say hello")
    print("llama-3.2-3b-instruct:", res.content)
except Exception as e:
    print("3.2 3b failed:", e)
