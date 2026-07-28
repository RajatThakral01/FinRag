import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from graph.nodes import query_analysis_prompt
from tools.output_parsers import parse_query_analysis

llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", temperature=0)

q = "What was Microsoft's net income?"
print(f"Testing: {q}")
for i in range(4):
    res = llm.invoke(query_analysis_prompt.format(question=q))
    parsed = parse_query_analysis(res.content)
    print(f"Run {i+1}: {parsed}")
