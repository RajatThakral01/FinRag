import json
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from graph.nodes import query_analysis_prompt
from tools.output_parsers import parse_query_analysis
import config

llm = ChatNVIDIA(model=config.MODEL_ROUTER, temperature=0.0)
chain = query_analysis_prompt | llm

cases = [
    "What was Apple's revenue?",
    "Compare Apple and Microsoft's margins",
    "Which of Apple, Google, and Amazon had the highest R&D?",
    "Which company had the highest revenue?",
    "What does EBITDA mean?",
    "What is Tesla's tax rate?",
    "What was Alphabet's net income?"
]

for c in cases:
    resp = chain.invoke({"question": c})
    res = parse_query_analysis(resp.content)
    print(f"Q: {c}\nA: {res}\n")
