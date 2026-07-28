from langchain_groq import ChatGroq
import config

try:
    llm = ChatGroq(model=config.MODEL_ROUTER, groq_api_key=config.GROQ_API_KEY)
    res = llm.invoke("Say hello")
    print(res.content)
except Exception as e:
    print("FAILED:", e)
