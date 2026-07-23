import config
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA( model = config.MODEL_ROUTER  )

print(llm.invoke("what is python").content)