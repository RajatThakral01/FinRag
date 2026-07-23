import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.vectorstore import get_vectorstore

vs = get_vectorstore()
results = vs.get(where={"ticker": "INTC"}, limit=3038)

item_numbers = [m.get("item_number") for m in results["metadatas"]]
from collections import Counter
print(Counter(item_numbers))