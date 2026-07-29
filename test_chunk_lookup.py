from tools.vectorstore import get_vectorstore

vs = get_vectorstore()
col = vs._collection
res = col.get(where={"chunk_id": "aapl_2024_item8_table_038_000"})

print("LOOKUP BY CHUNK_ID RESULT:")
print("Found IDs:", res["ids"])
print("Found Metadatas:", res["metadatas"])
print("Text snippet (first 100 chars):", res["documents"][0][:100] if res["documents"] else "None")
