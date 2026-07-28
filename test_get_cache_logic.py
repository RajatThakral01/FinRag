import sqlite3, json, numpy as np
conn = sqlite3.connect("session_data.db")
c = conn.cursor()
c.execute("SELECT companies_json, metric_category, question_embedding FROM retrieval_cache WHERE metric_category = 'revenue_sales' LIMIT 1")
row = c.fetchone()
print(repr(row[0]), repr(row[1]))
emb = json.loads(row[2])
query_emb = [0.1]*1536
print("Cosine sim:", np.dot(emb, query_emb) / (np.linalg.norm(emb) * np.linalg.norm(query_emb)))
