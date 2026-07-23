from langchain_huggingface import HuggingFaceEmbeddings

print("Loading embedding model (this may download ~420MB on first run)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

test_text = "Apple's total revenue in fiscal year 2024."
vector = embeddings.embed_query(test_text)

print(f"\nEmbedded text: {test_text!r}")
print(f"Vector length: {len(vector)}")
print(f"First 5 values: {vector[:5]}")