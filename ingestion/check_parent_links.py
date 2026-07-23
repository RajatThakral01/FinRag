from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    collection_name="financial_10k",
    embedding_function=embeddings,
)

results = vectorstore.similarity_search("What was Apple's total revenue in fiscal year 2024?", k=3)

print("=== Parent-child link check ===")
for doc in results:
    parent_id = doc.metadata.get("parent_chunk_id", "")
    chunk_id = doc.metadata.get("chunk_id", "")
    print(f"\nchunk_id={chunk_id} parent_chunk_id={parent_id!r}")

    if parent_id:
        parent_result = vectorstore.get(ids=[parent_id])
        if parent_result["ids"]:
            parent_text = parent_result["documents"][0]
            parent_meta = parent_result["metadatas"][0]
            print(f"  -> Found parent. type={parent_meta['chunk_type']} text preview: {parent_text[:150]}")
        else:
            print(f"  -> PARENT NOT FOUND for id {parent_id}")
    else:
        print("  -> No parent set for this chunk")