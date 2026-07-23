from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from metadata_tagger import tag_all_chunks
from metadata_tagger import tag_all_companies


def sanitize_metadata(chunk: dict) -> dict:
    metadata_fields = [
        "chunk_id", "company", "ticker", "year", "item_number", "section_name",
        "chunk_type", "table_name", "page_start", "block_idx", "parent_chunk_id"
    ]
    clean = {}
    for field in metadata_fields:
        value = chunk.get(field)
        clean[field] = value if value is not None else ""
    return clean
    
def filter_boilerplate(chunks: list[dict]) -> list[dict]:
    return [c for c in chunks if c.get("item_number")]

def embed_and_store(chunks: list[dict], persist_directory: str, collection_name: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    texts = [c["text"] for c in chunks]
    metadatas = [sanitize_metadata(c) for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    return vectorstore

if __name__ == "__main__":
    print("Tagging all 10 companies...")
    all_results = tag_all_companies("extracted_text")

    all_chunks = []
    for ticker, chunks in all_results.items():
        with_item = sum(1 for c in chunks if c.get("item_number"))
        print(f"  {ticker}: {len(chunks)} chunks, {with_item} with item_number ({len(chunks) - with_item} without)")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all companies: {len(all_chunks)}")
    print("Embedding and storing (this will take a while for ~3500+ chunks)...")

    before_count = len(all_chunks)
    all_chunks = filter_boilerplate(all_chunks)
    after_count = len(all_chunks)
    print(f"Filtered out {before_count - after_count} boilerplate chunks (no item_number). Remaining: {after_count}")

    vectorstore = embed_and_store(
        chunks=all_chunks,
        persist_directory="./chroma_db",
        collection_name="financial_10k"
    )

    print("\nDone. Running cross-company test queries...")

    test_queries = [
        ("What was Apple's total revenue in fiscal year 2024?", "AAPL"),
        ("What was NVIDIA's revenue growth?", "NVDA"),
        ("What are Tesla's main risk factors?", "TSLA"),
        ("What was Intel's revenue in fiscal year 2024?", "INTC"),
    ]

    for query, expected_ticker in test_queries:
        results = vectorstore.similarity_search(query, k=3)
        print(f"\nQuery: {query!r} (expecting mostly {expected_ticker})")
        for doc in results:
            print(f"  ticker={doc.metadata.get('ticker')} chunk_id={doc.metadata.get('chunk_id')} section={doc.metadata.get('section_name')}")