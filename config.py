from dotenv import load_dotenv
import os
load_dotenv()
    
NVIDIA_API_KEY     = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL    = os.getenv("NVIDIA_BASE_URL")
    
EMBEDDING_MODEL    = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE         = 450   # tokens
CHUNK_OVERLAP      = 50    # tokens
TOP_K              = 5     # chunks to retrieve
MAX_RETRY          = 3     # max rewrite retries
CHROMA_PATH        = "./chroma_db"
COLLECTION_NAME    = "financial_10k"
BM25_INDEX_PATH    = "./bm25_index.pkl"
    
# Model per node (see Chapter 13)
MODEL_ROUTER       = "meta/llama-3.1-8b-instruct"   # fast, cheap
MODEL_GRADER       = "meta/llama-3.1-70b-instruct"   # fast, cheap
MODEL_GENERATOR    = "meta/llama-3.1-70b-instruct"  # powerful
MODEL_HALLUC       = "meta/llama-3.1-8b-instruct"   # fast, cheap
MODEL_REWRITE      = "meta/llama-3.1-8b-instruct"   # fast, cheap