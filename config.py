from dotenv import load_dotenv
import os
load_dotenv()
    
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
    
EMBEDDING_MODEL    = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE         = 450   # tokens
CHUNK_OVERLAP      = 50    # tokens
TOP_K              = 4     # chunks to retrieve
MAX_RETRY          = 3     # max rewrite retries
CHROMA_PATH        = "./chroma_db"
COLLECTION_NAME    = "financial_10k"
BM25_INDEX_PATH    = "./bm25_index.pkl"

# Session memory & retrieval cache (Phase A/B)
SESSION_DB_PATH           = "./session_data.db"  # SQLite file for sessions + turns + cache
CONTEXT_WINDOW            = 5                     # turns of history fed to context resolver
CACHE_SIMILARITY_THRESHOLD = 0.88                 # cosine sim threshold for retrieval cache hits

# Model per node (see Chapter 13)
MODEL_ROUTER       = "llama-3.1-8b-instant"   # fast, cheap
MODEL_GRADER       = "llama-3.3-70b-versatile"   # fast, cheap
MODEL_GENERATOR    = "llama-3.3-70b-versatile"  # powerful
MODEL_CALCULATOR   = "llama-3.3-70b-versatile"  # requires high reasoning for extraction
MODEL_HALLUC       = "llama-3.1-8b-instant"   # fast, cheap
MODEL_REWRITE      = "llama-3.1-8b-instant"   # fast, cheap

# API Error Handling
GROQ_MAX_RETRIES   = 5                    # Retry with exponential backoff on 429 Too Many Requests