from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
GROQ_MAX_RETRIES   = 3

EMBEDDING_MODEL    = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE         = 450   # tokens
CHUNK_OVERLAP      = 50    # tokens
TOP_K              = 5     # chunks to retrieve
MAX_RETRY          = 3     # max rewrite retries
CHROMA_PATH        = "./chroma_db"
COLLECTION_NAME    = "financial_10k"
BM25_INDEX_PATH    = "./bm25_index.pkl"

# Session memory & retrieval cache (Phase A/B)
SESSION_DB_PATH           = "./session_data.db"  # SQLite file for sessions + turns + cache
CONTEXT_WINDOW            = 5                     # turns of history fed to context resolver
CACHE_SIMILARITY_THRESHOLD = 0.88                 # cosine sim threshold for retrieval cache hits

# Model per node
MODEL_ROUTER       = "llama-3.1-8b-instant"
MODEL_GRADER       = "llama-3.1-8b-instant"
MODEL_GENERATOR    = "llama-3.3-70b-versatile"
MODEL_HALLUC       = "llama-3.1-8b-instant"
MODEL_REWRITE      = "llama-3.1-8b-instant"
MODEL_CALCULATOR   = "llama-3.1-8b-instant"