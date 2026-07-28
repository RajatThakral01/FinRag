import sqlite3
import json
import uuid
import struct
import numpy as np
import os
import hashlib
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

import config

def _get_conn():
    conn = sqlite3.connect(config.SESSION_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def _init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_cache (
                cache_id           TEXT PRIMARY KEY,
                route              TEXT NOT NULL,
                companies_json     TEXT NOT NULL,
                question_embedding BLOB NOT NULL,
                resolved_question  TEXT NOT NULL,
                chunks_json        TEXT NOT NULL,
                sources_json       TEXT NOT NULL,
                grade_result       TEXT NOT NULL,
                created_at         TEXT NOT NULL,
                hit_count          INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_route_companies ON retrieval_cache(route, companies_json)")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                corpus_hash TEXT NOT NULL
            )
        """)

# Call init on import
_init_db()

def _pack_vector(vector: List[float]) -> bytes:
    """Pack a list of floats into a binary blob for SQLite."""
    return struct.pack(f"{len(vector)}f", *vector)

def _unpack_vector(blob: bytes) -> np.ndarray:
    """Unpack a binary blob from SQLite back into a numpy array."""
    num_floats = len(blob) // 4
    floats = struct.unpack(f"{num_floats}f", blob)
    return np.array(floats, dtype=np.float32)

def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def get_cache(route: str, companies: List[str], metric_category: str, query_embedding: List[float]) -> Optional[Tuple[List[str], List[dict], str]]:
    """
    Look up a query in the semantic cache.
    Requires an EXACT match on route, companies_mentioned, and metric_category,
    and a cosine similarity > config.CACHE_SIMILARITY_THRESHOLD on the embedding.
    Returns (retrieved_chunks, chunk_sources, grade_result) if hit, else None.
    """
    if metric_category == "general":
        return None  # Bypass cache for general or multi-metric questions
        
    companies_json = json.dumps(sorted(companies))
    q_vec = np.array(query_embedding, dtype=np.float32)
    
    with _get_conn() as conn:
        # Fetch all candidates with the exact route, companies, and metric category
        # Only consider entries where grade_result was "yes" (good chunks)
        cursor = conn.execute(
            """SELECT cache_id, question_embedding, chunks_json, sources_json, grade_result 
               FROM retrieval_cache 
               WHERE route = ? AND companies_json = ? AND metric_category = ? AND grade_result = 'yes'""",
            (route, companies_json, metric_category)
        )
        
        best_cache_id = None
        best_sim = -1.0
        best_chunks = None
        best_sources = None
        best_grade = None
        
        for row in cursor.fetchall():
            cache_id, emb_blob, chunks_json, sources_json, grade_result = row
            cached_vec = _unpack_vector(emb_blob)
            
            sim = _cosine_similarity(q_vec, cached_vec)
            if sim > best_sim and sim >= config.CACHE_SIMILARITY_THRESHOLD:
                best_sim = sim
                best_cache_id = cache_id
                best_chunks = json.loads(chunks_json)
                best_sources = json.loads(sources_json)
                best_grade = grade_result
                
        if best_cache_id:
            # Update hit count
            conn.execute("UPDATE retrieval_cache SET hit_count = hit_count + 1 WHERE cache_id = ?", (best_cache_id,))
            return best_chunks, best_sources, best_grade
            
    return None

def put_cache(route: str, companies: List[str], metric_category: str, query_embedding: List[float], 
              resolved_question: str, retrieved_chunks: List[str], chunk_sources: List[dict], 
              grade_result: str):
    """
    Save a retrieval result to the cache.
    Only called if retrieval was performed (not on cache hit) and grade_result was "yes".
    """
    if grade_result != "yes" or metric_category == "general":
        return  # Do not cache failed retrievals or general/multi-metric questions
        
    cache_id = str(uuid.uuid4())
    companies_json = json.dumps(sorted(companies))
    emb_blob = _pack_vector(query_embedding)
    now = datetime.now(timezone.utc).isoformat()
    
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO retrieval_cache 
               (cache_id, route, companies_json, metric_category, question_embedding, resolved_question, 
                chunks_json, sources_json, grade_result, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cache_id, route, companies_json, metric_category, emb_blob, resolved_question,
             json.dumps(retrieved_chunks), json.dumps(chunk_sources), grade_result, now)
        )

def clear_cache():
    """Clear all entries from the cache (e.g. when BM25 index or configs change)."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM retrieval_cache")

def _compute_corpus_hash() -> str:
    hash_obj = hashlib.sha256()
    
    # 1. Config models and settings
    configs = [
        config.MODEL_ROUTER,
        config.MODEL_GRADER,
        config.MODEL_GENERATOR,
        config.MODEL_CALCULATOR,
        config.MODEL_HALLUC,
        config.MODEL_REWRITE,
        str(config.TOP_K)
    ]
    hash_obj.update("|".join(configs).encode("utf-8"))
    
    # 2. ChromaDB mtime
    chroma_file = os.path.join(config.CHROMA_PATH, "chroma.sqlite3")
    if os.path.exists(chroma_file):
        hash_obj.update(str(os.path.getmtime(chroma_file)).encode("utf-8"))
        
    # 3. BM25 mtime
    if os.path.exists(config.BM25_INDEX_PATH):
        hash_obj.update(str(os.path.getmtime(config.BM25_INDEX_PATH)).encode("utf-8"))
        
    return hash_obj.hexdigest()

def _check_and_invalidate_cache():
    current_hash = _compute_corpus_hash()
    with _get_conn() as conn:
        row = conn.execute("SELECT corpus_hash FROM cache_metadata WHERE id = 1").fetchone()
        if row is None:
            # First run, just store it
            conn.execute("INSERT INTO cache_metadata (id, corpus_hash) VALUES (1, ?)", (current_hash,))
        elif row[0] != current_hash:
            # Hash changed, clear cache and update
            print(f"Corpus/Config changed! Clearing cache. (old: {row[0]}, new: {current_hash})")
            conn.execute("DELETE FROM retrieval_cache")
            conn.execute("UPDATE cache_metadata SET corpus_hash = ? WHERE id = 1", (current_hash,))

# Call on import
_check_and_invalidate_cache()
