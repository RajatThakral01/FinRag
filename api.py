from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from tools.session.session_store import (
    create_session, list_sessions, get_session,
    update_session_title, get_history,
)
from tools.retrieval.vectorstore import get_vectorstore
import config
from graph.graph import run_session_query

app = FastAPI(
    title="Financial RAG API",
    description="REST API for the Financial RAG pipeline",
    version="1.0.0"
)

# CORS configuration
# Pin to specific origin as per requirements (needs tightening before prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SessionCreateRequest(BaseModel):
    title: Optional[str] = None

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health_check():
    """Simple 200 OK endpoint to detect backend availability."""
    return {"status": "ok"}

@app.post("/sessions")
def create_new_session(request: Optional[SessionCreateRequest] = None):
    """Create a new session and return its ID."""
    session_id = create_session()
    # If the request contains a title, update via session_store (uses WAL connection)
    if request and request.title:
        update_session_title(session_id, request.title)

    session_data = get_session(session_id)
    return {
        "session_id": session_id,
        "created_at": session_data["created_at"]
    }

@app.get("/sessions")
def get_all_sessions():
    """List all sessions ordered by most recently active first."""
    return list_sessions()

@app.get("/sessions/{session_id}/turns")
def get_session_turns(session_id: str):
    """List all turns for a specific session, ordered oldest-first."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    # Use session_store.get_history() — WAL-mode connection, companies_json parsed automatically
    return get_history(session_id, last_n=None)

@app.post("/sessions/{session_id}/query")
def submit_query(session_id: str, request: QueryRequest):
    """
    Submit a query to a session. The graph uses session history to resolve the question context.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    try:
        final_state, resolved_question = run_session_query(session_id, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # The boolean flag indicating whether the resolved question differs from the raw one
    question_was_resolved = request.question.strip() != resolved_question.strip()
    
    return {
        "raw_question": request.question,
        "resolved_question": resolved_question,
        "question_was_resolved": question_was_resolved,
        "final_answer": final_state.get("final_answer", ""),
        "cache_hit": final_state.get("cache_hit", False),
        "chunk_sources": final_state.get("chunk_sources", []),
        "error_message": final_state.get("error_message")
    }

@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str):
    """
    Fetch a single chunk by its ID from the ChromaDB vector store.
    Used by the frontend ChunkPreviewPanel to render the full text
    and metadata of a source citation when the user clicks on it.
    """
    vs = get_vectorstore()
    result = vs.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not result["documents"]:
        raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found.")
    return {
        "chunk_id": chunk_id,
        "text": result["documents"][0],
        "metadata": result["metadatas"][0] if result["metadatas"] else {},
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
