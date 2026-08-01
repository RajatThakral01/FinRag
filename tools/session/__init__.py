# tools/session/__init__.py
# Re-exports for backward compatibility — callers can use either:
#   from tools.session.session_store import create_session
#   from tools.session_store import create_session   (via tools/__init__.py shims)

from tools.session.session_store import (
    create_session,
    list_sessions,
    get_session,
    get_history,
    add_turn,
)
from tools.session.context_resolver import resolve_context, _format_history
from tools.session.retrieval_cache import get_cache, put_cache

__all__ = [
    "create_session",
    "list_sessions",
    "get_session",
    "get_history",
    "add_turn",
    "resolve_context",
    "_format_history",
    "get_cache",
    "put_cache",
]
