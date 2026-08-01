"""
tools/session_store.py
----------------------
SQLite-backed session store for multi-turn conversation history.

Design decisions:
  - WAL mode: allows concurrent readers + one writer without blocking
  - foreign_keys ON: cascading integrity between sessions and turns
  - Auto-title from first raw_question (first 50 chars, per Q3 decision)
  - get_history() returns oldest-first — natural conversation order for prompts
  - raw_question AND resolved_question both stored for debugging misresolutions
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone

import config


# ---------------------------------------------------------------------------
# Connection + schema init
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SESSION_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL,
                title       TEXT,
                last_active TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS turns (
                turn_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id        TEXT NOT NULL REFERENCES sessions(session_id),
                turn_number       INTEGER NOT NULL,
                raw_question      TEXT NOT NULL,
                resolved_question TEXT NOT NULL,
                route             TEXT,
                companies_json    TEXT,
                final_answer      TEXT,
                error_message     TEXT,
                created_at        TEXT NOT NULL,

                UNIQUE(session_id, turn_number)
            );
        """)


_init_db()


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def create_session() -> str:
    """Create a new session and return its UUID."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, title, last_active) "
            "VALUES (?, ?, ?, ?)",
            (session_id, now, None, now),
        )
    return session_id


def get_session(session_id: str) -> dict | None:
    """Return session metadata dict, or None if not found."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def list_sessions() -> list[dict]:
    """Return all sessions ordered by most recently active first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY last_active DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> None:
    """Delete a session and all its turns."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


# ---------------------------------------------------------------------------
# Turn CRUD
# ---------------------------------------------------------------------------

def add_turn(
    session_id: str,
    raw_question: str,
    resolved_question: str,
    route: str = "",
    companies: list[str] | None = None,
    final_answer: str = "",
    error_message: str | None = None,
) -> int:
    """
    Append a turn to a session.

    On the first turn (turn_number == 1), the session title is set to the
    first 50 characters of raw_question (per Q3 decision: auto-title from
    raw_question, no summarization LLM call).

    Returns the auto-incremented turn_id.
    """
    now = datetime.now(timezone.utc).isoformat()
    companies_json = json.dumps(sorted(companies or []))

    with _get_conn() as conn:
        result = conn.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        turn_number = result[0] + 1

        if turn_number == 1:
            # Auto-title: first 50 chars of raw question, strip trailing whitespace
            title = raw_question[:50].rstrip()
            conn.execute(
                "UPDATE sessions SET title = ?, last_active = ? WHERE session_id = ?",
                (title, now, session_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET last_active = ? WHERE session_id = ?",
                (now, session_id),
            )

        conn.execute(
            """INSERT INTO turns
               (session_id, turn_number, raw_question, resolved_question,
                route, companies_json, final_answer, error_message, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                turn_number,
                raw_question,
                resolved_question,
                route,
                companies_json,
                final_answer,
                error_message,
                now,
            ),
        )
        turn_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    return turn_id


def get_history(session_id: str, last_n: int | None = None) -> list[dict]:
    """
    Return turns for a session in oldest-first order (natural for prompts).

    `last_n` caps to the N most recent turns. Defaults to config.CONTEXT_WINDOW.
    The returned dicts include a 'companies' list (parsed from companies_json).
    """
    if last_n is None:
        last_n = config.CONTEXT_WINDOW

    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM turns
               WHERE session_id = ?
               ORDER BY turn_number DESC
               LIMIT ?""",
            (session_id, last_n),
        ).fetchall()

    # Reverse so oldest turn is first
    turns = [dict(r) for r in reversed(rows)]
    for t in turns:
        t["companies"] = json.loads(t.get("companies_json") or "[]")
    return turns


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # Use a temp DB for the test so we don't pollute the real one
    config.SESSION_DB_PATH = "./test_session_store.db"
    _init_db()

    print("=== session_store smoke test ===\n")

    sid = create_session()
    print(f"Created session: {sid}")

    t1 = add_turn(
        sid,
        raw_question="What was Apple's revenue in 2024?",
        resolved_question="What was Apple's revenue in 2024?",
        route="retrieve",
        companies=["Apple"],
        final_answer="Apple's total revenue in fiscal year 2024 was $391.0 billion.",
    )
    print(f"Added turn 1 (id={t1})")

    t2 = add_turn(
        sid,
        raw_question="What about their R&D expense?",
        resolved_question="What was Apple's R&D expense in 2024?",
        route="retrieve",
        companies=["Apple"],
        final_answer="Apple's R&D expense in fiscal year 2024 was $31.4 billion.",
    )
    print(f"Added turn 2 (id={t2})")

    session = get_session(sid)
    print(f"\nSession title: {session['title']!r}")

    history = get_history(sid)
    print(f"\nHistory ({len(history)} turns):")
    for t in history:
        print(f"  T{t['turn_number']}: raw={t['raw_question']!r}")
        print(f"         resolved={t['resolved_question']!r}")

    all_sessions = list_sessions()
    print(f"\nTotal sessions: {len(all_sessions)}")

    # Cleanup
    delete_session(sid)
    os.remove("./test_session_store.db")
    print("\nSmoke test passed — cleanup done.")
