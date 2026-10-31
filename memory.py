"""Persistent conversation memory in PostgreSQL.

Two layers share one connection pool:

- LangGraph checkpoints (`PostgresSaver`): the agent's state per thread_id — messages
  (possibly summarised to fit the model context), virtual files, pending HITL interrupts.
- Application tables (this module): conversation metadata, the full user-visible
  transcript, run records (one active run per thread), feedback and retrieved evidence.
  This is the source of truth for the UI; it is never trimmed or summarised.

Schema changes run only from `python manage.py init-db` (never on a UI rerun).
"""

import atexit
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

import config

ACTIVE = ("running", "awaiting_review")
# Staleness is judged by the database clock, so app/database clock skew (Docker, Neon) cannot
# make a live run look crashed or a crashed run look alive.
RUNS = ("*, (status = 'running' AND updated_at < now() - make_interval(secs => "
        f"{config.RUN_STALE_SECONDS})) AS is_stale")

# Application schema migrations, applied in order and recorded in app_schema_migrations.
# Each migration is one SQL script; statements are separated by ";" at line end.
MIGRATIONS = [
    """
    CREATE TABLE conversations (
        thread_id   uuid PRIMARY KEY,
        owner_id    text NOT NULL,
        title       text NOT NULL,
        settings    jsonb NOT NULL DEFAULT '{}',
        created_at  timestamptz NOT NULL DEFAULT now(),
        updated_at  timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX conversations_owner_updated ON conversations (owner_id, updated_at DESC);

    CREATE TABLE conversation_runs (
        run_id        uuid PRIMARY KEY,
        thread_id     uuid NOT NULL REFERENCES conversations ON DELETE CASCADE,
        submission_id text NOT NULL,
        status        text NOT NULL CHECK (status IN
                      ('running', 'awaiting_review', 'completed', 'failed', 'cancelled', 'rejected')),
        trace_id      text,
        settings      jsonb NOT NULL DEFAULT '{}',
        report_before text,
        attempts      integer NOT NULL DEFAULT 0,
        error         text,
        log           jsonb NOT NULL DEFAULT '[]',
        started_at    timestamptz NOT NULL DEFAULT now(),
        updated_at    timestamptz NOT NULL DEFAULT now(),
        UNIQUE (thread_id, submission_id)
    );
    -- at most one in-flight run per conversation (blocks conflicting runs)
    CREATE UNIQUE INDEX conversation_runs_one_active ON conversation_runs (thread_id)
        WHERE status IN ('running', 'awaiting_review');

    CREATE TABLE conversation_messages (
        message_id  uuid PRIMARY KEY,
        thread_id   uuid NOT NULL REFERENCES conversations ON DELETE CASCADE,
        run_id      uuid REFERENCES conversation_runs ON DELETE CASCADE,
        seq         bigserial,
        role        text NOT NULL CHECK (role IN ('user', 'assistant')),
        kind        text NOT NULL,
        content     text NOT NULL,
        guardrail   jsonb NOT NULL DEFAULT '{}',
        evaluation  jsonb NOT NULL DEFAULT '{}',
        trace_id    text,
        feedback    smallint CHECK (feedback IN (-1, 1)),
        created_at  timestamptz NOT NULL DEFAULT now(),
        UNIQUE (run_id, role)  -- one user and one assistant message per run: no duplicates on rerun
    );
    CREATE INDEX conversation_messages_thread ON conversation_messages (thread_id, seq);

    CREATE TABLE run_evidence (
        run_id         uuid NOT NULL REFERENCES conversation_runs ON DELETE CASCADE,
        url            text NOT NULL,
        title          text,
        content        text,
        published_date text,
        created_at     timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (run_id, url)
    );
    """,
]


class DatabaseError(RuntimeError):
    """The database is unreachable or not initialised; the message says what to do."""


class RunConflict(RuntimeError):
    """Another run is already in progress for this conversation."""


def clean(value):
    """Drop NUL characters, which PostgreSQL rejects in text and jsonb (web pages can contain them)."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def _describe(exc: Exception) -> str:
    """A connection error's message without any credentials in it."""
    from redaction import redact_text

    return redact_text(" ".join(str(exc).split()))[:300] or type(exc).__name__


# ---------------------------------------------------------------- connections
def database_url() -> str:
    url = config.require("DATABASE_URL", purpose="Persistent conversation memory")["DATABASE_URL"]
    # e.g. docker-compose.local-db.yml builds the URL from a POSTGRES_PASSWORD still left as placeholder
    if m := re.search(r"YOUR_([A-Z0-9_]+?)_HERE", url):
        name = "POSTGRES_PASSWORD" if m.group(1) == "LOCAL_POSTGRES_PASSWORD" else "DATABASE_URL"
        raise config.ConfigError([name], "Persistent conversation memory")
    if not url.startswith(("postgresql://", "postgres://")):
        raise config.ConfigError(["DATABASE_URL"], "Persistent conversation memory (expected postgresql://...)")
    return url


def connect_kwargs() -> dict:
    # autocommit + prepare_threshold=0 + dict_row are what PostgresSaver requires;
    # prepare_threshold=0 also keeps PgBouncer / Neon pooled endpoints working.
    return {"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row, "connect_timeout": 10}


def open_pool(url: str | None = None, *, max_size: int = 10, wait_seconds: float = 30) -> ConnectionPool:
    """Open a connection pool, waiting up to `wait_seconds` for the server to accept connections."""
    pool = ConnectionPool(url or database_url(), min_size=1, max_size=max_size, kwargs=connect_kwargs(),
                          check=ConnectionPool.check_connection, open=False, timeout=15,
                          name="mra")
    try:
        pool.open(wait=True, timeout=wait_seconds)
    except PoolTimeout as exc:
        pool.close()
        try:  # one direct attempt to report the real cause (wrong password, unknown host, ...)
            psycopg.connect(url or database_url(), connect_timeout=5).close()
            cause = "the connection pool timed out"
        except Exception as direct:
            cause = _describe(direct)
        raise DatabaseError(
            f"Could not connect to PostgreSQL within {wait_seconds:.0f}s: {cause}. Check DATABASE_URL, that "
            "the server is running (docker compose ... up db) and reachable, and for Neon that the "
            "connection string includes sslmode=require.") from exc
    atexit.register(pool.close)
    return pool


def init_db(pool: ConnectionPool) -> list[str]:
    """Create/upgrade the checkpoint tables and application tables. Safe to run repeatedly."""
    done = []
    PostgresSaver(pool).setup()
    done.append("LangGraph checkpoint tables up to date")
    with pool.connection() as conn, conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(281026)")  # serialise concurrent init runs
        conn.execute("CREATE TABLE IF NOT EXISTS app_schema_migrations "
                     "(version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
        current = conn.execute("SELECT coalesce(max(version), -1) AS v FROM app_schema_migrations").fetchone()["v"]
        for version in range(current + 1, len(MIGRATIONS)):
            for statement in MIGRATIONS[version].split(";\n"):
                if statement.strip():
                    conn.execute(statement)
            conn.execute("INSERT INTO app_schema_migrations (version) VALUES (%s)", (version,))
            done.append(f"applied application migration {version}")
    if len(done) == 1:
        done.append("application tables up to date")
    return done


def schema_problems(pool: ConnectionPool) -> list[str]:
    """What is missing compared with this code version (empty list = ready)."""
    problems = []
    with pool.connection() as conn:
        rows = conn.execute("SELECT to_regclass('checkpoint_migrations') AS cp, "
                            "to_regclass('app_schema_migrations') AS app").fetchone()
        if rows["cp"] is None:
            problems.append("LangGraph checkpoint tables are missing")
        else:
            v = conn.execute("SELECT coalesce(max(v), -1) AS v FROM checkpoint_migrations").fetchone()["v"]
            if v < len(PostgresSaver.MIGRATIONS) - 1:
                problems.append("LangGraph checkpoint tables need upgrading")
        if rows["app"] is None:
            problems.append("application tables are missing")
        else:
            v = conn.execute("SELECT coalesce(max(version), -1) AS v FROM app_schema_migrations").fetchone()["v"]
            if v < len(MIGRATIONS) - 1:
                problems.append("application tables need upgrading")
    return problems


def checkpointer(pool: ConnectionPool) -> PostgresSaver:
    return PostgresSaver(pool)


# ---------------------------------------------------------------- records
@dataclass
class Conversation:
    thread_id: str
    owner_id: str
    title: str
    settings: dict
    created_at: datetime
    updated_at: datetime


@dataclass
class Run:
    run_id: str
    thread_id: str
    submission_id: str
    status: str
    trace_id: str | None
    settings: dict
    report_before: str | None
    attempts: int
    error: str | None
    log: list
    started_at: datetime
    updated_at: datetime
    stale: bool = False  # 'running' but its process stopped updating it (crash, restart, closed tab)


def _conv(row) -> Conversation:
    return Conversation(str(row["thread_id"]), row["owner_id"], row["title"], row["settings"],
                        row["created_at"], row["updated_at"])


def _run(row) -> Run:
    return Run(str(row["run_id"]), str(row["thread_id"]), row["submission_id"], row["status"], row["trace_id"],
               row["settings"], row["report_before"], row["attempts"], row["error"], row["log"],
               row["started_at"], row["updated_at"], bool(row.get("is_stale")))


def _msg(row) -> dict:
    row = dict(row)
    for key in ("message_id", "thread_id", "run_id"):
        row[key] = str(row[key]) if row[key] is not None else None
    return row


class ConversationStore:
    """Conversation metadata, transcript and run records. Every query is scoped by owner."""

    def __init__(self, pool: ConnectionPool, saver: PostgresSaver):
        self.pool = pool
        self.saver = saver

    @contextmanager
    def _conn(self):
        try:
            with self.pool.connection() as conn:
                yield conn
        except (PoolTimeout, psycopg.OperationalError) as exc:
            raise DatabaseError(f"Database unavailable ({type(exc).__name__}). Check that PostgreSQL is "
                                "running and DATABASE_URL is correct, then retry.") from exc

    # -- conversations -------------------------------------------------------
    def create(self, owner: str, title: str = "New conversation", settings: dict | None = None) -> Conversation:
        with self._conn() as conn:
            row = conn.execute(
                "INSERT INTO conversations (thread_id, owner_id, title, settings) VALUES (%s, %s, %s, %s) "
                "RETURNING *", (uuid.uuid4(), owner, clean(title)[:200], Jsonb(clean(settings or {})))).fetchone()
        return _conv(row)

    def get(self, owner: str, thread_id: str) -> Conversation | None:
        try:
            uuid.UUID(str(thread_id))
        except ValueError:
            return None
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE thread_id = %s AND owner_id = %s",
                               (thread_id, owner)).fetchone()
        return _conv(row) if row else None

    def list(self, owner: str) -> list[Conversation]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM conversations WHERE owner_id = %s ORDER BY updated_at DESC",
                                (owner,)).fetchall()
        return [_conv(r) for r in rows]

    def rename(self, owner: str, thread_id: str, title: str) -> bool:
        title = " ".join(clean(title).split())[:200]
        if not title:
            return False
        with self._conn() as conn:
            cur = conn.execute("UPDATE conversations SET title = %s, updated_at = now() "
                               "WHERE thread_id = %s AND owner_id = %s", (title, thread_id, owner))
        return cur.rowcount == 1

    def delete(self, owner: str, thread_id: str) -> bool:
        """Delete a conversation: its LangGraph checkpoints, then transcript/runs/evidence."""
        if not self.get(owner, thread_id):
            return False
        active = self.active_run(thread_id)
        if active and active.status == "running" and not active.stale:
            raise RunConflict("A run is still in progress in this conversation; wait for it to finish.")
        self.saver.delete_thread(thread_id)  # supported checkpointer API: checkpoints, blobs, writes
        with self._conn() as conn:
            conn.execute("DELETE FROM conversations WHERE thread_id = %s AND owner_id = %s", (thread_id, owner))
        return True

    def touch(self, thread_id: str, settings: dict | None = None) -> None:
        with self._conn() as conn:
            if settings is None:
                conn.execute("UPDATE conversations SET updated_at = now() WHERE thread_id = %s", (thread_id,))
            else:
                conn.execute("UPDATE conversations SET updated_at = now(), settings = %s WHERE thread_id = %s",
                             (Jsonb(clean(settings)), thread_id))

    # -- runs ----------------------------------------------------------------
    def start_run(self, thread_id: str, submission_id: str, *, settings: dict, trace_id: str,
                  report_before: str | None, status: str = "running") -> tuple[Run, bool]:
        """Create a run, or return the existing one for this submission (created=False).

        Raises RunConflict when a different run is active. A stale 'running' run is
        marked failed first so a crashed process cannot block the conversation forever.
        """
        with self._conn() as conn, conn.transaction():
            existing = conn.execute(f"SELECT {RUNS} FROM conversation_runs WHERE thread_id = %s "
                                    "AND submission_id = %s",
                                    (thread_id, submission_id)).fetchone()
            if existing:
                return _run(existing), False
            conn.execute("UPDATE conversation_runs SET status = 'failed', updated_at = now(), "
                         "error = 'Interrupted: the app stopped while this run was in progress.' "
                         "WHERE thread_id = %s AND status = 'running' AND updated_at < now() - make_interval(secs => %s)",
                         (thread_id, config.RUN_STALE_SECONDS))
            try:
                with conn.transaction():
                    row = conn.execute(
                        "INSERT INTO conversation_runs (run_id, thread_id, submission_id, status, trace_id, settings, "
                        f"report_before) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING {RUNS}",
                        (uuid.uuid4(), thread_id, submission_id, status, trace_id, Jsonb(clean(settings)),
                         report_before)).fetchone()
            except psycopg.errors.UniqueViolation as exc:
                raise RunConflict("Another run is already in progress in this conversation "
                                  "(maybe in another browser tab). Finish or cancel it first.") from exc
        return _run(row), True

    def get_run(self, run_id: str) -> Run | None:
        with self._conn() as conn:
            row = conn.execute(f"SELECT {RUNS} FROM conversation_runs WHERE run_id = %s", (run_id,)).fetchone()
        return _run(row) if row else None

    def active_run(self, thread_id: str) -> Run | None:
        with self._conn() as conn:
            row = conn.execute(f"SELECT {RUNS} FROM conversation_runs WHERE thread_id = %s AND status = ANY(%s)",
                               (thread_id, list(ACTIVE))).fetchone()
        return _run(row) if row else None

    def latest_run(self, thread_id: str) -> Run | None:
        with self._conn() as conn:
            row = conn.execute(f"SELECT {RUNS} FROM conversation_runs WHERE thread_id = %s "
                               "ORDER BY started_at DESC LIMIT 1", (thread_id,)).fetchone()
        return _run(row) if row else None

    def transition(self, run_id: str, from_status: str, to_status: str) -> bool:
        """Atomically move a run between states; False if another session got there first."""
        try:
            with self._conn() as conn:
                cur = conn.execute("UPDATE conversation_runs SET status = %s, updated_at = now(), "
                                   "error = CASE WHEN %s = 'running' THEN NULL ELSE error END "
                                   "WHERE run_id = %s AND status = %s", (to_status, to_status, run_id, from_status))
        except psycopg.errors.UniqueViolation as exc:
            raise RunConflict("Another run is already in progress in this conversation.") from exc
        return cur.rowcount == 1

    def update_run(self, run_id: str, **fields) -> None:
        allowed = {"status", "error", "attempts", "trace_id", "log", "settings"}
        sets, values = [], []
        for key, value in fields.items():
            assert key in allowed, key
            sets.append(f"{key} = %s")
            values.append(Jsonb(clean(value)) if key in ("log", "settings") else clean(value))
        with self._conn() as conn:
            conn.execute(f"UPDATE conversation_runs SET {', '.join(sets)}, updated_at = now() WHERE run_id = %s",
                         (*values, run_id))

    def heartbeat(self, run_id: str, log: list | None = None) -> None:
        with self._conn() as conn:
            if log is None:
                conn.execute("UPDATE conversation_runs SET updated_at = now() WHERE run_id = %s", (run_id,))
            else:
                conn.execute("UPDATE conversation_runs SET updated_at = now(), log = %s WHERE run_id = %s",
                             (Jsonb(clean(log)), run_id))

    # -- transcript ----------------------------------------------------------
    def add_message(self, thread_id: str, run_id: str, role: str, content: str, *, kind: str = "message",
                    guardrail: dict | None = None, evaluation: dict | None = None,
                    trace_id: str | None = None) -> dict:
        """Insert a transcript message; a second insert for the same run+role returns the first."""
        with self._conn() as conn:
            row = conn.execute(
                "INSERT INTO conversation_messages (message_id, thread_id, run_id, role, kind, content, guardrail, "
                "evaluation, trace_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (run_id, role) DO NOTHING RETURNING *",
                (uuid.uuid4(), thread_id, run_id, role, kind, clean(content), Jsonb(clean(guardrail or {})),
                 Jsonb(clean(evaluation or {})), trace_id)).fetchone()
            if row is None:
                row = conn.execute("SELECT * FROM conversation_messages WHERE run_id = %s AND role = %s",
                                   (run_id, role)).fetchone()
            conn.execute("UPDATE conversations SET updated_at = now() WHERE thread_id = %s", (thread_id,))
        return _msg(row)

    def messages(self, thread_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM conversation_messages WHERE thread_id = %s ORDER BY seq",
                                (thread_id,)).fetchall()
        return [_msg(r) for r in rows]

    def get_message(self, owner: str, message_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT m.* FROM conversation_messages m JOIN conversations c USING (thread_id) "
                               "WHERE m.message_id = %s AND c.owner_id = %s", (message_id, owner)).fetchone()
        return _msg(row) if row else None

    def set_feedback(self, owner: str, message_id: str, value: int | None) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "UPDATE conversation_messages m SET feedback = %s FROM conversations c "
                "WHERE m.thread_id = c.thread_id AND c.owner_id = %s AND m.message_id = %s "
                "AND m.role = 'assistant' RETURNING m.*", (value, owner, message_id)).fetchone()
        return _msg(row) if row else None

    def set_evaluation(self, message_id: str, evaluation: dict) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE conversation_messages SET evaluation = %s WHERE message_id = %s",
                         (Jsonb(clean(evaluation)), message_id))

    # -- evidence ------------------------------------------------------------
    def add_evidence(self, run_id: str, results: list[dict]) -> None:
        rows = [(run_id, clean(r["url"]), clean(r.get("title")), clean(r.get("content") or "")[:4000],
                 clean(r.get("published_date"))) for r in results if r.get("url")]
        if not rows:
            return
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO run_evidence (run_id, url, title, content, published_date) "
                                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (run_id, url) DO NOTHING", rows)

    def evidence(self, *, run_id: str | None = None, thread_id: str | None = None) -> list[dict]:
        """Search results retrieved by one run, or by every run in a conversation."""
        with self._conn() as conn:
            if run_id:
                rows = conn.execute("SELECT url, title, content, published_date FROM run_evidence "
                                    "WHERE run_id = %s ORDER BY created_at", (run_id,)).fetchall()
            else:
                rows = conn.execute("SELECT DISTINCT ON (e.url) e.url, e.title, e.content, e.published_date "
                                    "FROM run_evidence e JOIN conversation_runs r USING (run_id) "
                                    "WHERE r.thread_id = %s ORDER BY e.url, e.created_at", (thread_id,)).fetchall()
        return [dict(r) for r in rows]


def wait_for_database(url: str, timeout: float = 60) -> None:
    """Block until PostgreSQL accepts connections (used by init-db in Docker)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with psycopg.connect(url, connect_timeout=5) as conn:
                conn.execute("SELECT 1")
            return
        except psycopg.OperationalError as exc:
            detail = _describe(exc)
            if re.search(r"password authentication failed|does not exist|no pg_hba.conf entry", detail, re.I):
                raise DatabaseError(f"PostgreSQL refused the connection: {detail}. Retrying will not help: fix "
                                    "DATABASE_URL (local Docker database: POSTGRES_PASSWORD must match the one "
                                    "the volume was created with).") from exc
            if time.monotonic() > deadline:
                raise DatabaseError(f"PostgreSQL not reachable after {timeout:.0f}s: {detail}. "
                                    "Check DATABASE_URL and that the database server is running.") from exc
            time.sleep(2)
