"""Shared fixtures. Paid/external services are faked; PostgreSQL is a throwaway test server.

Database tests need a disposable PostgreSQL server (never your real one):

    docker run -d --rm --name mra-test-pg -e POSTGRES_USER=mra_test -e POSTGRES_PASSWORD=test_only_pw \
        -e POSTGRES_DB=mra_test -p 127.0.0.1:55432:5432 postgres:17-alpine

Each test session creates its own uniquely named database on that server and drops only
that database afterwards. Override the server with TEST_DATABASE_URL.
"""

import os
import sys
import uuid
from pathlib import Path

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# keep tests hermetic: no real keys, no Langfuse, regardless of the developer's .env
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ["TAVILY_API_KEY"] = "test-tavily-key-not-real"

from tests.fakes import FakeResearchModel, FakeSearchClient  # noqa: E402

ADMIN_URL = os.getenv("TEST_DATABASE_URL", "postgresql://mra_test:test_only_pw@127.0.0.1:55432/mra_test")


def _server_available() -> bool:
    try:
        with psycopg.connect(ADMIN_URL, connect_timeout=3):
            return True
    except psycopg.OperationalError:
        return False


@pytest.fixture(scope="session")
def database_url():
    if not _server_available():
        pytest.skip("test PostgreSQL not running (see tests/conftest.py for the docker command)")
    name = f"mra_test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    yield ADMIN_URL.rsplit("/", 1)[0] + f"/{name}"
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture(scope="session")
def pool(database_url):
    import memory

    p = memory.open_pool(database_url, wait_seconds=10)
    memory.init_db(p)
    yield p
    p.close()


@pytest.fixture
def store(pool):
    import memory

    return memory.ConversationStore(pool, memory.checkpointer(pool))


class RecordingTracer:
    """Tracer stand-in that records what would be sent to Langfuse."""

    def __init__(self):
        from observability import Tracer
        import config

        self._real = Tracer(config.LangfuseSettings(False, "disabled in tests"))
        self.scores: list[dict] = []
        self.turns: list[dict] = []

    enabled = False

    def callbacks(self):
        return []

    def turn(self, **kwargs):
        self.turns.append(kwargs)
        return self._real.turn(**kwargs)

    def score(self, **kwargs):
        self.scores.append(kwargs)
        return True

    def status(self, check=False):
        return "disabled", "tests"


@pytest.fixture
def tracer():
    return RecordingTracer()


@pytest.fixture
def fake_model():
    return FakeResearchModel()


@pytest.fixture
def search_client():
    return FakeSearchClient()


@pytest.fixture
def runner(store, tracer, fake_model, search_client):
    from agents import build_agent
    from runner import Runner

    def factory(**kwargs):
        kwargs["model"] = fake_model
        return build_agent(**kwargs)

    return Runner(store, tracer, agent_factory=factory, search_client=search_client)

