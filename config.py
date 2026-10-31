"""Central configuration: reads environment variables and validates them before use.

Every service checks its settings here first, so a missing value or an unchanged
placeholder from `.env.example` fails with a message naming the variable to fix,
instead of an opaque error deep inside a client library.
"""

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Placeholders used by .env.example (YOUR_..._HERE) and by older versions of it (sk-..., your-...).
_PLACEHOLDER = re.compile(r"^(YOUR_[A-Z0-9_]+_HERE|your-.*|.*\.\.\.|<.*>|changeme)$", re.I)

GUIDANCE = {
    "OPENAI_API_KEY": "create one at https://platform.openai.com/api-keys",
    "TAVILY_API_KEY": "create one at https://app.tavily.com",
    "DATABASE_URL": "a PostgreSQL connection string, e.g. postgresql://user:password@localhost:5432/mra "
                    "(Neon: copy it from the Neon console > Connect)",
    "POSTGRES_PASSWORD": "any password (letters and digits) for the local Docker database; set it before "
                         "the database volume is first created",
    "LANGFUSE_PUBLIC_KEY": "Langfuse > Project settings > API keys",
    "LANGFUSE_SECRET_KEY": "Langfuse > Project settings > API keys",
    "LANGFUSE_BASE_URL": "https://cloud.langfuse.com (EU), https://us.cloud.langfuse.com (US) "
                         "or your self-hosted Langfuse URL",
}


class ConfigError(RuntimeError):
    """A required setting is missing or still a placeholder."""

    def __init__(self, names: list[str], purpose: str):
        self.names = names
        lines = [f"{purpose} needs configuration. Set these in your .env file (or the environment):"]
        lines += [f"  - {n}: {GUIDANCE.get(n, 'see .env.example')}" for n in names]
        super().__init__("\n".join(lines))


def is_placeholder(value: str | None) -> bool:
    return not value or not value.strip() or bool(_PLACEHOLDER.match(value.strip()))


def get(name: str, default: str | None = None) -> str | None:
    """Return a setting, treating placeholders as unset."""
    value = os.getenv(name)
    return default if is_placeholder(value) else value.strip()


def missing(*names: str) -> list[str]:
    return [n for n in names if get(n) is None]


def require(*names: str, purpose: str) -> dict[str, str]:
    """Return the named settings, or raise ConfigError listing every missing one."""
    if absent := missing(*names):
        raise ConfigError(absent, purpose)
    return {n: get(n) for n in names}


def flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- app settings
DEFAULT_MODEL = get("MODEL", "openai:gpt-5.4-mini")
# The app has no login: every conversation belongs to this single local owner.
OWNER_ID = get("APP_OWNER_ID", "local")
# Bounded automatic repair when a finished report fails structural validation.
MAX_OUTPUT_RETRIES = 1
# A run whose heartbeat is older than this is treated as interrupted (crash/restart).
RUN_STALE_SECONDS = 120


@dataclass(frozen=True)
class LangfuseSettings:
    enabled: bool
    reason: str  # why tracing is off, shown in the UI
    public_key: str | None = None
    secret_key: str | None = None
    base_url: str | None = None


def langfuse_settings() -> LangfuseSettings:
    """Langfuse is optional: off when LANGFUSE_TRACING_ENABLED=false or keys are not configured."""
    if not flag("LANGFUSE_TRACING_ENABLED", True):
        return LangfuseSettings(False, "disabled (LANGFUSE_TRACING_ENABLED=false)")
    if absent := missing("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        return LangfuseSettings(False, "not configured: set " + " and ".join(absent)
                                + " in .env, or LANGFUSE_TRACING_ENABLED=false to silence this")
    return LangfuseSettings(True, "enabled", get("LANGFUSE_PUBLIC_KEY"), get("LANGFUSE_SECRET_KEY"),
                            get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"))
