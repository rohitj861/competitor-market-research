# Competitor Market Research — LangChain Deep Agents + Streamlit

Enter a product or company name; an orchestrator deep agent delegates every step to a subagent:

1. **discovery_agent** – finds the top 3 competitors (Tavily search)
2. **researcher_agent** – pricing, core features, market position, recent news per competitor (Tavily)
3. **synthesizer_agent** – structured JSON (`/research/<target>/structured.json`) + `final_report.md` with comparison table & references
4. **quality_control_agent** – fact-checks attributions, tightens narrative, adds context

Then keep talking: follow-up questions in the same conversation are answered from the earlier
research (the agent remembers the conversation), and you can start a new research target at any time.

## Features
- Research depth: **Basic / Standard / Advanced** (controls Tavily depth, result count, report scope)
- **Human-in-the-loop**: before each subagent step the app pauses with **Approve / Reject / Cancel** buttons; you can edit the task description before approving or give feedback when rejecting. Toggle off in the sidebar for fully autonomous runs.
- **Persistent conversations** (PostgreSQL): create, list, reopen, rename and delete conversations; each one keeps its full transcript and agent memory across browser refreshes and app/Docker restarts. A paused approval step is still waiting for you after a restart.
- **Langfuse observability** (optional): one trace per request, grouped into a session per conversation, with every model call, tool call and subagent, token usage, guardrail outcomes, quality scores and your 👍/👎 feedback.
- **Guardrails AI** validation of what you send, of the web pages the agents read, and of every response before it is displayed or exported.
- **Evaluation**: a synthetic dataset and a repeatable experiment command (free offline mode; paid live and LLM-as-judge modes are opt-in).
- Output: rendered report, Markdown source, and **Markdown + PDF downloads**; **Save PDF to reports folder** writes the PDF straight to `reports/` (useful when antivirus software blocks browser downloads).
- If a run fails part-way (rate limit, timeout, app restart), **Retry from last step** resumes from the last checkpoint.

## Requirements
- An **OpenAI API key** and a **Tavily API key** (https://tavily.com).
- A **PostgreSQL** database: an external one such as [Neon](https://neon.tech), or the local PostgreSQL container provided here.
- Either **Docker** (recommended) or **Python 3.12+**.
- Optional: a [Langfuse](https://langfuse.com) project (cloud or self-hosted) for tracing.

## Configuration (`.env`)

Copy the template and fill it in (Windows PowerShell: `Copy-Item .env.example .env`). `.env` is
git-ignored and excluded from Docker builds; it is read at runtime only.

| Variable | Required | Where it comes from |
|---|---|---|
| `OPENAI_API_KEY` | yes | https://platform.openai.com/api-keys (or paste it in the app sidebar) |
| `TAVILY_API_KEY` | yes | https://app.tavily.com (or paste it in the sidebar) |
| `MODEL` | no | any LangChain `provider:model` string (default `openai:gpt-5.4-mini`) |
| `DATABASE_URL` | yes* | PostgreSQL URL, e.g. Neon console → *Connect* (keep `?sslmode=require`) |
| `POSTGRES_PASSWORD` | local DB only | any password (letters/digits) for `docker-compose.local-db.yml` |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | no | Langfuse → *Project settings* → *API keys* |
| `LANGFUSE_BASE_URL` | no | `https://cloud.langfuse.com` (EU), `https://us.cloud.langfuse.com` (US) or your self-hosted URL |
| `LANGFUSE_TRACING_ENABLED` | no | `false` turns tracing off explicitly |
| `JUDGE_MODEL`, `APP_OWNER_ID`, `APP_PORT`, `POSTGRES_PORT` | no | see `.env.example` |

\* With `docker-compose.local-db.yml` the containers build their own `DATABASE_URL` from `POSTGRES_PASSWORD`.

Values still set to a placeholder (`YOUR_..._HERE`) count as missing. Check your setup without printing any secret:

```powershell
python manage.py check-config
```

There is **no fallback to temporary memory**: without a working database the app shows what to fix
and refuses to run, so conversations are never silently lost. Guardrails AI needs no key (all
validators run locally). Langfuse is optional; without it the sidebar shows *Langfuse tracing: disabled*.

## Run with Docker (recommended)

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine with the Compose plugin (Linux).

1. **Clone the repository**
   ```bash
   git clone https://github.com/rohitj861/competitor-market-research.git
   cd competitor-market-research
   ```
2. **Configure** `.env` (see above).
3. **Start** — choose one:

   **A. Local PostgreSQL in Docker** (needs `POSTGRES_PASSWORD` in `.env`, set *before* the first start):
   ```powershell
   docker compose -f docker-compose.yml -f docker-compose.local-db.yml up --build -d
   ```
   **B. External PostgreSQL / Neon** (needs `DATABASE_URL` in `.env`; no local database is started):
   ```powershell
   docker compose up --build -d
   ```
   The one-shot `init-db` service waits for the database (up to 90 s), creates or upgrades the
   tables, and exits; the app starts only after it succeeds. The first build takes a few minutes.
4. **Open the app** at **http://localhost:8501**
5. **Logs**: `docker compose logs -f app` (setup: `docker compose logs init-db`)
6. **Stop**: `docker compose down` (add the same `-f` files you started with). This keeps your data.

PDFs saved with **Save PDF to reports folder** appear in `reports/` next to `docker-compose.yml`.
**Port 8501 or 5432 already in use?** Set `APP_PORT=8502` / `POSTGRES_PORT=5433` in `.env` (if you change
`POSTGRES_PORT`, change the port in `DATABASE_URL` too for runs outside Docker).

Builds are reproducible: `constraints.txt` pins every transitive dependency to the versions in the
tested image. After changing `requirements.txt`, rebuild and regenerate it with
`docker run --rm --entrypoint pip competitor-research:latest freeze > constraints.txt` (keep the header comment).
The one-shot `init-db` service receives only `DATABASE_URL`, not your API keys.

> ⚠️ **Local conversation data lives in the Docker volume `mra-pgdata`.**
> `docker compose ... down -v`, `docker volume rm` or *Clean / Purge data* in Docker Desktop
> **permanently delete all local conversations**. Back up first (below).

## Run locally without Docker (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env        # then fill in the values
# Optional local database in Docker only (port 5432 on localhost):
docker compose -f docker-compose.yml -f docker-compose.local-db.yml up -d db
#   and in .env: DATABASE_URL=postgresql://mra:<POSTGRES_PASSWORD>@localhost:5432/mra
python manage.py init-db           # explicit, repeatable; run again after upgrading the app
streamlit run app.py
```

## Database setup and migrations

`python manage.py init-db` (Docker: runs automatically as the `init-db` service, or manually with
`docker compose run --rm init-db`) runs LangGraph's `PostgresSaver.setup()` and the application
migrations in `memory.py`. It is idempotent, only ever adds tables, and never deletes data. The app
checks the schema at startup and tells you to run it if something is missing; it never changes
the schema itself during normal use.

Tables: LangGraph's `checkpoint*` tables (agent state per conversation) plus `conversations`,
`conversation_messages` (full transcript), `conversation_runs` (one active run per conversation,
progress log, status) and `run_evidence` (search results used for evaluation).

## Conversations

- The first message of a new conversation is the product/company to research. Later messages are
  follow-up questions answered from the conversation (e.g. *"Which competitor is cheapest for a
  10-person team?"*), or `Research the competitors of: <name>` for a new target.
- **New conversation**, the **conversation list**, **Rename** and **Delete** are in the sidebar.
  Delete asks for confirmation and removes the transcript, runs, evidence and the agent's
  checkpoints. It does not delete traces already sent to Langfuse.
- Times in the conversation list and report headers are shown in UTC.
- The open conversation is in the URL (`?thread=<id>`), so refresh/bookmark works; the database is
  the source of truth. Each conversation is a separate LangGraph thread: memory is never shared
  between conversations.
- The full transcript is stored separately from the model context. The agent's own context is
  summarised automatically when it grows (deepagents summarisation), but the transcript you see is
  never trimmed.
- Only one run can be active per conversation; a second tab sees *"A run is in progress"*.
  Submitting the same message twice (double click, rerun) never creates duplicates. If the app stops
  mid-run, the conversation shows the interruption with **Retry from last step**.
- **Single-user application:** there is no login. Every conversation belongs to `APP_OWNER_ID`
  (default `local`), and all queries are scoped by it, but anyone who can open the app can see
  all conversations. Put it behind authentication (e.g. a reverse proxy) before sharing it.
- Conversations from before this version were held only in memory and were never stored, so they
  cannot be recovered.

## Observability (Langfuse)

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL`, then restart the app. In Langfuse:

- **Sessions** → one session per conversation (`session_id` = thread id); **Traces** → one trace per
  request (`research-turn`), including HITL pauses/resumes, with nested subagents (`task`), tool calls
  (`internet_search`, `write_file`), model generations, timings, errors and the token usage the
  provider reports (cost appears when Langfuse knows the model's price; nothing is estimated).
- **Scores** on each trace: `guardrail.input` / `guardrail.output` (pass/redact/reject/flagged),
  `eval.*` deterministic quality scores, and `user_feedback` from 👍/👎.
- Everything sent is masked first: API keys, tokens, passwords and personal data become
  `[REDACTED:type]`. If Langfuse is unreachable the app keeps working and the sidebar shows
  *unavailable*.

## Evaluation

```powershell
python manage.py eval                     # offline: fixture responses + real guardrails, free
python manage.py eval --langfuse          # same, uploaded as dataset "mra-synthetic-v1" + a dataset run
python manage.py eval --mode live --confirm-cost --items research-basic-notion   # real agent: PAID
python manage.py eval --judge --confirm-cost                                     # LLM-as-judge: PAID
```

Results are written to `eval_results/<run>.json` (scores, comments, evaluator/judge versions,
score definitions). The dataset (`eval_dataset.json`, fictional companies) covers normal research,
follow-ups, thread isolation, legitimate comparisons, and failure cases (empty, oversized, secret,
injection, leaked secret in output).

- **Deterministic scores** (`evaluation.py`, also attached to every live response): relevance,
  completeness against explicit requirements, citation integrity, **citation support** (cited URLs
  that were actually retrieved) and **groundedness** (numeric claims found in the retrieved evidence).
  Without evidence these are reported as *unavailable*, not guessed. A well-formed citation, or a
  URL that resolves, is not proof that a claim is correct.
- **LLM-as-judge** (opt-in): 1–5 rubric for relevance, completeness and groundedness against the
  evidence, normalised to 0–1, recorded with model and rubric version.
- **Cost guidance**: live mode runs the full agent, roughly 15–40 model calls and 5–40 Tavily
  searches per research item (Basic is cheapest); the judge adds one model call per item. Paid modes
  refuse to run without `--confirm-cost`. Use `--items` to limit a run. Live mode uses throwaway
  conversations (owner `eval:<run>`) that are deleted afterwards unless `--keep-threads` is set.

Guardrails (runtime allow/redact/block decisions) and evaluation (quality measurement) are kept
separate: different modules and different score names.

## Guardrails

Built on the [Guardrails AI](https://github.com/guardrails-ai/guardrails) library (`guardrails-ai`)
with validators registered locally in `guards.py`: no Hub token, no remote inference, no model
downloads, and Guardrails' own telemetry is switched off. They are lightweight regex/heuristic
checks, so they need no GPU and negligible CPU/RAM.

| Stage | Checks | Outcome |
|---|---|---|
| Your message (before the agent runs) | empty / > 2,000 characters; credentials (API keys, tokens, private keys, passwords); instruction-override / prompt-extraction phrases | **reject**: nothing is sent; the transcript stores a redacted copy |
| | personal data (email, phone, SSN, payment card) | **redact** and continue |
| Web search results (untrusted) | instruction-like text, credentials, personal data | **redact** before any agent reads them; if validation cannot run, results are withheld |
| Final report | empty | **reject** (not displayed or exported) |
| | references not retrieved by the agents' own searches | prevented by the prompts; measured by the `citation_support` score |
| | credentials / personal data | **redact** |
| | report structure (title, comparison table, key takeaways, references) and citations | **retry**: one automatic repair turn; if still failing, shown and exportable with a visible *flagged* warning |
| Follow-up answer | empty → reject; credentials/PII → redact; no citation in a long answer → advisory note |

The progress log is redacted before it is shown, and Markdown/PDF exports are validated again just
before download. A validator that fails to load or crashes blocks the text (**fail closed**); it is
never skipped. Legitimate company names and comparisons (e.g. *"Compare Notion vs Coda; ignore
pricing"*) pass. Guardrails reduce risk but do **not** guarantee factual accuracy and cannot stop
every prompt injection.

## Backup and recovery (local Docker database)

```powershell
# backup
docker compose -f docker-compose.yml -f docker-compose.local-db.yml exec -T db pg_dump -U mra -d mra -Fc > mra-backup.dump
# restore into the (running) local database
Get-Content mra-backup.dump -AsByteStream -ReadCount 0 | docker compose -f docker-compose.yml -f docker-compose.local-db.yml exec -T db pg_restore -U mra -d mra --clean --if-exists
```
(PowerShell 5.1: use `cmd /c "docker compose ... exec -T db pg_restore ... < mra-backup.dump"` instead of `Get-Content -AsByteStream`.)
For Neon, use Neon's branching / point-in-time restore, or `pg_dump "$env:DATABASE_URL"`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| *Persistent conversation memory is not available … DATABASE_URL* | Set `DATABASE_URL` (or `POSTGRES_PASSWORD` for the local DB); `python manage.py check-config` |
| *The database is not initialised* | `python manage.py init-db` (Docker: `docker compose run --rm init-db`) |
| *Could not connect to PostgreSQL* | Is the DB running (`docker compose ps`)? Neon: include `sslmode=require`; Neon may need a few seconds to wake up, so retry. |
| `password authentication failed` with the local DB | The volume was initialised with an earlier `POSTGRES_PASSWORD`; use that one (changing the variable does not change an existing database). `init-db` stops immediately on this error instead of retrying. |
| *Could not save the PDF* (Linux Docker hosts) | `reports/` was created by root: `sudo chown 1000:1000 reports`, or use **Download PDF** |
| `init-db` exited with an error, app not started | `docker compose logs init-db` |
| *A run is in progress … in another tab* | Wait for it, or after a crash wait ~2 minutes and use **Retry from last step** / **Discard this run** |
| Sidebar: *Langfuse tracing: unavailable* | Check keys and `LANGFUSE_BASE_URL` (region!); research continues without tracing |
| *Request not sent to the research agents* | A guardrail refused the message (e.g. it contained an API key); rephrase it |

## Tests

```powershell
pip install -r requirements-dev.txt
# disposable test database (never point tests at your real one)
docker run -d --rm --name mra-test-pg -e POSTGRES_USER=mra_test -e POSTGRES_PASSWORD=test_only_pw -e POSTGRES_DB=mra_test -p 127.0.0.1:55432:5432 postgres:17-alpine
python -m pytest tests -q
docker stop mra-test-pg
```
The model and web search are replaced by deterministic fakes, and Langfuse by an in-memory
exporter, so the tests cost nothing; each run creates and drops its own temporary database.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: conversations, transcript, streaming progress, HITL approval, feedback, downloads |
| `runner.py` | One conversation turn end-to-end: guardrails → run lock → agent → output validation/repair → transcript → scores |
| `agents.py` | Tavily tool (with untrusted-content sanitising), prompts, subagents, `create_deep_agent` |
| `memory.py` | PostgreSQL pool, `PostgresSaver`, migrations, conversation/transcript/run store |
| `guards.py` / `redaction.py` | Guardrails AI validators and policy / secret & PII redaction |
| `observability.py` | Langfuse tracing, masking, scores, feedback |
| `evaluation.py` / `experiments.py` / `eval_dataset.json` | Evaluators, experiment runner, synthetic dataset |
| `config.py` / `manage.py` | Configuration validation / CLI (`check-config`, `init-db`, `eval`) |
| `pdf_export.py` | Markdown → PDF (fpdf2; Unicode fonts from Windows, or DejaVu/Lohit/WenQuanYi in Docker) |
| `Dockerfile`, `docker-compose.yml`, `docker-compose.local-db.yml` | Image, app + init-db, optional local PostgreSQL |
| `constraints.txt` | Exact versions of every dependency in the tested image (used by the Dockerfile) |
| `tests/` | Automated tests (fakes for paid services) |

## Notes
- Each run calls the OpenAI and Tavily APIs and uses your credits; **Basic** depth is the cheapest.
- Emoji have no font in the Docker image and are left out of the PDF (the Markdown output keeps them).
- Reloading the page while agents are working interrupts the run; use **Retry from last step**
  (completed steps are not repeated).
