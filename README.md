# Competitor Market Research — LangChain Deep Agents + Streamlit

Enter a product or company name; an orchestrator deep agent delegates every step to a subagent:

1. **discovery_agent** – finds the top 3 competitors (Tavily search)
2. **researcher_agent** – pricing, core features, market position, recent news per competitor (Tavily)
3. **synthesizer_agent** – structured JSON (`/research/structured.json`) + `final_report.md` with comparison table & references
4. **quality_control_agent** – fact-checks attributions, tightens narrative, adds context

## Features
- Research depth: **Basic / Standard / Advanced** (controls Tavily depth, result count, report scope)
- **Human-in-the-loop**: before each subagent step the app pauses with **Approve / Reject / Cancel** buttons; you can edit the task description before approving or give feedback when rejecting. Toggle off in the sidebar for fully autonomous runs.
- Output: rendered report, Markdown source, structured JSON, and **Markdown + PDF downloads**.
- **Save PDF to reports folder** writes the PDF straight to `reports/` (useful when antivirus software blocks browser downloads).
- If a run fails part-way (rate limit, timeout), **Retry from last step** resumes from the last checkpoint.

## Requirements
- An **OpenAI API key** and a **Tavily API key** (https://tavily.com).
- Either **Docker** (recommended — nothing else to install) or **Python 3.12+**.

## Run with Docker (recommended)

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine with the Compose plugin (Linux).

1. **Clone the repository**
   ```bash
   git clone https://github.com/rohitj861/competitor-market-research.git
   cd competitor-market-research
   ```

2. **Configure your API keys** — copy the example file and fill in your keys:
   ```bash
   cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
   ```
   ```ini
   OPENAI_API_KEY=sk-your-real-key
   TAVILY_API_KEY=tvly-your-real-key
   MODEL=openai:gpt-5.4-mini   # optional: any LangChain provider:model string
   ```
   `.env` is read when the container starts; it is never copied into the image or committed to Git.
   (You can also skip this step and paste the keys into the app's sidebar.)

3. **Build and start the app**
   ```bash
   docker compose up --build -d
   ```
   The first build takes a few minutes; later starts are fast.

4. **Open the app** at **http://localhost:8501**

5. **View logs** (optional): `docker compose logs -f`

6. **Stop the app**
   ```bash
   docker compose down
   ```

PDFs saved with **Save PDF to reports folder** appear in the `reports/` folder next to `docker-compose.yml`.

**Port 8501 already in use?** Change the left side of the port mapping in `docker-compose.yml`, e.g. `"8502:8501"`, then open http://localhost:8502.

<details>
<summary>Without Docker Compose (plain <code>docker</code> commands)</summary>

```bash
docker build -t competitor-research .
docker run -d --name competitor-research -p 8501:8501 --env-file .env \
  -v "$(pwd)/reports:/app/reports" competitor-research
# stop and remove
docker stop competitor-research && docker rm competitor-research
```
</details>

## Run locally without Docker
```bash
pip install -r requirements.txt
copy .env.example .env   # add OPENAI_API_KEY and TAVILY_API_KEY (or enter them in the sidebar)
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, streaming progress log, HITL approval flow, downloads |
| `agents.py` | Tavily tool, prompts, subagents, `create_deep_agent` |
| `pdf_export.py` | Markdown → PDF (fpdf2; Unicode fonts from Windows, or DejaVu/Lohit/WenQuanYi in Docker) |
| `Dockerfile` | Python 3.14 slim image with PDF fonts, runs Streamlit on `0.0.0.0:8501` as a non-root user |
| `docker-compose.yml` | One-command start: port 8501, `.env` at runtime, `reports/` volume |
| `.env.example` | Template for the required environment variables |

## Notes
- Agent state is kept in memory, so restarting the container clears any run in progress.
- Each run calls the OpenAI and Tavily APIs and uses your credits; **Basic** depth is the cheapest.
- Emoji have no font in the Docker image and are left out of the PDF (the Markdown output keeps them).
