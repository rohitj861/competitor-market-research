"""LangChain Deep Agents setup for the competitor market-research pipeline."""

import os
from datetime import date
from typing import Literal

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

DEFAULT_MODEL = os.getenv("MODEL", "openai:gpt-5.4-mini")

REPORT_PATH = "/final_report.md"
JSON_PATH = "/research/structured.json"

# Research depth presets: how hard each agent searches and how rich the report is.
DEPTHS = {
    "basic": {
        "search_depth": "basic",
        "max_results": 3,
        "searches_per_competitor": 1,
        "report": "a concise report (~500 words): overview, comparison table, key takeaways, references",
    },
    "standard": {
        "search_depth": "basic",
        "max_results": 5,
        "searches_per_competitor": 2,
        "report": "a standard report (~1,000 words): executive summary, per-competitor profiles, "
        "comparison table, positioning analysis, key takeaways, references",
    },
    "advanced": {
        "search_depth": "advanced",
        "max_results": 8,
        "searches_per_competitor": 4,
        "report": "an in-depth report (~2,000 words): executive summary, detailed per-competitor profiles, "
        "comparison table, SWOT per competitor, pricing analysis, recent-news timeline, "
        "strategic recommendations, references",
    },
}


def make_search_tool(depth: str):
    """Build a Tavily search tool whose defaults follow the chosen research depth."""
    cfg = DEPTHS[depth]
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def internet_search(
        query: str,
        topic: Literal["general", "news"] = "general",
        max_results: int | None = None,
    ) -> list[dict]:
        """Search the web with Tavily. Use topic="news" for recent news.

        Returns a list of results with title, url, content and (for news) published_date.
        """
        try:
            response = client.search(
                query,
                topic=topic,
                search_depth=cfg["search_depth"],
                max_results=max_results or cfg["max_results"],
            )
        except Exception as exc:  # a failed search shouldn't kill the whole run
            return [{"error": f"Search failed ({type(exc).__name__}: {exc}). Try a different query."}]
        return [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
                "published_date": r.get("published_date"),
            }
            for r in response.get("results", [])
        ]

    return internet_search


# Content and formatting contract shared by the synthesizer and quality-control agents
# (the report is shown as Markdown and exported to PDF).
REPORT_RULES = """   Content:
   - The report analyses the competitors FROM THE TARGET'S POINT OF VIEW. Title it
     "<Target>: Competitor Analysis", and compare every competitor against the target.
   - Comparison table columns: Company, Pricing, Core features, Market position. The FIRST row
     is the target, then one row per competitor. Cells are specific (actual prices, user
     numbers, named features), 8-25 words, no line breaks or lists inside cells.
   - "Key takeaways" is a numbered list (1. 2. 3. ...), each item stating what it means for the
     target.
   - Citations: cite inline as [n]. Each DISTINCT source URL gets its own reference number —
     never group several sources under one reference. Expect roughly 8-20 references.
   Formatting:
   - One `#` title, `##` for sections, `###` for each company; no deeper headings.
   - Leave a blank line before and after every heading, table and list.
   - References section: one per line, separated by blank lines, as
     `[n] Source title — https://full-url`.
   - Plain Markdown only: no HTML, no emojis, no footnote syntax."""


def build_prompts(depth: str) -> dict[str, str]:
    cfg = DEPTHS[depth]
    today = date.today().isoformat()

    orchestrator = f"""You are the orchestrator of a market-research team. Today is {today}.
The user gives you a product or company name. Research depth: {depth.upper()}.

Follow these steps strictly. EVERY STEP MUST BE DELEGATED TO A SUBAGENT with the `task` tool.
Never do the research or writing yourself.

1. Call `discovery_agent` to find the top 3 competitors of the target.
   It must return ONLY the 3 competitor names.
2. Call `researcher_agent` once per competitor AND once for the target itself (you may run
   these in parallel). Each call collects pricing, core features, market position and recent
   news, and saves notes to /research/<slug>.md.
3. Call `synthesizer_agent` to turn the notes into validated, structured JSON
   ({JSON_PATH}) and write {REPORT_PATH}.
4. Call `quality_control_agent` to verify facts and attributions, improve narrative flow and
   readability, add context where necessary, and save the improved report to {REPORT_PATH}.

Subagents only see the task description you write, so ALWAYS name the target in every task
(e.g. "Target: WhatsApp. Competitors: ...") — the report is a competitor analysis FOR the target.

When a subagent call is rejected by the human reviewer, read their feedback, adjust the task
description accordingly and try again (or skip the step if they ask you to).

When all steps are done, reply with ONLY the final contents of {REPORT_PATH}."""

    discovery = f"""You are the discovery agent. Today is {today}.
Use `internet_search` to identify the top 3 direct competitors of the given product or company
(same market, similar customers). Prefer recent sources and analyst/comparison articles.
Return ONLY the 3 competitor names, one per line. No other text."""

    researcher = f"""You are the research analysis agent. Today is {today}.
For the single company/product you are given (a competitor, or the target itself), use `internet_search` (about
{cfg['searches_per_competitor']} searches per topic below; use topic="news" for news) to collect:
- Pricing (plans, tiers, price points, currency, free tier)
- Core features and differentiators
- Market position (target customers, market share, funding/revenue, notable customers)
- Recent news (last 12 months, with dates)

Record every fact with its source URL. Do not invent numbers; write "not publicly disclosed"
when a fact cannot be found.
Save your notes as Markdown to /research/<competitor-slug>.md using `write_file`
(slug = lowercase name with hyphens). Return a 3-line summary and the file path."""

    synthesizer = f"""You are the synthesizer agent. Today is {today}.
1. Read all files in /research/ (use `ls` and `read_file`).
2. Transform the unstructured notes into validated, structured JSON and save it to {JSON_PATH}.
   Schema: {{"target": str, "generated_on": str, "competitors": [{{"name": str,
   "pricing": str, "core_features": [str], "market_position": str,
   "recent_news": [{{"date": str, "headline": str, "source": str}}],
   "sources": [str]}}]}}
   The JSON must be valid (double quotes, no trailing commas, no comments).
3. Synthesize the findings into {REPORT_PATH}: {cfg['report']}.
{REPORT_RULES}
Return a one-line confirmation."""

    quality_control = f"""You are the critique / quality-control agent. Today is {today}.
Read {REPORT_PATH} and the notes in /research/. Then review and tighten the report:
- Verify every fact and attribution against the notes; use `internet_search` to spot-check
  doubtful or high-impact claims. Remove or flag anything unsupported.
- Make sure every citation [n] matches a reference URL.
- Improve narrative flow and readability; add context where a reader would need it.
- Keep the report framed around the target, and keep the comparison table, Key takeaways list
  and References section. Never merge or drop references — every distinct source URL keeps its
  own [n]. The final report must follow these rules:
{REPORT_RULES}
Overwrite {REPORT_PATH} with the improved version using `write_file`.
Return a short bullet list of the changes you made."""

    return {
        "orchestrator": orchestrator,
        "discovery": discovery,
        "researcher": researcher,
        "synthesizer": synthesizer,
        "quality_control": quality_control,
    }


def build_agent(depth: str = "standard", hitl: bool = True, model: str = DEFAULT_MODEL):
    """Create the orchestrator deep agent with its four subagents.

    With `hitl=True`, every subagent delegation (the `task` tool) pauses for human approval.
    """
    search = make_search_tool(depth)
    prompts = build_prompts(depth)

    subagents = [
        {
            "name": "discovery_agent",
            "description": "Finds the top 3 competitors of a product/company. Returns only their names.",
            "system_prompt": prompts["discovery"],
            "tools": [search],
        },
        {
            "name": "researcher_agent",
            "description": "Researches ONE competitor: pricing, core features, market position, recent news. "
            "Saves notes to /research/<competitor-slug>.md.",
            "system_prompt": prompts["researcher"],
            "tools": [search],
        },
        {
            "name": "synthesizer_agent",
            "description": f"Turns /research/ notes into structured JSON ({JSON_PATH}) and writes {REPORT_PATH} "
            "with a comparison table and references.",
            "system_prompt": prompts["synthesizer"],
            "tools": [],
        },
        {
            "name": "quality_control_agent",
            "description": f"Fact-checks, edits and tightens {REPORT_PATH}.",
            "system_prompt": prompts["quality_control"],
            "tools": [search],
        },
    ]

    interrupt_on = (
        {
            "task": {
                "allowed_decisions": ["approve", "edit", "reject"],
                "description": "The orchestrator wants to delegate a step to a subagent.",
            }
        }
        if hitl
        else None
    )

    return create_deep_agent(
        model=model,
        tools=[search],
        system_prompt=prompts["orchestrator"],
        subagents=subagents,
        interrupt_on=interrupt_on,
        checkpointer=InMemorySaver(),  # required for human-in-the-loop pause/resume
    )
