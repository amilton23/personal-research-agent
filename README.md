# Personal Research Agent (AI in Healthcare)

Simple multi-agent workflow for gathering **academic** and **corporate** internet research about AI in healthcare.

## What it does

- Uses **LangGraph** to orchestrate two agents in parallel.
- Uses **OpenRouter** as provider for both model families:
  - OpenAI model (corporate analysis + final synthesis)
  - Gemini model (academic analysis)
- Uses a live web search utility (DuckDuckGo Lite + HTML fallback parsing).
- Prioritizes recent and reachable links to reduce stale references.
- Produces a single final report with references.

## Architecture

1. `start_query`
2. Parallel fan-out:
   - `academic_researcher` (Gemini via OpenRouter)
   - `corporate_researcher` (OpenAI via OpenRouter)
3. `summarize` (OpenAI via OpenRouter)
4. End

This keeps the design intentionally minimal (YAGNI) and with clear responsibilities (SOLID).

## Setup

1. Install dependencies:

```bash
pip install -e .
```

2. Create `.env/.env` with at least:

```env
OPENROUTER_API_KEY=your_key_here

# Optional model config
# OPENROUTER_OPENAI_MODEL=openai/gpt-4o-mini
# GEMINI_FOUNDATION_MODEL=google/gemini-3.1-flash-lite
# OPENROUTER_HTTP_REFERER=https://your-app-url
# OPENROUTER_APP_TITLE=Personal Research Agent

# Optional LangSmith tracing (recommended only with a valid dedicated key)
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=lsv2_pt_xxx
# LANGSMITH_PROJECT=personal-research-agent
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

## Run

```bash
python main.py "corporate and academic research on AI in healthcare"
```

## Main files

- `src/graph/research_graph.py` - workflow orchestration
- `src/agents/academic_researcher.py` - academic agent
- `src/agents/corporate_researcher.py` - corporate agent
- `src/tools/web_search.py` - live web search helper with freshness + URL reachability checks
- `src/utils/langsmith.py` - safe LangSmith tracing bootstrap/validation
- `src/llm_providers/` - OpenRouter-backed model factories
