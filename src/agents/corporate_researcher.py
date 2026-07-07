from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm_providers.openai import get_openai_llm
from src.tools.web_search import format_results, search_web

_llm = get_openai_llm()

INSTRUCTIONS = (
    "You are a corporate AI healthcare analyst. "
    "Use only the provided sources, prioritizing recent and reachable sources. "
    "If older references are used, briefly explain why they still matter. "
    "Return concise bullet points with citations [title - url]."
)


def corporate_researcher_node(state: dict) -> dict:
    query = state["query"]
    sources = search_web(
        f"{query} (site:microsoft.com OR site:google.com OR site:ibm.com OR site:who.int OR site:fda.gov OR site:ema.europa.eu OR site:healthcareitnews.com)",
        max_results=12,
        recent=True,
        verify_urls=True,
    )

    prompt = (
        f"Research question: {query}\n\n"
        "Corporate/industry web findings:\n"
        f"{format_results(sources)}\n\n"
        "Task: summarize products, deployments, regulation, and business trends, emphasizing current developments."
    )

    response = _llm.invoke([SystemMessage(content=INSTRUCTIONS), HumanMessage(content=prompt)])
    return {
        "messages": [response],
        "sources": [item.url for item in sources],
    }
