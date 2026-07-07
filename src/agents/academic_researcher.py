from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm_providers.gemini import get_gemini_llm
from src.tools.web_search import format_results, search_web

_llm = get_gemini_llm()

INSTRUCTIONS = (
    "You are an academic AI healthcare researcher. "
    "Use only the provided sources, prioritizing recent and reachable sources. "
    "If older studies are used, explicitly justify their relevance. "
    "Return concise bullet points with citations [title - url]."
)


def academic_researcher_node(state: dict) -> dict:
    query = state["query"]
    sources = search_web(
        f"{query} site:arxiv.org OR site:nature.com OR site:nih.gov OR site:thelancet.com OR site:nejm.org OR site:bmj.com",
        max_results=12,
        recent=True,
        verify_urls=True,
    )

    prompt = (
        f"Research question: {query}\n\n"
        "Academic web findings:\n"
        f"{format_results(sources)}\n\n"
        "Task: summarize key academic evidence (methods, outcomes, limitations), emphasizing the most recent findings."
    )

    response = _llm.invoke([SystemMessage(content=INSTRUCTIONS), HumanMessage(content=prompt)])
    return {
        "messages": [response],
        "sources": [item.url for item in sources],
    }
