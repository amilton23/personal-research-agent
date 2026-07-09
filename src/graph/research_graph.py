from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from src.agents.academic_researcher import academic_researcher_node
from src.agents.corporate_researcher import corporate_researcher_node
from src.llm_providers.openai import get_openai_llm
from src.utils.logs import get_logger


class ResearchState(TypedDict):
    query: str
    sources: Annotated[list[str], operator.add]
    final_report: str
    messages: Annotated[list[BaseMessage], operator.add]


logger = get_logger(__name__)
_summarizer = get_openai_llm()
_SUMMARY_INSTRUCTIONS = (
    "You are a senior AI healthcare analyst. "
    "Merge academic and corporate findings into a short report with sections:\n"
    "1) Key Academic Findings\n"
    "2) Industry Initiatives\n"
    "3) Opportunities and Risks\n"
    "4) What Changed Recently (newest signals)\n"
    "5) References (deduplicated URLs only; prefer reachable/recent sources and include academic sources when available)."
)


def start_query_node(state: ResearchState) -> dict:
    logger.info("graph.start_query | query=%r", state["query"])
    return {"messages": [HumanMessage(content=state["query"])], "sources": []}


def fan_out(state: ResearchState) -> list[Send]:
    logger.info("graph.fan_out | query=%r", state["query"])
    return [
        Send("academic_researcher", {"query": state["query"]}),
        Send("corporate_researcher", {"query": state["query"]}),
    ]


def summarize_node(state: ResearchState) -> dict:
    logger.info(
        "graph.summarize | messages=%s sources=%s",
        len(state["messages"]),
        len(state["sources"]),
    )
    response = _summarizer.invoke(
        [SystemMessage(content=_SUMMARY_INSTRUCTIONS), *state["messages"]]
    )
    deduped = _dedupe(state["sources"])
    logger.info("graph.summarize done | final_sources=%s", len(deduped))
    return {
        "final_report": response.content,
        "messages": [response],
        "sources": deduped,
    }


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("start_query", start_query_node)
    graph.add_node("academic_researcher", academic_researcher_node)
    graph.add_node("corporate_researcher", corporate_researcher_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("start_query")
    graph.add_conditional_edges("start_query", fan_out)
    graph.add_edge(["academic_researcher", "corporate_researcher"], "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


research_graph = build_research_graph()
