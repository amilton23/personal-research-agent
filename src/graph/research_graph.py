from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing import Annotated, TypedDict
import operator

from src.agents.academic_researcher import academic_researcher_node
from src.agents.corporate_researcher import corporate_researcher_node
from src.llm_providers.openai import get_openai_llm

#######################
# CLASS
#######################

class ResearchState(TypedDict):
    query: str
    sources: list[str]
    summary: str
    final_report: str
    messages: Annotated[list[BaseMessage], operator.add]

#######################
# FUNCTIONS
#######################

def start_query_node(state: ResearchState) -> dict:
    return {"messages": [HumanMessage(content=state["query"])]}

_summarizer = get_openai_llm()
summarize_instructions = (
    "You are an expert in synthesizing academic and corporate research about AI in healthcare. "
    "Given the research results, generate a cohesive, structured final report with sections: "
    "Key Academic Findings, Industry Trends, Gaps and Opportunities."
)
def summarize_node(state: ResearchState) -> dict:
    messages = [
        SystemMessage(content=summarize_instructions),
        *state["messages"],
    ]
    response = _summarizer.invoke(messages)
    return {"final_report": response.content, "messages": [response]}

def evaluate_node(state: ResearchState) -> dict:
    """
    Evaluates if the research is sufficient.
    Populates 'summary' with a flag for needs_more_research to inspect.
    """
    last_content = state["final_report"].lower()
    is_sufficient = (
        len(state["sources"]) >= 4
        and len(state["final_report"]) > 200
        and "insufficient" not in last_content
    )
    return {"summary": "sufficient" if is_sufficient else "insufficient"}

def needs_more_research(state: ResearchState) -> bool:
    few_sources = len(state["sources"]) < 4
    empty_report = not state["final_report"].strip()
    last_msg_unsatisfactory = (
        state["messages"]
        and "insufficient" in state["messages"][-1].content.lower()
    )
    return few_sources or empty_report or last_msg_unsatisfactory

#######################
# GRAPH
#######################

def route_after_evaluate(state: ResearchState) -> str:
    if needs_more_research(state):
        return "start_query"
    return END

def build_research_graph():
    graph = StateGraph(ResearchState)

    ### NODES
    graph.add_node("start_query", start_query_node)
    graph.add_node("academic_researcher", academic_researcher_node)
    graph.add_node("corporate_researcher", corporate_researcher_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("evaluate", evaluate_node)

    def fan_out(state: ResearchState) -> list[Send]:
        return [
            Send("academic_researcher", state),
            Send("corporate_researcher", state),
        ]

    ### EDGES
    graph.set_entry_point("start_query")
    graph.add_conditional_edges("start_query", fan_out)  # fan-out paralelo
    graph.add_edge("academic_researcher", "summarize")
    graph.add_edge("corporate_researcher", "summarize")
    graph.add_edge("summarize", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluate)

    return graph.compile()

research_graph = build_research_graph()