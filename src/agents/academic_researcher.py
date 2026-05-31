from src.llm_providers.gemini import get_gemini_llm
from langchain_core.messages import SystemMessage

_llm = get_gemini_llm()

instructions = (
    "You are an expert in searching academic papers, articles, and scholarly content about AI in healthcare. "
    "Always cite sources, authors, and publication dates. "
    "Focus on peer-reviewed journals, conferences (NeurIPS, MICCAI, AAAI), and preprint servers like arXiv."
)

def academic_researcher_node(state: dict) -> dict:
    messages = [
        SystemMessage(content=instructions),
        *state["messages"],
    ]
    response = _llm.invoke(messages)
    return {"messages": [response]}