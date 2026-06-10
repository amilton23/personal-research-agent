from src.llm_providers.gemini import get_gemini_llm
from langchain_core.messages import SystemMessage

_llm = get_gemini_llm()

instructions = (
    "You are an expert in searching corporate blogs, news articles, press releases, and industry reports about AI in healthcare. "
    "Focus on content from companies like Google Health, Microsoft Health, IBM Watson Health, and major hospital networks. "
    "Always cite the source, company, and publication date."
)

def corporate_researcher_node(state: dict) -> dict:
    messages = [
        SystemMessage(content=instructions),
        *state["messages"],
    ]
    response = _llm.invoke(messages)
    return {"messages": [response]}