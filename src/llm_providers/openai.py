from langchain_openai import ChatOpenAI

OPENAI_FOUNDATION_MODEL = "gpt-4o"

def get_openai_llm(model: str = OPENAI_FOUNDATION_MODEL) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0)