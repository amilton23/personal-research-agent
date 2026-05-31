from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_FOUNDATION_MODEL = "gemini-3.5-pro"

def get_gemini_llm(model: str = GEMINI_FOUNDATION_MODEL) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model, temperature=0)