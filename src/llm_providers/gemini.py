from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path

API_PATH = Path(__file__).parents[2] / ".env" / ".env"

load_dotenv(dotenv_path=API_PATH)

GEMINI_FOUNDATION_MODEL = "gemini-3.5-flash"

def get_gemini_llm(model: str = GEMINI_FOUNDATION_MODEL) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model, temperature=0)