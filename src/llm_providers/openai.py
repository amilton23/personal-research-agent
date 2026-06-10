from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from pathlib import Path

API_PATH = Path(__file__).parents[2] / ".env" / ".env"

load_dotenv(dotenv_path=API_PATH)

OPENAI_FOUNDATION_MODEL = "gpt-4o"
OPENROUTER_FOUNDATION_MODEL = "openai/gpt-4o"

def get_openai_llm(model: str = OPENAI_FOUNDATION_MODEL) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0)

def get_openrouter_llm(model: str = OPENROUTER_FOUNDATION_MODEL) -> ChatOpenRouter:
    return ChatOpenRouter(model=model, temperature=0)