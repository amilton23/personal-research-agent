from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

OPENROUTER_FOUNDATION_MODEL = "openai/gpt-4o"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def get_openrouter_llm(model: str = OPENROUTER_FOUNDATION_MODEL) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=0,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )