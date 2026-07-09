from dotenv import load_dotenv
from pathlib import Path
import os
from src.llm_providers.gemini import get_gemini_llm

API_PATH = Path(__file__).parents[2] / ".env" / ".env"

if __name__ == "__main__":
    load_dotenv(dotenv_path=API_PATH)

    print(API_PATH)  # mostra o path completo
    print(API_PATH.exists())  # True ou False?

    print("=== GOOGLE_API_KEY ===")
    print(os.environ.get("GOOGLE_API_KEY"))
    print("======================")

    llm = get_gemini_llm()
    resp = llm.invoke("Only answer: It works!")
    print(resp.content)
