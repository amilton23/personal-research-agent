import os
import sys

from src.graph.research_graph import research_graph
from src.utils.logs import get_logger

logger = get_logger(__name__)


def main() -> None:
    query = (
        " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI applications in healthcare"
    )

    logger.info("main.start | query=%r", query)
    logger.info(
        "main.env | tracing=%s tracing_v2=%s project=%s workspace=%s log_level=%s",
        os.getenv("LANGSMITH_TRACING"),
        os.getenv("LANGCHAIN_TRACING_V2"),
        os.getenv("LANGSMITH_PROJECT"),
        bool(
            os.getenv("LANGSMITH_WORKSPACE_ID") or os.getenv("LANGCHAIN_WORKSPACE_ID")
        ),
        os.getenv("LOG_LEVEL", "INFO"),
    )

    print(f"\nResearch query: {query}\n")

    result = research_graph.invoke(
        {
            "query": query,
            "sources": [],
            "messages": [],
            "final_report": "",
        }
    )

    logger.info(
        "main.done | report_chars=%s sources=%s",
        len(str(result.get("final_report", ""))),
        len(result.get("sources", [])),
    )

    print("Final Report:\n")
    print(result["final_report"])


if __name__ == "__main__":
    main()
