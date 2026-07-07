import sys

from src.graph.research_graph import research_graph


def main() -> None:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI applications in healthcare"

    print(f"\nResearch query: {query}\n")

    result = research_graph.invoke(
        {
            "query": query,
            "sources": [],
            "messages": [],
            "final_report": "",
        }
    )

    print("Final Report:\n")
    print(result["final_report"])


if __name__ == "__main__":
    main()
