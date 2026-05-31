import sys
from src.graph.research_graph import research_graph, ResearchState


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI applications in healthcare 2026"

    print(f"\nResearch query: {query}\n")

    initial_state = ResearchState(query=query)
    result = research_graph.invoke(initial_state)

    print("\nFinal Report:\n")
    print(result["final_report"])


if __name__ == "__main__":
    main()