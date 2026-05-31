# Multi-Agent Solution for AI in Healthcare Research

This project is a multi-agent solution for gathering corporative and academic research on AI for healthcare. It uses LangGraph to orchestrate the agents and leverages two LLM providers: OpenAI and Gemini.

## Project Structure

```
personal-research-agent/
├── .env/
│   └── .env
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   └── 01_initial_research.ipynb
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── academic_researcher.py
│   │   └── corporate_researcher.py
│   ├── graph/
│   │   ├── __init__.py
│   │   └── research_graph.py
│   ├── llm_providers/
│   │   ├── __init__.py
│   │   ├── gemini.py
│   │   └── openai.py
│   ├── tools/
│   │   ├── __init__.py
│   │   └── web_search.py
│   └── utils/
│       ├── __init__.py
│       └── file_utils.py
├── tests/
│   ├── __init__.py
│   ├── agents/
│   │   └── test_academic_researcher.py
│   └── tools/
│       └── test_web_search.py
└── README.md
```

## Development Plan

### Phase 1: Setup and Configuration

1.  **Project Initialization**: Set up the project structure and initialize a Git repository.
2.  **LLM Provider Integration**:
    *   Implement the `OpenAI` and `Gemini` connectors in the `src/llm_providers/` directory.
    *   Create a configuration file `configs/config.yaml` to store API keys and other settings. Add `configs/config.yaml` to `.gitignore`.
3.  **Basic Tooling**:
    *   Implement a basic web search tool in `src/tools/web_search.py`. This tool will be used by the agents to search for information on the web.

### Phase 2: Agent Development

1.  **Academic Research Agent**:
    *   Develop the `academic_researcher` agent in `src/agents/academic_researcher.py`.
    *   This agent will be responsible for searching for academic papers, articles, and other scholarly content.
    *   It will use the web search tool to find relevant information and will be able to extract key information from the search results.
2.  **Corporate Research Agent**:
    *   Develop the `corporate_researcher` agent in `src/agents/corporate_researcher.py`.
    *   This agent will be responsible for searching for corporate blogs, news articles, and other industry-related content.
    *   It will use the web search tool to find relevant information and will be able to extract key information from the search results.

### Phase 3: Graph Implementation

1.  **Research Graph**:
    *   Implement the `research_graph` in `src/graph/research_graph.py`.
    *   This graph will define the workflow of the multi-agent system.
    *   It will start with a user query, then the academic and corporate research agents will work in parallel to gather information.
    *   The graph will then have a step to synthesize the information from both agents and generate a final report.

### Phase 4: Testing and Evaluation

1.  **Unit Tests**:
    *   Write unit tests for the tools and agents in the `tests/` directory.
2.  **Integration Tests**:
    *   Write integration tests for the research graph to ensure that the agents are working together as expected.
3.  **Evaluation**:
    *   Evaluate the performance of the multi-agent system on a set of test queries.
    *   The evaluation will focus on the quality of the generated reports and the efficiency of the system.

### Phase 5: Deployment and Production

1.  **API**:
    *   Expose the multi-agent system as an API using a web framework like FastAPI.
2.  **Deployment**:
    *   Deploy the multi-agent system to a cloud provider like AWS or GCP.
3.  **Monitoring**:
    *   Set up monitoring and logging to track the performance of the system in production.

## Getting Started

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/personal-research-agent.git
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up the configuration file:
    ```bash
    cp .env/.env.example .env/.env
    ```
4.  Add your API keys to `.env/.env`.
5.  Run the multi-agent system:
    ```bash
    python main.py "your research query"
    ```
