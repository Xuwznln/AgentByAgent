import json
import logging
from pathlib import Path

from crewai import Agent, Task, Crew, Process
from openai import OpenAI

from mcp_client import MCPClient

CONFIG_FILE = Path("config.json")


def load_config():
    if not CONFIG_FILE.exists():
        raise RuntimeError(f"Missing {CONFIG_FILE}. Please create it with API_KEY and BASE_URL.")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logging(log_file: str = "run.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )


def create_clients():
    cfg = load_config()
    openai_client = OpenAI(api_key=cfg["API_KEY"], base_url=cfg["BASE_URL"])
    mcp_client = MCPClient(cfg["API_KEY"], cfg["BASE_URL"])
    return openai_client, mcp_client


def run(query: str):
    setup_logging()
    openai_client, mcp_client = create_clients()

    # initial tool fetch
    tools = mcp_client.list_tools()
    logging.info("Initial tools: %s", tools)

    researcher = Agent(
        role="Researcher",
        goal="Search for libraries to answer the query",
        backstory="Expert at finding information online.",
        verbose=True,
    )

    coder = Agent(
        role="Coder",
        goal="Use libraries to implement solution",
        backstory="Writes concise Python code.",
        verbose=True,
    )

    tasks = [
        Task(
            description=f"Find libraries or APIs required to answer: {query}",
            expected_output="Library name and usage",
            agent=researcher,
        ),
        Task(
            description="Provide Python code using the library to solve the problem",
            expected_output="Runnable Python code",
            agent=coder,
        ),
    ]

    crew = Crew(
        agents=[researcher, coder],
        tasks=tasks,
        process=Process.sequential,
        openai_api_client=openai_client,
    )

    result = crew.kickoff()
    logging.info("Crew result: %s", result)

    # refresh tools after run
    mcp_client.refresh()
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mcp_crew_agent.py 'Your query'")
        raise SystemExit(1)

    print(run(sys.argv[1]))
