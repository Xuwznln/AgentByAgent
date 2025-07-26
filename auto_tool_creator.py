import os
import sys
from openai import OpenAI
from crewai import Agent, Task, Crew, Process


def create_client():
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY environment variable is required")
    base_url = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def run_auto_tool(query: str):
    client = create_client()

    planner = Agent(
        role="Planner",
        goal="Decide what tool is needed to fulfill the user's query",
        backstory="Understands requirements and defines new tools to build.",
        verbose=True,
    )

    researcher = Agent(
        role="Researcher",
        goal="Find Python packages or APIs that help implement the tool",
        backstory="Expert at searching GitHub and reading documentation.",
        verbose=True,
    )

    coder = Agent(
        role="Coder",
        goal="Implement the tool as Python code",
        backstory="Skilled Python developer producing concise utilities.",
        verbose=True,
    )

    tasks = [
        Task(
            description=f"Analyze the user query `{query}` and describe the tool to create.",
            expected_output="Clear description of the required tool",
            agent=planner,
        ),
        Task(
            description="Search for an appropriate Python library to build the tool and explain how to use it.",
            expected_output="Name of a suitable library and installation command",
            agent=researcher,
        ),
        Task(
            description="Provide Python source code implementing the tool using the chosen library.",
            expected_output="Python function code",
            agent=coder,
        ),
    ]

    crew = Crew(
        agents=[planner, researcher, coder],
        tasks=tasks,
        process=Process.sequential,
        openai_api_client=client,
    )

    return crew.kickoff()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_tool_creator.py 'Your query'")
        sys.exit(1)
    result = run_auto_tool(sys.argv[1])
    print(result)
