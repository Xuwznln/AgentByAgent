import os
from openai import OpenAI
from crewai import Agent, Task, Crew, Process


def create_client():
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY environment variable is required")
    base_url = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def run_stock_price_example():
    client = create_client()

    researcher = Agent(
        role="Researcher",
        goal="Find Python libraries for obtaining stock prices",
        backstory="Expert at combing GitHub and documentation for useful packages.",
        verbose=True,
        allow_delegation=False,
    )

    coder = Agent(
        role="Coder",
        goal="Write a get_stock_price function using the chosen library",
        backstory="Python veteran who can quickly implement small utilities.",
        verbose=True,
        allow_delegation=False,
    )

    tasks = [
        Task(
            description="Search GitHub or the web for a Python library that can fetch stock prices.",
            expected_output="Name of a suitable library such as yfinance",
            agent=researcher,
        ),
        Task(
            description="Implement a function get_stock_price(ticker) using the suggested library.",
            expected_output="Python function source code",
            agent=coder,
        ),
    ]

    crew = Crew(
        agents=[researcher, coder],
        tasks=tasks,
        process=Process.sequential,
        openai_api_client=client,
    )

    return crew.kickoff()


if __name__ == "__main__":
    print(run_stock_price_example())
