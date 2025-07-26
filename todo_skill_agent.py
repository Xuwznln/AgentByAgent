import os
import json
from openai import OpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool,
)

class TodoList:
    def __init__(self, path="todo.json"):
        self.path = path
        self.items = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.items = json.load(f)
            except Exception:
                self.items = []

    def add(self, item):
        self.items.append(item)
        self._save()

    def __len__(self):
        return len(self.items)

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, indent=2)

class SkillStore:
    def __init__(self, path="skills.json"):
        self.path = path
        self.skills = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.skills = json.load(f)
            except Exception:
                self.skills = {}

    def add(self, name, code):
        self.skills[name] = code
        self._save()

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.skills, f, indent=2)

def create_client():
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY environment variable is required")
    base_url = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(api_key=api_key, base_url=base_url)

def run_blog_pipeline():
    client = create_client()

    todos = TodoList()
    skills = SkillStore()

    docs_tool = DirectoryReadTool(directory="./blog-posts")
    file_tool = FileReadTool()
    search_tool = SerperDevTool()
    web_rag_tool = WebsiteSearchTool()

    todo_agent = Agent(
        role="Todo Tracker",
        goal="Record every step in a todo list",
        backstory="Maintains an up-to-date list of tasks before execution.",
        verbose=True,
    )

    summarizer = Agent(
        role="Skill Aggregator",
        goal="Condense repeated workflows into reusable skills",
        backstory="Creates Python functions when enough steps are recorded.",
        verbose=True,
    )

    researcher = Agent(
        role="Market Research Analyst",
        goal="Provide up-to-date market analysis of the AI industry",
        backstory="An expert analyst with a keen eye for market trends.",
        tools=[search_tool, web_rag_tool],
        verbose=True,
    )

    writer = Agent(
        role="Content Writer",
        goal="Craft engaging blog posts about the AI industry",
        backstory="A skilled writer with a passion for technology.",
        tools=[docs_tool, file_tool],
        verbose=True,
    )

    tasks = [
        Task(
            description="Add 'research trends' to the todo list",
            expected_output="todo updated",
            agent=todo_agent,
        ),
        Task(
            description="Research the latest trends in the AI industry and provide a summary.",
            expected_output="Summary of top AI developments",
            agent=researcher,
        ),
        Task(
            description="Add 'write blog post' to the todo list",
            expected_output="todo updated",
            agent=todo_agent,
        ),
        Task(
            description="Write an engaging blog post about the AI industry, based on the research analyst's summary.",
            expected_output="Markdown blog post",
            agent=writer,
            output_file="blog-posts/new_post.md",
        ),
        Task(
            description="If more than 3 todo items exist, summarize them into a new skill and save to skills.json",
            expected_output="Updated skills.json",
            agent=summarizer,
        ),
    ]

    crew = Crew(
        agents=[todo_agent, summarizer, researcher, writer],
        tasks=tasks,
        process=Process.sequential,
        openai_api_client=client,
    )

    return crew.kickoff()

if __name__ == "__main__":
    result = run_blog_pipeline()
    print(result)
