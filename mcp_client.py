import requests
import logging

class MCPClient:
    """Simple client for interacting with the MCP server."""

    def __init__(self, api_key: str, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def list_tools(self):
        """Return the list of available tools."""
        url = f"{self.base_url}/tools"
        logging.info("Fetching tool list from %s", url)
        resp = self.session.get(url)
        resp.raise_for_status()
        tools = resp.json()
        logging.info("Available tools: %s", tools)
        return tools

    def refresh(self):
        """Trigger a refresh of the tool list."""
        url = f"{self.base_url}/refresh"
        logging.info("Refreshing tools via %s", url)
        resp = self.session.post(url)
        resp.raise_for_status()
        data = resp.json()
        logging.info("Refreshed tools: %s", data)
        return data
