import os
import re
import subprocess
import sys
import json
from urllib import request, parse

class ToolCache:
    def __init__(self, path=".tool_cache.json"):
        self.path = path
        self.tools = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.tools = json.load(f)
            except Exception:
                self.tools = {}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.tools, f)

    def get(self, name):
        return self.tools.get(name)

    def add(self, name, info):
        self.tools[name] = info
        self.save()


def search_github(query, max_results=5):
    url = "https://api.github.com/search/repositories?q=" + parse.quote(query)
    with request.urlopen(url) as resp:
        data = json.load(resp)
    items = data.get("items", [])[:max_results]
    return [{"name": i["full_name"], "url": i["html_url"]} for i in items]


def install_package(pkg_name):
    result = subprocess.run([sys.executable, "-m", "pip", "install", pkg_name], capture_output=True, text=True)
    return result.returncode == 0, result.stdout + "\n" + result.stderr


class StockPriceTool:
    def __init__(self):
        self.ready = False

    def prepare(self):
        try:
            import yfinance  # noqa: F401
            self.ready = True
        except ImportError:
            ok, out = install_package("yfinance")
            self.ready = ok
            if not ok:
                print("Failed to install yfinance:", out)

    def get_price(self, ticker):
        if not self.ready:
            self.prepare()
        if not self.ready:
            raise RuntimeError("yfinance not available")
        import yfinance as yf
        info = yf.Ticker(ticker)
        data = info.history(period="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
        return None


class Agent:
    def __init__(self):
        self.cache = ToolCache()

    def run(self, query):
        if "stock" in query.lower():
            ticker_match = re.search(r"(\b[A-Z]{1,5}\b)", query)
            ticker = ticker_match.group(1) if ticker_match else "NVDA"
            tool_info = self.cache.get("stock_price")
            if not tool_info:
                print("Searching for stock price library on GitHub...")
                results = search_github("python stock price")
                if results:
                    tool_info = {"name": results[0]["name"], "url": results[0]["url"], "pkg": "yfinance"}
                    self.cache.add("stock_price", tool_info)
            tool = StockPriceTool()
            price = tool.get_price(ticker)
            print(f"Latest price of {ticker} is {price}")
            return price
        else:
            print("I don't know how to handle this query yet.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py 'Your query'")
        sys.exit(1)
    agent = Agent()
    agent.run(sys.argv[1])
