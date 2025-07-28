#!/usr/bin/env python3
"""
Dynamic MCP Server - Dynamic tools folder monitoring MCP server

This server will:
- Monitor Python files in the tools folder
- Dynamically import and register non-underscore-prefixed functions as tools
- Detect tool changes and record differences
- Provide SSE interface listening on 0.0.0.0:3001
- Smart caching: Only reload the currently called tool during tool calls (not global reload)
"""

import asyncio
import json
import re
import traceback
import openai
from typing import Dict, Any, List, Optional
from urllib import request, parse
from datetime import datetime
from pathlib import Path
import hashlib

from fastmcp.server.proxy import FastMCPProxy, ProxyTool

# Apply JSON monkey patch to fix pydantic serialization issues
from tools.json_patch import apply_json_patch
apply_json_patch()

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.timing import DetailedTimingMiddleware
from fastmcp.server.middleware.logging import StructuredLoggingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.tools import FunctionTool

# Import new environment management and proxy modules
from tools.tool_env_manager import ToolEnvironmentManager
from tools.tool_proxy import ToolProxyManager, ToolProxy

# Import enhanced logging system
from tools.logger_config import dynamic_logger, console

# ================================
# Configuration
# ================================

SERVER_NAME = "DynamicToolsServer"
SERVER_VERSION = "1.0.0"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 3002
TOOLS_DIR = "tools"

# Configure enhanced logging system
logger = dynamic_logger.get_logger("dynamic-mcp-server")

# Initialize FastMCP server
mcp = FastMCP(SERVER_NAME, version=SERVER_VERSION)
config = json.load(open("config.json", "r", encoding="utf-8"))

# ================================
# Tool Change Manager
# ================================

class ToolChangeManager:
    """Manage tool change detection and recording"""
    
    def __init__(self):
        self.previous_tools: Dict[str, Dict[str, Any]] = {}
        self.current_tools: Dict[str, Dict[str, Any]] = {}
        self.change_history: List[Dict[str, Any]] = []
        self.file_hashes: Dict[str, str] = {}
    
    def get_file_hash(self, filepath: str) -> str:
        """Get MD5 hash of a file"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def update_tools(self, existing_tools_desc: Dict[str, Dict[str, Any]], new_tools_desc: Dict[str, Dict[str, Any]]):
        """Update tool list and detect changes"""
        self.previous_tools = existing_tools_desc.copy()
        self.current_tools = new_tools_desc.copy()
        
        # Detect changes
        changes = self.detect_changes()
        if changes and (len(changes["added"]) or len(changes["modified"]) or len(changes["removed"])):
            change_record = {
                "timestamp": datetime.now().isoformat(),
                "changes": changes
            }
            self.change_history.append(change_record)

        return changes
    
    def detect_changes(self) -> Dict[str, Any]:
        """Detect tool changes with detailed value comparison"""
        changes = {
            "added": [],
            "removed": [],
            "modified": []
        }
        
        # Detect newly added tools
        for tool_name in self.current_tools:
            if tool_name not in self.previous_tools:
                changes["added"].append({
                    "name": tool_name,
                    "details": self.current_tools[tool_name]
                })
        
        # Detect removed tools
        for tool_name in self.previous_tools:
            if tool_name not in self.current_tools:
                changes["removed"].append({
                    "name": tool_name,
                    "details": self.previous_tools[tool_name]
                })
        
        # Detect modified tools
        for tool_name in self.current_tools:
            if tool_name in self.previous_tools:
                if self.current_tools[tool_name] != self.previous_tools[tool_name]:
                    # Detailed difference comparison
                    diff_details = self._get_detailed_diff(
                        self.previous_tools[tool_name], 
                        self.current_tools[tool_name]
                    )
                    changes["modified"].append({
                        "name": tool_name,
                        "previous": self.previous_tools[tool_name],
                        "current": self.current_tools[tool_name],
                        "differences": diff_details
                    })
        
        return changes
    
    def _get_detailed_diff(self, old_desc: Dict[str, Any], new_desc: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed differences between two tool descriptions"""
        differences = {}
        
        # Check all key changes
        all_keys = set(old_desc.keys()) | set(new_desc.keys())
        
        for key in all_keys:
            old_value = old_desc.get(key)
            new_value = new_desc.get(key)
            
            if old_value != new_value:
                differences[key] = {
                    "old": old_value,
                    "new": new_value
                }
        
        return differences
    
    def get_change_summary(self) -> Dict[str, Any]:
        """Get change summary"""
        return {
            "current_tools_count": len(self.current_tools),
            "previous_tools_count": len(self.previous_tools),
            "recent_changes": self.change_history[-2:] if self.change_history else [],
            "tool_details": {
                "current": list(self.current_tools.keys()),
                "previous": list(self.previous_tools.keys())
            }
        }

# Global change manager
change_manager = ToolChangeManager()

# ================================
# Dynamic Tool Loader
# ================================

class DynamicToolLoader:
    """Dynamic tool loader - supports isolated environments"""
    
    def __init__(self, tools_dir: str):
        self.tools_dir = Path(tools_dir)
        self.loaded_modules: Dict[str, Any] = {}
        self.current_tools: Dict[str, Any] = {}
        
        # Initialize environment manager and proxy manager
        self.env_manager = ToolEnvironmentManager(tools_dir)
        self.proxy_manager = ToolProxyManager()
        
        # Ensure tools directory exists
        self.tools_dir.mkdir(exist_ok=True)
    
    def scan_and_load_tools(self, request_tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Scan and load tools from the tools directory (using isolated environments)"""
        new_tools = {}
        # Use environment manager to load all tools
        load_result = self.env_manager.load_all_tools(request_tool_name.split(".")[0] if request_tool_name is not None else None)
        
        if "tools" in load_result:
            # load_result["tools"] is dictionary format {tool_name: tool_data}
            for tool_name, tool_data in load_result["tools"].items():
                # Get tool directory
                tool_dir_name = tool_name.split(".")[0]
                tool_dir = self.tools_dir / tool_dir_name
                
                if tool_dir.exists():
                    try:
                        # Get Python executable file
                        python_exe = self.env_manager.get_python_executable(tool_dir)
                        
                        # Create tool proxy
                        proxy = self.proxy_manager.create_proxy(tool_data, tool_dir, python_exe)
                        
                        # Store tool data and proxy
                        new_tools[tool_name] = {
                            "tool_data": tool_data,
                            "proxy": proxy,
                            "tool_dir": tool_dir,
                            "python_exe": python_exe
                        }
                        
                        logger.debug(f"Loaded tool via proxy: {tool_name}")
                        
                    except Exception as e:
                        logger.error(f"Error creating proxy for tool {tool_name}: {e}")
        
        # Record loading errors
        if "errors" in load_result and load_result["errors"]:
            for error_info in load_result["errors"]:
                logger.error(f"Tool loading error in {error_info['tool_dir']}: {error_info['error']}")
        
        logger.info(f"Loaded {len(new_tools)} tools via environment isolation")
        return new_tools

    def register_tools_to_mcp(self, tools: Dict[str, Any], request_tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Register tools to MCP server (using FunctionTool data and proxy)"""
        existing_tools: Dict[str, FunctionTool] = mcp._tool_manager._tools  # type: ignore
        existing_tools_desc = {k: v.model_dump() for k, v in existing_tools.items() if "." in k}
        for v in existing_tools_desc.values():
            del v["fn"]
        request_tool_dir = request_tool_name.split(".")[0] if request_tool_name is not None else self.tools_dir.name
        # Register new tools
        new_tools_desc = {}
        for tool_name, tool_info in tools.items():
            try:
                tool_data = tool_info["tool_data"]
                proxy: ToolProxy = tool_info["proxy"]
                tool_data["fn"] = proxy.__call__
                
                # Reconstruct FunctionTool object
                function_tool = FunctionTool.model_validate({k: v for k, v in tool_data.items() if k not in [
                    "source_module", "function_name", "tool_name_prefix"
                ]})
                # Add directly to MCP server
                if tool_name in existing_tools:
                    mcp.remove_tool(tool_name)
                mcp.add_tool(function_tool)
                # Record tool description (for change detection)
                tool_desc = function_tool.model_dump()
                del tool_desc["fn"]
                new_tools_desc[tool_name] = tool_desc
                logger.info(f"Registered proxied tool: {tool_name}")
                
            except Exception as e:
                logger.error(f"Error registering tool {tool_name}: {e.args}\n{traceback.format_exc()}")
        
        # Remove tools that no longer exist
        for tool_name in existing_tools_desc:
            if tool_name not in tools and "." in tool_name:
                # For on tool call cases, don't remove custom tools
                if request_tool_dir is not None and not tool_name.startswith(f"{request_tool_dir}."):
                    continue
                try:
                    mcp.remove_tool(tool_name)
                    logger.info(f"Removed tool: {tool_name}")
                except Exception as e:
                    logger.error(f"Error removing tool {tool_name}: {e}")
        
        # Detect and record changes
        changes = change_manager.update_tools(existing_tools_desc, new_tools_desc)
        return changes

# Global tool loader
tool_loader = DynamicToolLoader(TOOLS_DIR)

# ================================
# Middleware
# ================================

class DynamicToolMiddleware(Middleware):
    """Dynamic tool middleware that refreshes tools on every tool call and list tools"""

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Refresh current tool when calling tool"""
        tool_name = context.message.name
        if "." in tool_name:
            logger.info(f"Refreshing specific tool before calling: {tool_name}")
            # Only reload the currently called tool
            reloaded_tools = tool_loader.scan_and_load_tools(tool_name)
            if reloaded_tools:
                # Re-register the tool to MCP
                register_result = tool_loader.register_tools_to_mcp(reloaded_tools, tool_name)
            else:
                logger.warning(f"No tools were reloaded for {tool_name}")
        
        # Continue executing tool call
        return await call_next(context)
    
    async def on_list_tools(self, context: MiddlewareContext, call_next):
        """Refresh tools directory when listing tools"""
        logger.info("Refreshing all tools before listing tools")
        # Re-scan and load tools
        tools = tool_loader.scan_and_load_tools()
        tool_loader.register_tools_to_mcp(tools)
        # Continue executing list tools
        return await call_next(context)

# ================================
# Add Middleware
# ================================

mcp.add_middleware(ErrorHandlingMiddleware(include_traceback=True))
mcp.add_middleware(DynamicToolMiddleware())
mcp.add_middleware(DetailedTimingMiddleware())
mcp.add_middleware(StructuredLoggingMiddleware(include_payloads=True))

# ================================
# Built-in Tools
# ================================

@mcp.tool
def search_github(query: str, max_results: int = 10, sort_by: str = "stars") -> List[Dict[str, Any]]:
    """
    Search GitHub Python language repositories and sort by specified criteria
    
    Args:
        query: Search keywords
        max_results: Maximum number of results to return
        sort_by: Sort method, options: "stars", "forks", "updated"
    
    Returns:
        Repository list sorted by specified criteria, including star and fork counts
    
    Examples:
        search_github("machine learning")  # Search Python machine learning projects
    """
    if not query:
        raise Exception("query is empty")
    
    # Build search query string
    search_query = query
    search_query += f" language:Python"
    
    # GitHub search API supports sort parameters
    url = f"https://api.github.com/search/repositories?q={parse.quote(search_query)}&sort={sort_by}&order=desc"
    logger.info(url)
    try:
        with request.urlopen(url) as resp:
            data = json.load(resp)
        
        items = data.get("items", [])[:max_results]
        
        # Extract key information and format
        results = []
        for i, item in enumerate(items, 1):
            repo_info = {
                "rank": i,
                "name": item["full_name"],
                "url": item["html_url"],
                "description": item.get("description", "No description"),
                "stars": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
                "language": item.get("language", "Unknown"),
                "updated_at": item.get("updated_at", ""),
                "topics": item.get("topics", [])
            }
            results.append(repo_info)
        
        logger.info(f"GitHub search '{query}' returned {len(results)} results, sorted by {sort_by}")
        return results
        
    except Exception as e:
        logger.error(f"GitHub search failed: {e}")
        return [{"error": f"Search failed: {str(e)}"}]


# noinspection PyTypeChecker
@mcp.tool
def advanced_web_search(query: str) -> str:
    """
    Use AI-enhanced web search functionality to provide smarter search results with citations. You need to combine with other tools to actively verify correctness
    
    Args:
        query: Search query, supports both Chinese and English, can be questions or keywords
    
    Returns:
        AI-analyzed and organized search results
    
    Examples:
        advanced_web_search("Python async programming best practices")
        advanced_web_search("What is the latest news about AI development?")
    """
    try:
        # Configure OpenAI client with configuration values
        openai_config = config.get("openai", {})
        api_key = openai_config.get("api_key", "")
        base_url = openai_config.get("base_url", "https://api.openai.com/v1/")
        model = openai_config.get("model", "gpt-4o")
        if not api_key:
            return "❌ Configuration error: OpenAI API key not set, please check config.json file"
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        logger.info(f"Executing advanced web search for query: '{query}'")
        # Execute search with AI enhancement
        response_with_search = client.responses.create(
            model=model,
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "medium",
            }],
            input=query,
            temperature=0.1
        )
        search_result = response_with_search.output_text
        
        # Process URL citations
        annotations = []
        citations_text = ""
        
        try:
            # Use dynamic access to avoid type checking issues
            output = getattr(response_with_search, 'output', None)
            if output and len(output) > 1:
                output_item = output[1]
                content = getattr(output_item, 'content', None)
                if content and len(content) > 0:
                    content_item = content[0]
                    annotations = getattr(content_item, 'annotations', [])
            
            logger.info(f"📎 Found {len(annotations)} URL citations")
            
            # Format citation information
            if annotations:
                citations_text = "\n\n**Reference Sources:**\n"
                for i, annotation in enumerate(annotations, 1):
                    try:
                        # Clean URL, remove utm_source parameters
                        title = getattr(annotation, 'title', 'Unknown Title')
                        url = getattr(annotation, 'url', '#')
                        clean_url = url.split('?utm_source=')[0] if '?utm_source=' in url else url
                        citations_text += f"{i}. [{title}]({clean_url})\n"
                        logger.info(f"📖 Citation {i}: {title} -> {clean_url}")
                    except Exception as citation_error:
                        logger.warning(f"⚠️ Failed to process citation {i}: {citation_error}")
                        
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract citations: {e}")
            logger.debug(f"🔍 Response type: {type(response_with_search)}")
            citations_text = ""
        
        # Merge search results and citation information
        final_result = search_result + citations_text
        
        logger.info(f"🎉 Advanced web search completed successfully for query: '{query}'")
        logger.info(f"📊 Result: {len(search_result)} chars + {len(citations_text)} chars citations")
        
        # Return formatted result with citations
        return final_result
        
    except Exception as e:
        logger.error(f"❌ OpenAI API call failed: {str(e)}")
        logger.error(f"🔍 Error details: {traceback.format_exc()}")
        return f"❌ Search error: {str(e)}"

@mcp.tool
def get_tools_changes() -> Dict[str, Any]:
    """
    Get tool change information, comparing current and previous versions
    
    Returns:
        Dictionary containing tool change details
    """
    return change_manager.get_change_summary()

@mcp.tool
def refresh_tools() -> Dict[str, Any]:
    """
    Manually refresh tools in the tools directory
    
    Returns:
        Result of the refresh operation
    """
    try:
        # Re-scan and load tools
        tools = tool_loader.scan_and_load_tools()
        tool_loader.register_tools_to_mcp(tools)
        
        changes = change_manager.get_change_summary()
        
        return {
            "status": "success",
            "message": "Tools refreshed successfully",
            "loaded_tools": list(tools.keys()),
            "changes": changes
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error refreshing tools: {str(e)}"
        }

@mcp.tool
def get_server_status() -> Dict[str, Any]:
    """
    Get server status information
    
    Returns:
        Server status information
    """
    return {
        "server_name": SERVER_NAME,
        "version": SERVER_VERSION,
        "host": SERVER_HOST,
        "port": SERVER_PORT,
        "tools_directory": str(Path(TOOLS_DIR).absolute()),
        "uptime": "running",
        "loaded_modules": list(tool_loader.loaded_modules.keys()),
        "current_tools": list(tool_loader.current_tools.keys()),
        "change_history_count": len(change_manager.change_history),
        "environment_mode": "isolated",
        "proxy_tools_count": len(tool_loader.proxy_manager.list_proxies())
    }

@mcp.tool
def create_tool_environment(tool_name: str, requirements: Optional[List[str]] = None, template_content: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new tool environment including directory, virtual environment, requirements.txt and basic tool.py file
    
    Args:
        tool_name: Tool name (can only contain letters, numbers, underscores)
        requirements: Dependency package list, e.g. ["fastmcp", "requests>=2.25.0", "pandas"]
        template_content: Template content for tool.py, uses default template if empty
        
    Returns:
        Result and detailed information of the creation operation
        
    Examples:
        create_tool_environment("my_calculator", ["fastmcp", "numpy"])
        create_tool_environment("web_scraper", ["requests", "beautifulsoup4", "fastmcp"])
    """
    # Validate tool name
    if not tool_name or not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', tool_name):
        return {
            "status": "error",
            "message": "Tool name can only contain letters, numbers, underscores, and must start with a letter"
        }
    
    try:
        tool_dir = Path(TOOLS_DIR) / tool_name
        
        # Check if directory already exists
        if tool_dir.exists():
            return {
                "status": "error",
                "message": f"Tool directory {tool_name} already exists"
            }
        
        # Create tool directory
        tool_dir.mkdir(parents=True, exist_ok=True)
        
        # Create requirements.txt
        requirements_content = []
        if requirements:
            # Validate and clean dependency format
            for req in requirements:
                req = req.strip()
                if req and not req.startswith("#"):
                    requirements_content.append(req)
        
        # Add fastmcp dependency by default
        if "fastmcp" not in str(requirements_content):
            requirements_content.insert(0, "fastmcp")
        
        requirements_file = tool_dir / "requirements.txt"
        with open(requirements_file, 'w', encoding='utf-8') as f:
            f.write("# Tool dependencies\n")
            for req in requirements_content:
                f.write(f"{req}\n")
        
        # Create default tool.py file
        if not template_content:
            template_content = f'''"""
{tool_name.title()} Tool - {tool_name} tool

This is an auto-generated tool template.
Please edit this file to implement your tool functionality.
"""

# Add more tool functions here...
'''
        tool_file = tool_dir / "tool.py"
        with open(tool_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        # Use environment manager to create virtual environment
        try:
            venv_path = tool_loader.env_manager.ensure_virtual_environment(tool_dir)
            venv_created = True
        except Exception as e:
            logger.warning(f"Virtual environment creation failed: {e}")
            venv_created = False
        
        # Try to install dependencies
        deps_installed = False
        if venv_created:
            try:
                deps_installed = tool_loader.env_manager.install_requirements(tool_dir)
            except Exception as e:
                logger.warning(f"Dependency installation failed: {e}")

        # noinspection PyUnboundLocalVariable
        return {
            "status": "success",
            "message": f"Tool environment {tool_name} created successfully",
            "details": {
                "tool_name": tool_name,
                "tool_directory": str(tool_dir.absolute()),
                "requirements_file": str(requirements_file.absolute()),
                "tool_file": str(tool_file.absolute()),
                "virtual_environment": str(venv_path.absolute()) if venv_created else None,
                "dependencies": requirements_content,
                "venv_created": venv_created,
                "dependencies_installed": deps_installed
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create tool environment: {str(e)}"
        }

# ================================
# Resources
# ================================

@mcp.resource("config://server")
def get_server_config() -> dict:
    """Get server configuration information"""
    config = {
        "server": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
            "host": SERVER_HOST,
            "port": SERVER_PORT
        },
        "tools": {
            "directory": str(Path(TOOLS_DIR).absolute()),
            "loaded_modules": list(tool_loader.loaded_modules.keys()),
            "current_tools": list(tool_loader.current_tools.keys())
        },
        "features": [
            "dynamic_tool_loading",
            "change_detection",
            "file_monitoring",
            "sse_transport"
        ]
    }
    return config

# ================================
# Startup Function
# ================================

def main():
    """Start the dynamic MCP server"""
    dynamic_logger.print_banner(
        "Dynamic MCP Server - Dynamic Tool Server",
        f"v{SERVER_VERSION} | {SERVER_HOST}:{SERVER_PORT} | SSE Protocol"
    )
    console.print()
    dynamic_logger.print_section("Server Configuration", [
        f"Server Name: [bold cyan]{SERVER_NAME}[/bold cyan]",
        f"Version: [bold green]{SERVER_VERSION}[/bold green]", 
        f"Listen Address: [bold yellow]{SERVER_HOST}:{SERVER_PORT}[/bold yellow]",
        f"Transport Protocol: [bold magenta]SSE[/bold magenta]",
        f"Tools Directory: [bold blue]{Path(TOOLS_DIR).absolute()}[/bold blue]"
    ], "cyan")
    console.print()
    # ================================
    # Mirror Remote MCP Server Tools
    # ================================
    try:
        proxy: FastMCPProxy = FastMCP.as_proxy("http://127.0.0.1:8931/sse/")
        remote_tools = asyncio.run(proxy.get_tools())
        tool_info: ProxyTool
        for tool_name, tool_info in remote_tools.items(): # type: ignore
            try:
                # Create local copy
                local_tool = tool_info.copy()
                # Add to local server
                mcp.add_tool(local_tool)
                logger.info(f"Mirrored tool from remote server: {tool_info.name}")
            except Exception as e:
                logger.error(f"Failed to mirror tool {tool_info.name}: {e}")
    except Exception as e:
        dynamic_logger.warning(f"Unable to connect to remote MCP server: {e}")
        dynamic_logger.info("Continuing startup with local tools only...")
    # ================================
    # Load Local Tools
    # ================================
    dynamic_logger.info("Loading local tools...")
    tools = tool_loader.scan_and_load_tools()
    tool_loader.register_tools_to_mcp(tools)
    dynamic_logger.success(f"Loaded {len(tools)} local tools")
    
    dynamic_logger.print_status("Startup", "Server is starting...", True)
    console.print()
    # ================================
    # Load Code Tools
    # ================================
    from mcp_claude_code.server import ClaudeCodeServer
    ClaudeCodeServer(mcp_instance=mcp, allowed_paths=["/home/wz/AgentByAgent/tools"], enable_agent_tool=False)
    # Start server
    mcp.run(transport="http", host=SERVER_HOST, port=SERVER_PORT)

if __name__ == "__main__":
    main()  # Execute synchronously, async handling is done internally by mcp.run