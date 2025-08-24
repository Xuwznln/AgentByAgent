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
import subprocess

from fastmcp.server.proxy import ProxyTool

# Apply JSON monkey patch to fix pydantic serialization issues
from tools.json_patch import apply_json_patch
apply_json_patch()

from fastmcp import Client, FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.timing import DetailedTimingMiddleware
from fastmcp.server.middleware.logging import StructuredLoggingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.tools import FunctionTool
from fastmcp.tools.tool import ToolResult

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
    
    async def scan_and_load_tools(self, request_tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Scan and load tools from the tools directory (using isolated environments)"""
        new_tools = {}
        # Use environment manager to load all tools
        load_result = await self.env_manager.load_all_tools(request_tool_name.split("-")[0] if request_tool_name is not None else None)
        
        if "tools" in load_result:
            # load_result["tools"] is dictionary format {tool_name: tool_data}
            for tool_name, tool_data in load_result["tools"].items():
                # Get tool directory
                tool_dir_name = tool_name.split("-")[0]
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
        existing_tools_desc = {k: v.model_dump() for k, v in existing_tools.items() if "-" in k}
        for v in existing_tools_desc.values():
            del v["fn"]
        request_tool_dir = request_tool_name.split("-")[0] if request_tool_name is not None else self.tools_dir.name
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
            if tool_name not in tools and "-" in tool_name:
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
    browser_mcp_client: Optional[Client] = None
    async def init_client(self):
        if self.browser_mcp_client is None:
            logger.info("Initializing browser MCP client")
            self.browser_mcp_client = Client('http://localhost:8931/mcp')
            await self.browser_mcp_client._connect()


    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Refresh current tool when calling tool"""
        tool_name = context.message.name
        start_time = datetime.now()

        if "-" in tool_name:
            logger.info(f"Refreshing specific tool before calling: {tool_name}")
            # Only reload the currently called tool
            reloaded_tools = await tool_loader.scan_and_load_tools(tool_name)
            if reloaded_tools:
                # Re-register the tool to MCP
                register_result = tool_loader.register_tools_to_mcp(reloaded_tools, tool_name)
            else:
                logger.warning(f"No tools were reloaded for {tool_name}")
        
        # Execute tool
        logger.warning(f"🚀 Execute: {tool_name} Arguments: {getattr(context.message, 'arguments', {})}")
        if tool_name.startswith("browser_"):
            await self.init_client()
            assert self.browser_mcp_client is not None
            result = await self.browser_mcp_client.call_tool(tool_name, getattr(context.message, 'arguments', {}))
            result = ToolResult(result.content, result.structured_content)
        else:
            result = await call_next(context)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        # Log successful execution
        logger.info(f"✅ Complete: {tool_name} {execution_time:.3f} seconds")
        # Log result summary
        if hasattr(result, 'content') and result.content and tool_name not in ["tool_environment_current_functions"]:
            try:
                content_item = result.content[0]
                content_preview = getattr(content_item, 'text', f"[{type(content_item).__name__}]")
                logger.info(f"📋 Result {type(result).__name__}({len(result.content)}): {content_preview}")
            except (IndexError, AttributeError):
                logger.info(f"📋 Result {type(result).__name__}({len(result.content)}): [content not accessible]")
        else:
            logger.info(f"📋 Result {type(result).__name__}: {vars(result)}")
        return result

    
    async def on_list_tools(self, context: MiddlewareContext, call_next):
        """Refresh tools directory when listing tools"""
        logger.info("Refreshing all tools before listing tools")
        # Re-scan and load tools
        tools = await tool_loader.scan_and_load_tools()
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
# @mcp.tool
def llm_web_search(query: str) -> str:
    """
    Use AI-enhanced web search functionality to provide smarter search results with citations. You need to combine with other tools to actively verify correctness
    
    Args:
        query: Search query, can be questions or keywords
    
    Returns:
        AI-analyzed and organized search results
    
    Examples:
        llm_web_search("Python async programming best practices")
        llm_web_search("What is the latest news about AI development?")
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
            input=f"""You will receive a search request.
**Do not add, infer, or guess any facts—use only the text in those snippets.**
**Avoid any info sources from huggingface and other AI-related datasets.**

Search query: {query}""",
            temperature=0
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
async def refresh_tools() -> Dict[str, Any]:
    """
    Manually refresh tools in the tools directory
    
    Returns:
        Result of the refresh operation
    """
    try:
        # Re-scan and load tools
        tools = await tool_loader.scan_and_load_tools()
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
async def tool_environment_create(environment_name: str, pip_requirements: Optional[List[str]] = None, tool_py_file_content: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new isolated tool environment with directory structure, virtual environment, dependencies, and tool implementation
    
    Args:
        environment_name: Environment name (alphanumeric and underscores only, must start with letter)
        pip_requirements: List of Python packages to install, e.g. ["requests>=2.25.0", "pandas>=1.3.0"]
        tool_py_file_content: Custom Python code for tool.py. If not provided, creates a basic template.
                             Functions not starting with "_" will be automatically exported as MCP tools.
                             Example:
                             ```python
                             import numpy as np
                             
                             def _helper_function():  # Private function - not exported
                                 return np.array([1, 2, 3])
                             
                             def calculate_sum():  # Public function - will be exported as MCP tool
                                 result = _helper_function()
                                 return result.sum().item()  # Return JSON-serializable type
                             ```

    Returns:
        Dict containing creation status, environment details, and installation results
        
    Examples:
        tool_environment_create("my_calculator", ["numpy>=1.21.0"])
        tool_environment_create("web_scraper", ["requests", "beautifulsoup4", "lxml"])
        tool_environment_create("data_processor", ["pandas", "openpyxl"], custom_tool_code)
    """
    # Validate environment name
    if not environment_name or not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', environment_name):
        return {
            "status": "error",
            "message": "Environment name must start with a letter and contain only letters, numbers, and underscores"
        }
    
    try:
        tool_dir = Path(TOOLS_DIR) / environment_name
        
        # Check if directory already exists
        if tool_dir.exists():
            return {
                "status": "error",
                "message": f"Tool directory '{environment_name}' already exists"
            }
        
        # Create tool directory
        tool_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare requirements list
        requirements_content = []
        if pip_requirements:
            # Validate and clean dependency format
            for req in pip_requirements:
                req = req.strip()
                if req and not req.startswith("#"):
                    requirements_content.append(req)

        # Create requirements.txt with fastmcp as base dependency
        requirements_file = tool_dir / "requirements.txt"
        with open(requirements_file, 'w', encoding='utf-8') as f:
            f.write("# Tool dependencies\nfastmcp\n")
            for req in requirements_content:
                f.write(f"{req}\n")
        
        # Create tool.py file with template or custom content
        if not tool_py_file_content:
            tool_py_file_content = f'''"""
{environment_name.title()} Tool

Auto-generated tool template for {environment_name}.
Edit this file to implement your tool functionality.

Functions not starting with underscore (_) will be automatically
exported as MCP tools and available for use.
"""

def example_function():
    """
    Example tool function - replace with your implementation
    
    Returns:
        str: Example return value
    """
    return f"Hello from {environment_name} tool!"

# Add your tool functions here...
'''
        
        tool_file = tool_dir / "tool.py"
        with open(tool_file, 'w', encoding='utf-8') as f:
            f.write(tool_py_file_content)
        
        # Create virtual environment
        try:
            venv_path = tool_loader.env_manager.ensure_virtual_environment(tool_dir)
            venv_created = True
        except Exception as e:
            logger.warning(f"Virtual environment creation failed: {e}")
            venv_created = False
        
        # Install dependencies
        install_result = {"success": False, "message": "Skipped - no virtual environment", "output": [], "error_output": []}
        if venv_created:
            try:
                install_result = await tool_loader.env_manager.install_requirements(tool_dir)
            except Exception as e:
                logger.warning(f"Dependency installation failed: {e}")
                install_result = {
                    "success": False,
                    "message": f"Installation exception: {str(e)}",
                    "output": [],
                    "error_output": [str(e)]
                }

        # noinspection PyUnboundLocalVariable
        return {
            "status": "success",
            "message": f"Tool environment '{environment_name}' created successfully",
            "details": {
                "environment_name": environment_name,
                "tool_directory": str(tool_dir.absolute()),
                "requirements_file": str(requirements_file.absolute()),
                "tool_file": str(tool_file.absolute()),
                "virtual_environment": str(venv_path.absolute()) if venv_created else None,
                "dependencies": requirements_content,
                "venv_created": venv_created,
                "dependencies_installed": install_result["success"],
                "installation_details": install_result
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create tool environment: {str(e)}"
        }

@mcp.tool
def tool_environment_get_info(environment_name: str) -> Dict[str, Any]:
    """
    获取特定工具环境的详细信息
    
    Args:
        environment_name: 环境名称
        
    Returns:
        工具环境的详细信息，包括虚拟环境路径、Python和pip位置、依赖信息等
        
    Examples:
        tool_environment_get_info("calculator")
        tool_environment_get_info("web_scraper")
    """
    try:
        tool_dir = Path(TOOLS_DIR) / environment_name
        
        if not tool_dir.exists():
            return {
                "status": "error",
                "message": f"Tool directory {environment_name} does not exist"
            }
        
        # 使用环境管理器获取详细信息
        env_info = tool_loader.env_manager.get_tool_environment_info(tool_dir)
        
        return {
            "status": "success",
            "environment_name": environment_name,
            "environment_info": env_info
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting environment info for {environment_name}: {str(e)}"
        }

@mcp.tool
def tool_environment_diagnose(environment_name: str) -> Dict[str, Any]:
    """
    诊断工具环境，检查问题
    
    Args:
        environment_name: 环境名称
        
    Returns:
        环境诊断结果和建议
        
    Examples:
        tool_environment_diagnose("calculator")
    """
    try:
        tool_dir = Path(TOOLS_DIR) / environment_name
        
        if not tool_dir.exists():
            return {
                "status": "error",
                "message": f"Tool directory {environment_name} does not exist"
            }
        
        # 获取环境信息
        env_info = tool_loader.env_manager.get_tool_environment_info(tool_dir)
        
        # 诊断结果
        diagnosis = {
            "environment_name": environment_name,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "status": "healthy"
        }
        
        # 检查基本文件
        if not env_info.get("has_tool_py"):
            diagnosis["issues"].append("Missing tool.py file")
            diagnosis["recommendations"].append("Create tool.py file with tool functions")
        
        if not env_info.get("has_requirements"):
            diagnosis["warnings"].append("No requirements.txt file found")
            diagnosis["recommendations"].append("Create requirements.txt if dependencies are needed")
        
        # 检查虚拟环境
        if not env_info.get("has_venv"):
            diagnosis["issues"].append("Virtual environment not created")
            diagnosis["recommendations"].append("Run create_tool_environment to set up virtual environment")
        elif not env_info.get("venv_valid"):
            diagnosis["issues"].append("Virtual environment is invalid")
            diagnosis["recommendations"].append("Recreate virtual environment")
        
        # 检查pip
        if env_info.get("has_venv") and not env_info.get("pip_exists"):
            diagnosis["issues"].append("pip is not available in virtual environment")
            diagnosis["recommendations"].append("Reinstall virtual environment with pip support")
        
        # 检查包安装
        if env_info.get("has_requirements") and env_info.get("packages_count", 0) == 0:
            diagnosis["warnings"].append("requirements.txt exists but no packages installed")
            diagnosis["recommendations"].append("Install requirements using: pip install -r requirements.txt")
        
        # 检查fastmcp依赖
        packages = env_info.get("installed_packages", [])
        fastmcp_installed = any(pkg.get("name") == "fastmcp" for pkg in packages)
        if not fastmcp_installed and env_info.get("has_venv"):
            diagnosis["warnings"].append("fastmcp not installed")
            diagnosis["recommendations"].append("Install fastmcp: pip install fastmcp")
        
        # 设置总体状态
        if diagnosis["issues"]:
            diagnosis["status"] = "critical"
        elif diagnosis["warnings"]:
            diagnosis["status"] = "warning"
        
        return {
            "status": "success",
            "diagnosis": diagnosis,
            "environment_info": env_info
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error diagnosing environment for {environment_name}: {str(e)}"
        }

@mcp.tool  
def tool_environment_repair(environment_name: str, force_recreate: bool = False) -> Dict[str, Any]:
    """
    修复工具环境的常见问题
    
    Args:
        environment_name: 环境名称
        force_recreate: 是否强制重新创建虚拟环境
        
    Returns:
        修复操作的结果
        
    Examples:
        tool_environment_repair("calculator")
        tool_environment_repair("web_scraper", force_recreate=True)
    """
    try:
        tool_dir = Path(TOOLS_DIR) / environment_name
        
        if not tool_dir.exists():
            return {
                "status": "error",
                "message": f"Tool directory {environment_name} does not exist"
            }
        
        repair_actions = []
        
        # 如果强制重新创建，先删除虚拟环境
        if force_recreate:
            if tool_loader.env_manager.cleanup_environment(tool_dir):
                repair_actions.append("Cleaned up existing virtual environment")
        
        # 确保虚拟环境存在
        try:
            venv_path = tool_loader.env_manager.ensure_virtual_environment(tool_dir)
            repair_actions.append(f"Ensured virtual environment at {venv_path}")
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to create virtual environment: {str(e)}",
                "actions_taken": repair_actions
            }
        
        # 安装依赖
        install_result = asyncio.run(tool_loader.env_manager.install_requirements(tool_dir))
        if install_result["success"]:
            repair_actions.append("Installed requirements successfully")
        else:
            repair_actions.append(f"Warning: Requirements installation failed - {install_result['message']}")
            if install_result.get("error_output"):
                repair_actions.append(f"Errors: {'; '.join(install_result['error_output'][:3])}")  # 显示前3个错误
        
        # 验证修复结果
        env_info = tool_loader.env_manager.get_tool_environment_info(tool_dir)
        
        return {
            "status": "success",
            "message": f"Tool environment {environment_name} repaired successfully",
            "actions_taken": repair_actions,
            "environment_info": env_info,
            "venv_requirements_installation_details": install_result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error repairing environment for {environment_name}: {str(e)}"
        }

@mcp.tool
async def tool_environment_update(environment_name: str, pip_requirements: Optional[List[str]] = None, tool_py_file_content: Optional[str] = None, force_reinstall: bool = False) -> Dict[str, Any]:
    """
    Update an existing tool environment with new requirements or template content
    
    Args:
        environment_name: Existing environment name
        pip_requirements: New dependency package list to update, e.g. ["fastmcp", "requests>=2.25.0", "pandas"]
        tool_py_file_content: New template content for tool.py, if provided will overwrite existing content
        force_reinstall: Whether to force reinstall all packages
        
    Returns:
        Result of the update operation
        
    Examples:
        tool_environment_update("calculator", ["fastmcp", "numpy", "scipy"])
        tool_environment_update("web_scraper", force_reinstall=True)
    """
    try:
        tool_dir = Path(TOOLS_DIR) / environment_name
        
        if not tool_dir.exists():
            return {
                "status": "error",
                "message": f"Tool directory {environment_name} does not exist"
            }
        
        update_actions = []
        install_result = {"success": True, "message": "No installation required", "output": [], "error_output": []}
        
        # Update requirements.txt if provided
        if pip_requirements is not None:
            requirements_content = []
            
            # Validate and clean dependency format
            for req in pip_requirements:
                req = req.strip()
                if req and not req.startswith("#"):
                    requirements_content.append(req)
            
            # Add fastmcp dependency by default if not present
            if "fastmcp" not in str(requirements_content):
                requirements_content.insert(0, "fastmcp")
            
            requirements_file = tool_dir / "requirements.txt"
            with open(requirements_file, 'w', encoding='utf-8') as f:
                f.write("# Tool dependencies\n")
                for req in requirements_content:
                    f.write(f"{req}\n")
            
            update_actions.append(f"Updated requirements.txt with {len(requirements_content)} dependencies")
        
        # Update tool.py if tool_py_file_content is provided
        if tool_py_file_content is not None:
            tool_file = tool_dir / "tool.py"
            with open(tool_file, 'w', encoding='utf-8') as f:
                f.write(tool_py_file_content)
            
            update_actions.append("Updated tool.py with new template content")
        
        # Reinstall dependencies if requirements were updated or force_reinstall is True
        if pip_requirements is not None or force_reinstall:
            try:
                # Ensure virtual environment exists
                venv_path = tool_loader.env_manager.ensure_virtual_environment(tool_dir)
                update_actions.append(f"Ensured virtual environment at {venv_path}")
                
                # If force_reinstall, uninstall all packages first (except pip, setuptools, wheel)
                if force_reinstall:
                    try:
                        python_exe = tool_loader.env_manager.get_python_executable(tool_dir)
                        
                        # Get list of installed packages
                        result = subprocess.run([
                            str(python_exe), "-m", "pip", "list", "--format=json"
                        ], capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            packages = json.loads(result.stdout)
                            
                            # Filter out system packages
                            user_packages = [
                                pkg["name"] for pkg in packages 
                                if pkg["name"].lower() not in ["pip", "setuptools", "wheel"]
                            ]
                            
                            if user_packages:
                                # Uninstall user packages
                                subprocess.run([
                                    str(python_exe), "-m", "pip", "uninstall", "-y"
                                ] + user_packages, capture_output=True, text=True, timeout=120)
                                
                                update_actions.append(f"Uninstalled {len(user_packages)} existing packages")
                    
                    except Exception as e:
                        logger.warning(f"Error during force reinstall for {environment_name}: {e}")
                
                # Install requirements
                install_result = await tool_loader.env_manager.install_requirements(tool_dir)
                if install_result["success"]:
                    update_actions.append("Successfully installed/updated requirements")
                else:
                    update_actions.append(f"Warning: Requirements installation failed - {install_result['message']}")
                    if install_result.get("error_output"):
                        update_actions.append(f"Installation errors: {'; '.join(install_result['error_output'][:3])}")  # 显示前3个错误
                    
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error updating environment: {str(e)}",
                    "actions_taken": update_actions
                }
        
        # Get updated environment info
        env_info = tool_loader.env_manager.get_tool_environment_info(tool_dir)
        
        return {
            "status": "success",
            "message": f"Tool environment {environment_name} updated successfully",
            "actions_taken": update_actions,
            "environment_info": env_info,
            "installation_details": install_result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error updating environment for {environment_name}: {str(e)}"
        }

@mcp.tool
async def tool_environment_current_functions(environment_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about all currently loaded functions and tools
    
    Args:
        environment_name: Optional environment name to reload specific tools
    
    Returns:
        Detailed information about all loaded functions, including built-in and dynamic tools
        
    Examples:
        tool_environment_current_functions()
        tool_environment_current_functions("calculator")
    """
    try:
        # Only reload the currently called tool
        reloaded_tools = await tool_loader.scan_and_load_tools(environment_name)
        if reloaded_tools:
            register_result = tool_loader.register_tools_to_mcp(reloaded_tools, environment_name)
        # Get all tools from MCP server
        existing_tools: Dict[str, FunctionTool] = mcp._tool_manager._tools  # type: ignore
        
        # Categorize tools
        builtin_tools = {}
        dynamic_tools = {}
        proxy_tools = {}
        
        for tool_name, tool_obj in existing_tools.items():
            tool_info = {
                "name": tool_name,
                "description": tool_obj.description,
                "parameters": tool_obj.parameters,
                "enabled": tool_obj.enabled,
                "tags": list(tool_obj.tags) if tool_obj.tags else []
            }
            
            # Categorize based on tool name pattern
            if "-" in tool_name:
                # Dynamic tool (has tool_dir.function_name format)
                dynamic_tools[tool_name] = tool_info
            elif tool_name.startswith("tool_environment_") or tool_name in [
                "search_github", "llm_web_search", "get_tools_changes", 
                "refresh_tools", "get_server_status"
            ]:
                # Built-in server tools
                builtin_tools[tool_name] = tool_info
            else:
                # Proxy tools from remote server
                proxy_tools[tool_name] = tool_info
        
        # Get environment information for dynamic tools
        dynamic_tool_environments = {}
        if dynamic_tools:
            tool_dirs = set(tool_name.split("-")[0] for tool_name in dynamic_tools.keys())
            for tool_dir_name in tool_dirs:
                tool_dir = Path(TOOLS_DIR) / tool_dir_name
                if tool_dir.exists():
                    dynamic_tool_environments[tool_dir_name] = tool_loader.env_manager.get_tool_environment_info(tool_dir)
        return {
            "status": "success",
            "summary": {
                "total_functions": len(existing_tools),
                "builtin_tools": len(builtin_tools),
                "dynamic_tools": len(dynamic_tools),
                "proxy_tools": len(proxy_tools),
                "tool_environments": len(dynamic_tool_environments)
            },
            "builtin_tools": builtin_tools,
            "dynamic_tools": dynamic_tools,
            "proxy_tools": proxy_tools,
            "tool_environments": dynamic_tool_environments,
            "server_info": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
                "tools_directory": str(Path(TOOLS_DIR).absolute())
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting current functions: {str(e)}"
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

async def main():
    """Start the dynamic MCP server"""
    dynamic_logger.print_section(
        "Dynamic MCP Server - Dynamic Tool Server",
        [f"v{SERVER_VERSION} | {SERVER_HOST}:{SERVER_PORT} | SSE Protocol"]
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
    # proxy = FastMCP.as_proxy("http://127.0.0.1:8931/mcp/")
    # This proxy will reuse the connected session for all requests
    from fastmcp import Client
    connected_client = Client("http://127.0.0.1:8931/mcp/")
    proxy = FastMCP.as_proxy(connected_client)
    remote_tools = await proxy.get_tools()
    tool_info: ProxyTool
    for tool_name, tool_info in remote_tools.items(): # type: ignore
        if tool_name in ["browser_resize", "browser_install", "browser_take_screenshot"]:
            continue
        try:
            # Create local copy
            local_tool = tool_info.copy()
            # Add to local server
            mcp.add_tool(local_tool)
            logger.info(f"Mirrored tool from remote server: {tool_info.name}")
        except Exception as e:
            logger.error(f"Failed to mirror tool {tool_info.name}: {e}")
    await connected_client.close()
    # ================================
    # Load Local Tools
    # ================================
    dynamic_logger.info("Loading local tools...")
    tools = await tool_loader.scan_and_load_tools()
    tool_loader.register_tools_to_mcp(tools)
    dynamic_logger.success(f"Loaded {len(tools)} local tools")
 
    dynamic_logger.print_status("Startup", "Server is starting...", True)
    console.print()
    # ================================
    # Load Code Tools
    # ================================
    from mcp_claude_code.server import ClaudeCodeServer
    ClaudeCodeServer(mcp_instance=mcp, allowed_paths=["/Users/yunhong/Desktop/AgentByAgent"], enable_agent_tool=False, command_timeout=300)
    mcp.remove_tool("multi_edit")
    mcp.remove_tool("notebook_read")
    mcp.remove_tool("notebook_edit")
    mcp.remove_tool("batch")
    mcp.remove_tool("todo_write")
    mcp.remove_tool("todo_read")
    # Start server
    await mcp.run_async(transport="http", host=SERVER_HOST, port=SERVER_PORT)

if __name__ == "__main__":
    asyncio.run(main())