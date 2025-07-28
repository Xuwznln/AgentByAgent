#!/usr/bin/env python3
"""
Dynamic MCP Server - 动态监控tools文件夹的MCP服务器

这个服务器会：
- 监控tools文件夹中的Python文件
- 动态导入和注册非_开头的函数作为工具
- 检测工具变更并记录差异
- 提供SSE接口在0.0.0.0:3001监听
- 智能缓存：只在工具调用时重新加载当前被调用的工具（而非全局重载）
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

# 应用 JSON monkey patch 修复 pydantic 序列化问题
from tools.json_patch import apply_json_patch
apply_json_patch()

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.timing import DetailedTimingMiddleware
from fastmcp.server.middleware.logging import StructuredLoggingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.tools import FunctionTool

# 导入新的环境管理和代理模块
from tools.tool_env_manager import ToolEnvironmentManager
from tools.tool_proxy import ToolProxyManager, ToolProxy

# 导入增强的日志系统
from tools.logger_config import dynamic_logger, console

# ================================
# 配置
# ================================

SERVER_NAME = "DynamicToolsServer"
SERVER_VERSION = "1.0.0"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 3002
TOOLS_DIR = "tools"

# 配置增强日志系统
logger = dynamic_logger.get_logger("dynamic-mcp-server")

# 初始化FastMCP服务器
mcp = FastMCP(SERVER_NAME, version=SERVER_VERSION)
config = json.load(open("config.json", "r", encoding="utf-8"))

# ================================
# 工具变更管理器
# ================================

class ToolChangeManager:
    """管理工具的变更检测和记录"""
    
    def __init__(self):
        self.previous_tools: Dict[str, Dict[str, Any]] = {}
        self.current_tools: Dict[str, Dict[str, Any]] = {}
        self.change_history: List[Dict[str, Any]] = []
        self.file_hashes: Dict[str, str] = {}
    
    def get_file_hash(self, filepath: str) -> str:
        """获取文件的MD5哈希"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def update_tools(self, existing_tools_desc: Dict[str, Dict[str, Any]], new_tools_desc: Dict[str, Dict[str, Any]]):
        """更新工具列表并检测变更"""
        self.previous_tools = existing_tools_desc.copy()
        self.current_tools = new_tools_desc.copy()
        
        # 检测变更
        changes = self.detect_changes()
        if changes and (len(changes["added"]) or len(changes["modified"]) or len(changes["removed"])):
            change_record = {
                "timestamp": datetime.now().isoformat(),
                "changes": changes
            }
            self.change_history.append(change_record)

        return changes
    
    def detect_changes(self) -> Dict[str, Any]:
        """检测工具变更，提供详细的值对比"""
        changes = {
            "added": [],
            "removed": [],
            "modified": []
        }
        
        # 检测新增的工具
        for tool_name in self.current_tools:
            if tool_name not in self.previous_tools:
                changes["added"].append({
                    "name": tool_name,
                    "details": self.current_tools[tool_name]
                })
        
        # 检测删除的工具
        for tool_name in self.previous_tools:
            if tool_name not in self.current_tools:
                changes["removed"].append({
                    "name": tool_name,
                    "details": self.previous_tools[tool_name]
                })
        
        # 检测修改的工具
        for tool_name in self.current_tools:
            if tool_name in self.previous_tools:
                if self.current_tools[tool_name] != self.previous_tools[tool_name]:
                    # 详细对比差异
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
        """获取两个工具描述之间的详细差异"""
        differences = {}
        
        # 检查所有键的变化
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
        """获取变更摘要"""
        return {
            "current_tools_count": len(self.current_tools),
            "previous_tools_count": len(self.previous_tools),
            "recent_changes": self.change_history[-2:] if self.change_history else [],
            "tool_details": {
                "current": list(self.current_tools.keys()),
                "previous": list(self.previous_tools.keys())
            }
        }

# 全局变更管理器
change_manager = ToolChangeManager()

# ================================
# 动态工具加载器
# ================================

class DynamicToolLoader:
    """动态工具加载器 - 支持独立环境"""
    
    def __init__(self, tools_dir: str):
        self.tools_dir = Path(tools_dir)
        self.loaded_modules: Dict[str, Any] = {}
        self.current_tools: Dict[str, Any] = {}
        
        # 初始化环境管理器和代理管理器
        self.env_manager = ToolEnvironmentManager(tools_dir)
        self.proxy_manager = ToolProxyManager()
        
        # 确保tools目录存在
        self.tools_dir.mkdir(exist_ok=True)
    
    def scan_and_load_tools(self, request_tool_name: Optional[str] = None) -> Dict[str, Any]:
        """扫描并加载tools目录中的工具（使用独立环境）"""
        new_tools = {}
        # 使用环境管理器加载所有工具
        load_result = self.env_manager.load_all_tools(request_tool_name.split(".")[0] if request_tool_name is not None else None)
        
        if "tools" in load_result:
            # load_result["tools"] 是字典格式 {tool_name: tool_data}
            for tool_name, tool_data in load_result["tools"].items():
                # 获取工具目录
                tool_dir_name = tool_name.split(".")[0]
                tool_dir = self.tools_dir / tool_dir_name
                
                if tool_dir.exists():
                    try:
                        # 获取Python可执行文件
                        python_exe = self.env_manager.get_python_executable(tool_dir)
                        
                        # 创建工具代理
                        proxy = self.proxy_manager.create_proxy(tool_data, tool_dir, python_exe)
                        
                        # 存储工具数据和代理
                        new_tools[tool_name] = {
                            "tool_data": tool_data,
                            "proxy": proxy,
                            "tool_dir": tool_dir,
                            "python_exe": python_exe
                        }
                        
                        logger.debug(f"Loaded tool via proxy: {tool_name}")
                        
                    except Exception as e:
                        logger.error(f"Error creating proxy for tool {tool_name}: {e}")
        
        # 记录加载错误
        if "errors" in load_result and load_result["errors"]:
            for error_info in load_result["errors"]:
                logger.error(f"Tool loading error in {error_info['tool_dir']}: {error_info['error']}")
        
        logger.info(f"Loaded {len(new_tools)} tools via environment isolation")
        return new_tools

    def register_tools_to_mcp(self, tools: Dict[str, Any], request_tool_name: Optional[str] = None) -> Dict[str, Any]:
        """将工具注册到MCP服务器（使用FunctionTool数据和代理）"""
        existing_tools: Dict[str, FunctionTool] = mcp._tool_manager._tools  # type: ignore
        existing_tools_desc = {k: v.model_dump() for k, v in existing_tools.items() if "." in k}
        for v in existing_tools_desc.values():
            del v["fn"]
        request_tool_dir = request_tool_name.split(".")[0] if request_tool_name is not None else self.tools_dir.name
        # 注册新工具
        new_tools_desc = {}
        for tool_name, tool_info in tools.items():
            try:
                tool_data = tool_info["tool_data"]
                proxy: ToolProxy = tool_info["proxy"]
                tool_data["fn"] = proxy.__call__
                
                # 重新构造FunctionTool对象
                function_tool = FunctionTool.model_validate({k: v for k, v in tool_data.items() if k not in [
                    "source_module", "function_name", "tool_name_prefix"
                ]})
                # 直接添加到MCP服务器
                if tool_name in existing_tools:
                    mcp.remove_tool(tool_name)
                mcp.add_tool(function_tool)
                # 记录工具描述（用于变更检测）
                tool_desc = function_tool.model_dump()
                del tool_desc["fn"]
                new_tools_desc[tool_name] = tool_desc
                logger.info(f"Registered proxied tool: {tool_name}")
                
            except Exception as e:
                logger.error(f"Error registering tool {tool_name}: {e.args}\n{traceback.format_exc()}")
        
        # 移除不再存在的工具
        for tool_name in existing_tools_desc:
            if tool_name not in tools and "." in tool_name:
                # 对于on tool call的情况，不要删除自定义tool
                if request_tool_dir is not None and not tool_name.startswith(f"{request_tool_dir}."):
                    continue
                try:
                    mcp.remove_tool(tool_name)
                    logger.info(f"Removed tool: {tool_name}")
                except Exception as e:
                    logger.error(f"Error removing tool {tool_name}: {e}")
        
        # 检测并记录变更
        changes = change_manager.update_tools(existing_tools_desc, new_tools_desc)
        return changes

# 全局工具加载器
tool_loader = DynamicToolLoader(TOOLS_DIR)

# ================================
# 中间件
# ================================

class DynamicToolMiddleware(Middleware):
    """动态工具中间件，在每次工具调用和列出工具时刷新tools"""

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """在工具调用时刷新当前工具"""
        tool_name = context.message.name
        if "." in tool_name:
            logger.info(f"Refreshing specific tool before calling: {tool_name}")
            # 只重新加载当前被调用的工具
            reloaded_tools = tool_loader.scan_and_load_tools(tool_name)
            if reloaded_tools:
                # 重新注册该工具到MCP
                register_result = tool_loader.register_tools_to_mcp(reloaded_tools, tool_name)
            else:
                logger.warning(f"No tools were reloaded for {tool_name}")
        
        # 继续执行工具调用
        return await call_next(context)
    
    async def on_list_tools(self, context: MiddlewareContext, call_next):
        """在列出工具时刷新tools目录"""
        logger.info("Refreshing all tools before listing tools")
        # 重新扫描和加载工具
        tools = tool_loader.scan_and_load_tools()
        tool_loader.register_tools_to_mcp(tools)
        # 继续执行列出工具
        return await call_next(context)

# ================================
# 添加中间件
# ================================

mcp.add_middleware(ErrorHandlingMiddleware(include_traceback=True))
mcp.add_middleware(DynamicToolMiddleware())
mcp.add_middleware(DetailedTimingMiddleware())
mcp.add_middleware(StructuredLoggingMiddleware(include_payloads=True))

# ================================
# 内置工具
# ================================

@mcp.tool
def search_github(query: str, max_results: int = 10, sort_by: str = "stars") -> List[Dict[str, Any]]:
    """
    搜索GitHub的Python语言仓库并按指定条件排序
    
    Args:
        query: 搜索关键词
        max_results: 返回结果数量限制
        sort_by: 排序方式，可选: "stars", "forks", "updated"
    
    Returns:
        按指定条件排序的仓库列表，包含star和fork数量
    
    Examples:
        search_github("machine learning")  # 搜索Python机器学习项目
    """
    if not query:
        raise Exception("query is empty")
    
    # 构建搜索查询字符串
    search_query = query
    search_query += f" language:Python"
    
    # GitHub搜索API支持排序参数
    url = f"https://api.github.com/search/repositories?q={parse.quote(search_query)}&sort={sort_by}&order=desc"
    logger.info(url)
    try:
        with request.urlopen(url) as resp:
            data = json.load(resp)
        
        items = data.get("items", [])[:max_results]
        
        # 提取关键信息并格式化
        results = []
        for i, item in enumerate(items, 1):
            repo_info = {
                "rank": i,
                "name": item["full_name"],
                "url": item["html_url"],
                "description": item.get("description", "无描述"),
                "stars": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
                "language": item.get("language", "未知"),
                "updated_at": item.get("updated_at", ""),
                "topics": item.get("topics", [])
            }
            results.append(repo_info)
        
        logger.info(f"GitHub搜索'{query}'返回{len(results)}个结果，按{sort_by}排序")
        return results
        
    except Exception as e:
        logger.error(f"GitHub搜索失败: {e}")
        return [{"error": f"搜索失败: {str(e)}"}]


# noinspection PyTypeChecker
@mcp.tool
def advanced_web_search(query: str) -> str:
    """
    使用AI增强的网络搜索功能，提供更智能和准确的搜索结果
    
    Args:
        query: 搜索查询，支持中英文，可以是问题或关键词
    
    Returns:
        AI分析和整理后的搜索结果
    
    Examples:
        advanced_web_search("Python异步编程最佳实践")
        advanced_web_search("What is the latest news about AI development?")
    """
    try:
        # Configure OpenAI client with configuration values
        openai_config = config.get("openai", {})
        api_key = openai_config.get("api_key", "")
        base_url = openai_config.get("base_url", "https://api.openai.com/v1/")
        model = openai_config.get("model", "gpt-4.1")
        if not api_key:
            return "❌ 配置错误: OpenAI API密钥未设置，请检查config.json文件"
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
        logger.info(f"Advanced web search completed successfully, result length: {len(search_result)} characters")
        # Return formatted result
        return search_result
        
    except Exception as e:
        error_msg = f"高级网络搜索失败: {str(e)}"
        logger.error(error_msg)
        return f"❌ 搜索错误: {error_msg}"

@mcp.tool
def get_tools_changes() -> Dict[str, Any]:
    """
    获取工具变更信息，对比本次和上一次的不同
    
    Returns:
        包含工具变更详情的字典
    """
    return change_manager.get_change_summary()

@mcp.tool
def refresh_tools() -> Dict[str, Any]:
    """
    手动刷新tools目录中的工具
    
    Returns:
        刷新操作的结果
    """
    try:
        # 重新扫描和加载工具
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
    获取服务器状态信息
    
    Returns:
        服务器状态信息
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
    创建新的工具环境，包括目录、虚拟环境、requirements.txt和基础tool.py文件
    
    Args:
        tool_name: 工具名称（只能包含字母、数字、下划线）
        requirements: 依赖包列表，例如 ["fastmcp", "requests>=2.25.0", "pandas"]
        template_content: tool.py的模板内容，如果为空则使用默认模板
        
    Returns:
        创建操作的结果和详细信息
        
    Examples:
        create_tool_environment("my_calculator", ["fastmcp", "numpy"])
        create_tool_environment("web_scraper", ["requests", "beautifulsoup4", "fastmcp"])
    """
    # 验证工具名称
    if not tool_name or not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', tool_name):
        return {
            "status": "error",
            "message": "工具名称只能包含字母、数字、下划线，且必须以字母开头"
        }
    
    try:
        tool_dir = Path(TOOLS_DIR) / tool_name
        
        # 检查目录是否已存在
        if tool_dir.exists():
            return {
                "status": "error",
                "message": f"工具目录 {tool_name} 已存在"
            }
        
        # 创建工具目录
        tool_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建requirements.txt
        requirements_content = []
        if requirements:
            # 验证并清理依赖格式
            for req in requirements:
                req = req.strip()
                if req and not req.startswith("#"):
                    requirements_content.append(req)
        
        # 默认添加fastmcp依赖
        if "fastmcp" not in str(requirements_content):
            requirements_content.insert(0, "fastmcp")
        
        requirements_file = tool_dir / "requirements.txt"
        with open(requirements_file, 'w', encoding='utf-8') as f:
            f.write("# Tool dependencies\n")
            for req in requirements_content:
                f.write(f"{req}\n")
        
        # 创建默认tool.py文件
        if not template_content:
            template_content = f'''"""
{tool_name.title()} Tool - {tool_name}工具

这是一个自动生成的工具模板。
请编辑此文件来实现您的工具功能。
"""

# 在这里添加更多工具函数...
'''
        tool_file = tool_dir / "tool.py"
        with open(tool_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        # 使用环境管理器创建虚拟环境
        try:
            venv_path = tool_loader.env_manager.ensure_virtual_environment(tool_dir)
            venv_created = True
        except Exception as e:
            logger.warning(f"虚拟环境创建失败: {e}")
            venv_created = False
        
        # 尝试安装依赖
        deps_installed = False
        if venv_created:
            try:
                deps_installed = tool_loader.env_manager.install_requirements(tool_dir)
            except Exception as e:
                logger.warning(f"依赖安装失败: {e}")

        # noinspection PyUnboundLocalVariable
        return {
            "status": "success",
            "message": f"工具环境 {tool_name} 创建成功",
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
            "message": f"创建工具环境失败: {str(e)}"
        }

# ================================
# 资源
# ================================

@mcp.resource("config://server")
def get_server_config() -> dict:
    """获取服务器配置信息"""
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
# 启动函数
# ================================

def main():
    """启动动态MCP服务器"""
    dynamic_logger.print_banner(
        "Dynamic MCP Server - 动态工具服务器",
        f"v{SERVER_VERSION} | {SERVER_HOST}:{SERVER_PORT} | SSE协议"
    )
    console.print()
    dynamic_logger.print_section("服务器配置", [
        f"服务器名称: [bold cyan]{SERVER_NAME}[/bold cyan]",
        f"版本: [bold green]{SERVER_VERSION}[/bold green]", 
        f"监听地址: [bold yellow]{SERVER_HOST}:{SERVER_PORT}[/bold yellow]",
        f"传输协议: [bold magenta]SSE[/bold magenta]",
        f"工具目录: [bold blue]{Path(TOOLS_DIR).absolute()}[/bold blue]"
    ], "cyan")
    console.print()
    # ================================
    # 镜像远程MCP服务器工具
    # ================================
    try:
        proxy: FastMCPProxy = FastMCP.as_proxy("http://127.0.0.1:8931/sse/")
        remote_tools = asyncio.run(proxy.get_tools())
        tool_info: ProxyTool
        for tool_name, tool_info in remote_tools.items(): # type: ignore
            try:
                # 创建本地副本
                local_tool = tool_info.copy()
                # 添加到本地服务器
                mcp.add_tool(local_tool)
                logger.info(f"Mirrored tool from remote server: {tool_info.name}")
            except Exception as e:
                logger.error(f"Failed to mirror tool {tool_info.name}: {e}")
    except Exception as e:
        dynamic_logger.warning(f"无法连接到远程MCP服务器: {e}")
        dynamic_logger.info("将仅使用本地工具继续启动...")
    # ================================
    # 加载本地工具
    # ================================
    dynamic_logger.info("正在加载本地工具...")
    tools = tool_loader.scan_and_load_tools()
    tool_loader.register_tools_to_mcp(tools)
    dynamic_logger.success(f"已加载 {len(tools)} 个本地工具")
    
    dynamic_logger.print_status("启动", "服务器正在启动...", True)
    console.print()
    # ================================
    # 加载代码工具
    # ================================
    from mcp_claude_code.server import ClaudeCodeServer
    ClaudeCodeServer(mcp_instance=mcp, allowed_paths=["/home/wz/AgentByAgent/tools"], enable_agent_tool=False)
    # 启动服务器
    mcp.run(transport="http", host=SERVER_HOST, port=SERVER_PORT)

if __name__ == "__main__":
    main()  # 同步直接执行，异步交给mcp.run内部处理