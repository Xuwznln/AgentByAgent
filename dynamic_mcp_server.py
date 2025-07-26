#!/usr/bin/env python3
"""
Dynamic MCP Server - 动态监控tools文件夹的MCP服务器

这个服务器会：
- 监控tools文件夹中的Python文件
- 动态导入和注册非_开头的函数作为工具
- 检测工具变更并记录差异
- 提供SSE接口在0.0.0.0:3001监听
"""

import sys
import importlib
import inspect
import json
import logging
from typing import Dict, Any, List, Callable
from datetime import datetime
from pathlib import Path
import hashlib

from fastmcp.tools import FunctionTool

# 应用 JSON monkey patch 修复 pydantic 序列化问题
from json_patch import apply_json_patch
apply_json_patch()

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.timing import DetailedTimingMiddleware
from fastmcp.server.middleware.logging import StructuredLoggingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware

# ================================
# 配置
# ================================

SERVER_NAME = "DynamicToolsServer"
SERVER_VERSION = "1.0.0"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 3001
TOOLS_DIR = "tools"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dynamic-mcp-server")

# 初始化FastMCP服务器
mcp = FastMCP(SERVER_NAME, version=SERVER_VERSION)

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
        if changes:
            change_record = {
                "timestamp": datetime.now().isoformat(),
                "changes": changes
            }
            self.change_history.append(change_record)
            logger.info(f"Detected tool changes: {json.dumps(changes, indent=2)}")
        
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
    """动态工具加载器"""
    
    def __init__(self, tools_dir: str):
        self.tools_dir = Path(tools_dir)
        self.loaded_modules: Dict[str, Any] = {}
        self.current_tools: Dict[str, Any] = {}
        
        # 确保tools目录存在
        self.tools_dir.mkdir(exist_ok=True)
        
        # 将tools目录添加到sys.path
        if str(self.tools_dir.absolute()) not in sys.path:
            sys.path.insert(0, str(self.tools_dir.absolute()))
    
    def scan_and_load_tools(self) -> Dict[str, Any]:
        """扫描并加载tools目录中的工具"""
        new_tools = {}
        
        # 扫描所有.py文件
        for py_file in self.tools_dir.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name == "__init__.py":
                continue
            
            module_name = py_file.stem
            try:
                # 重新加载模块以获取最新变更
                if module_name in self.loaded_modules:
                    importlib.reload(self.loaded_modules[module_name])
                else:
                    self.loaded_modules[module_name] = importlib.import_module(module_name)
                
                module = self.loaded_modules[module_name]
                
                # 获取模块中的所有函数
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    # 跳过私有函数（_开头）
                    if name.startswith("_"):
                        continue
                    
                    # 确保函数是在当前模块中定义的
                    if obj.__module__ == module_name:
                        tool_key = f"{module_name}.{name}"
                        new_tools[tool_key] = obj
                        logger.debug(f"Loaded tool: {tool_key}")
                
            except Exception as e:
                logger.error(f"Error loading module {module_name}: {e}")

        return new_tools
    
    def register_tools_to_mcp(self, tools: Dict[str, Callable]):
        """将工具注册到MCP服务器"""
        existing_tools: Dict[str, FunctionTool] = mcp._tool_manager._tools  # type: ignore
        existing_tools_desc = {k: v.model_dump() for k, v in existing_tools.items() if "." in k}
        for v in existing_tools_desc.values():
            del v["fn"]
        
        # 注册新工具和修改后的工具，构建新工具描述
        new_tools_desc = {}
        for tool_name, tool_func in tools.items():
            try:
                # 创建工具的描述
                doc = tool_func.__doc__ or f"Tool function {tool_name}"
                
                # 使用装饰器方式注册工具
                registered_tool = mcp.tool(
                    name=tool_name,
                    description=doc
                )(tool_func)
                
                # 获取注册后的工具描述
                updated_tools: Dict[str, FunctionTool] = mcp._tool_manager._tools  # type: ignore
                if tool_name in updated_tools:
                    new_tools_desc[tool_name] = updated_tools[tool_name].model_dump()
                    del new_tools_desc[tool_name]["fn"]
                
                logger.info(f"Registered tool: {tool_name}")
                
            except Exception as e:
                logger.error(f"Error registering tool {tool_name}: {e}")
        
        # 移除不再存在的工具
        for tool_name in existing_tools_desc:
            if tool_name not in tools and "." in tool_name:
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
        """在工具调用时刷新tools目录"""
        logger.warning(f"Refreshing tools before calling: {context.message.name}")
        
        # 重新扫描和加载工具
        tools = tool_loader.scan_and_load_tools()
        tool_loader.register_tools_to_mcp(tools)
        
        # 继续执行工具调用
        return await call_next(context)
    
    async def on_list_tools(self, context: MiddlewareContext, call_next):
        """在列出工具时刷新tools目录"""
        logger.warning("Refreshing tools before listing tools")
        
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
def list_tools_files() -> Dict[str, Any]:
    """
    列出tools目录中的所有Python文件
    
    Returns:
        tools目录中文件的信息
    """
    tools_path = Path(TOOLS_DIR)
    files_info = []
    
    if tools_path.exists():
        for py_file in tools_path.glob("*.py"):
            if not py_file.name.startswith("_"):
                files_info.append({
                    "filename": py_file.name,
                    "path": str(py_file),
                    "size": py_file.stat().st_size,
                    "modified_time": datetime.fromtimestamp(py_file.stat().st_mtime).isoformat(),
                    "hash": change_manager.get_file_hash(str(py_file))
                })
    
    return {
        "tools_directory": str(tools_path.absolute()),
        "files": files_info,
        "total_files": len(files_info)
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
        "change_history_count": len(change_manager.change_history)
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
    
    print("="*70)
    print("Dynamic MCP Server - 动态工具服务器")
    print("="*70)
    print(f"服务器: {SERVER_NAME} v{SERVER_VERSION}")
    print(f"监听地址: {SERVER_HOST}:{SERVER_PORT}")
    print(f"传输协议: SSE")
    print(f"工具目录: {Path(TOOLS_DIR).absolute()}")
    print()
    print("功能特性:")
    print("  - 动态监控tools文件夹")
    print("  - 自动加载非_开头的函数作为工具")
    print("  - 实时检测工具变更")
    print("  - 记录变更历史")
    print("  - SSE实时通信")
    print()
    print("内置工具:")
    print("  - get_tools_changes: 获取工具变更信息")
    print("  - refresh_tools: 手动刷新工具")
    print("  - list_tools_files: 列出工具文件")
    print("  - get_server_status: 获取服务器状态")
    print()
    print("启动服务器...")
    print("="*70)
    
    # 初始加载tools
    try:
        tools = tool_loader.scan_and_load_tools()
        tool_loader.register_tools_to_mcp(tools)
        print(f"初始加载了 {len(tools)} 个工具")
    except Exception as e:
        logger.error(f"初始工具加载失败: {e}")
    from mcp_claude_code.server import ClaudeCodeServer
    ClaudeCodeServer(mcp_instance=mcp, allowed_paths=["/home/wz/AgentByAgent/tools"], enable_agent_tool=False)
    # 启动服务器
    mcp.run(transport="sse", host=SERVER_HOST, port=SERVER_PORT)

if __name__ == "__main__":
    main() 