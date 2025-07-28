#!/usr/bin/env python3
"""
Tool Proxy - 工具代理模块

为独立环境中的工具提供代理功能：
- 在主进程中创建工具代理
- 通过子进程执行实际的工具调用
- 处理参数序列化和结果反序列化
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List
import os

logger = logging.getLogger(__name__)

class ToolProxy:
    """工具代理类"""
    
    def __init__(self, tool_data: Dict[str, Any], tool_dir: Path, python_executable: Path):
        self.tool_data = tool_data  # FunctionTool序列化数据
        self.tool_dir = tool_dir.resolve()  # 转换为绝对路径
        self.python_executable = python_executable.absolute()  # 转换为绝对路径
        self.tool_name = tool_data["name"]
        
        # 从tool_data中获取function_name，如果没有则从name中提取
        if "function_name" in tool_data:
            self.function_name = tool_data["function_name"]
        else:
            self.function_name = tool_data["name"].split(".")[-1]  # 获取原始函数名
        
    def get_execution_script_path(self) -> Path:
        """获取工具执行脚本的绝对路径"""
        # 获取项目根目录（从当前文件位置推导）
        current_file = Path(__file__).resolve()  # tools/tool_proxy.py
        project_root = current_file.parent.parent  # 回到项目根目录
        script_path = project_root / "tools" / "tool_execution_script.py"
        return script_path
    
    def __call__(self, *args, **kwargs) -> Any:
        """代理函数调用 - 出错时抛出异常让MCP框架处理"""
        try:
            # 获取执行脚本的绝对路径
            script_path = self.get_execution_script_path()
            if not script_path.exists():
                raise FileNotFoundError(f"Tool execution script not found: {script_path}")
            
            # 准备参数（使用相对路径，在工具目录下执行）
            params = {
                "module_path": "tool.py",  # 工具目录下的相对路径
                "function_name": self.function_name,
                "args": list(args),
                "kwargs": kwargs
            }
            params_json = json.dumps(params, ensure_ascii=False)
            logger.debug(f"Executing tool {self.tool_name} in directory {self.tool_dir}")
            logger.debug(f"Script: {script_path}, Params: {params}")
            # 在工具目录中执行脚本（使用绝对路径执行脚本）
            result = subprocess.run([
                    str(self.python_executable), str(script_path), params_json
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.tool_dir),  # 设置工作目录为工具目录
                env=dict(os.environ, PYTHONPATH=str(self.tool_dir))  # 添加工具目录到Python路径
            )
            output = json.loads(result.stdout)
            # 检查工具执行结果
            if output.get("success"):
                return output["result"]
            raise RuntimeError(f"Tool execution error: {output.get('error', '')}\n{output.get('traceback', '')}")
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Tool {self.tool_name} execution timeout (>60s)")

class ToolProxyManager:
    """工具代理管理器"""
    
    def __init__(self):
        self.proxies: Dict[str, ToolProxy] = {}
    
    def create_proxy(self, tool_data: Dict[str, Any], tool_dir: Path, python_executable: Path) -> ToolProxy:
        """创建工具代理"""
        tool_name = tool_data["name"]
        proxy = ToolProxy(tool_data, tool_dir, python_executable)
        self.proxies[tool_name] = proxy
        return proxy
    
    def get_proxy(self, tool_name: str) -> ToolProxy:
        """获取工具代理"""
        if tool_name not in self.proxies:
            raise KeyError(f"Tool proxy {tool_name} not found")
        return self.proxies[tool_name]
    
    def remove_proxy(self, tool_name: str) -> bool:
        """移除工具代理"""
        if tool_name in self.proxies:
            del self.proxies[tool_name]
            return True
        return False
    
    def list_proxies(self) -> List[str]:
        """列出所有代理工具"""
        return list(self.proxies.keys())
    
    def clear_proxies(self):
        """清除所有代理"""
        self.proxies.clear() 