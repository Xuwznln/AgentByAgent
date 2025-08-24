#!/usr/bin/env python3
"""
Tool Execution Script - 工具执行脚本

在独立的子进程中执行工具函数：
- 接收JSON格式的参数
- 动态导入工具模块
- 执行指定的函数
- 返回JSON格式的结果
- 在工具目录中运行，确保相对路径正确

使用方法:
    python tool_execution_script.py '{"module_path": "tool.py", "function_name": "add", "args": [1, 2], "kwargs": {}}'
"""

import sys
import json
import importlib.util
import traceback
import os

def serialize_result(obj):
    """序列化结果，处理不可序列化的对象"""
    try:
        # 尝试直接序列化
        json.dumps(obj, ensure_ascii=False)
        return obj
    except (TypeError, ValueError):
        # 如果无法序列化，转换为字符串
        return str(obj)

def execute_tool(module_path, function_name, args, kwargs):
    """执行工具函数并返回结果"""
    try:
        # 确保模块路径存在
        if not os.path.exists(module_path):
            return {"success": False, "error": f"Module file not found: {module_path}"}
        
        # 动态导入模块
        spec = importlib.util.spec_from_file_location("tool_module", module_path)
        if spec is None or spec.loader is None:
            return {"success": False, "error": f"Cannot load module from {module_path}"}
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 获取目标函数
        if not hasattr(module, function_name):
            available_functions = [name for name in dir(module) if not name.startswith('_')]
            return {"success": False, "error": f"Function {function_name} not found in module. Available functions: {available_functions}"}
        
        func = getattr(module, function_name)
        if not callable(func):
            return {"success": False, "error": f"{function_name} is not callable"}
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 序列化结果
        serialized_result = serialize_result(result)
        return {"success": True, "result": serialized_result}
    
    except Exception as e:
        return {
            "success": False, 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def main():
    """主函数"""
    if len(sys.argv) < 2:
        result = {"success": False, "error": "Usage: script.py <params_json>"}
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    
    try:
        # 解析参数
        params = json.loads(sys.argv[1])
        module_path = params.get("module_path")
        function_name = params.get("function_name") 
        args = params.get("args", [])
        kwargs = params.get("kwargs", {})
        
        # 验证必需参数
        if not module_path or not function_name:
            result = {"success": False, "error": "Missing required parameters: module_path, function_name"}
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(1)
        
        # 执行工具
        result = execute_tool(module_path, function_name, args, kwargs)
        print(json.dumps(result, ensure_ascii=False))
        
    except json.JSONDecodeError as e:
        result = {"success": False, "error": f"Invalid JSON parameters: {str(e)}"}
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    except Exception as e:
        result = {"success": False, "error": f"Script error: {str(e)}", "traceback": traceback.format_exc()}
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main() 