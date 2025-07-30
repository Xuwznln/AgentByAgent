import importlib.util
import inspect
import json
import sys
import traceback
from pathlib import Path



def load_tools_from_module(module_path, tool_name_prefix=""):
    """从模块加载工具并返回FunctionTool对象的序列化数据"""
    tools_data = []

    try:
        # 导入FastMCP
        try:
            from fastmcp import FastMCP
            from fastmcp.tools.tool import Tool
        except ImportError:
            return {"error": "FastMCP not available in this environment. Please install fastmcp."}

        # 确保模块文件存在
        if not Path(module_path).exists():
            return {"error": f"Module file not found: {module_path}"}

        # 创建临时FastMCP实例
        temp_mcp = FastMCP("TempToolLoader")

        # 动态导入模块
        spec = importlib.util.spec_from_file_location("tool_module", module_path)
        if spec is None or spec.loader is None:
            return {"error": f"Cannot load module from {module_path}"}

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 获取模块中的所有函数并注册为工具
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            # 跳过私有函数
            if name.startswith("_"):
                continue

            # 确保函数是在当前模块中定义的
            if obj.__module__ == "tool_module":
                try:
                    # 构建工具名称
                    tool_name = f"{tool_name_prefix}-{name}" if tool_name_prefix else name
                    function_tool = Tool.from_function(
                        obj,
                        name=tool_name,
                        description=obj.__doc__ or f"Tool function {name}",
                    )
                    tool_data = function_tool.model_dump()
                    del tool_data["fn"]
                    tool_data["tags"] = list(tool_data.get("tags", []))
                    # 添加额外的元数据
                    tool_data.update({
                        "source_module": str(module_path),
                        "function_name": name,
                        "tool_name_prefix": tool_name_prefix
                    })
                    tools_data.append(tool_data)
                except Exception as e:
                    traceback.print_exc()
                    return {"error": f"Error registering tool {name}: {str(e)}", "traceback": traceback.format_exc()}
                    
    except Exception as e:
        return {"error": f"Error loading tools: {str(e)}", "traceback": traceback.format_exc()}

    return {"tools": tools_data}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        result = {"error": "Usage: script.py <module_path> <tool_name_prefix>"}
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)

    module_path = sys.argv[1]
    tool_name_prefix = sys.argv[2]

    result = load_tools_from_module(module_path, tool_name_prefix)
    print(json.dumps(result, ensure_ascii=False))
