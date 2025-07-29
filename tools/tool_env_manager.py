#!/usr/bin/env python3
"""
Tool Environment Manager - 工具环境管理器

管理tools目录下每个工具文件夹的虚拟环境：
- 自动创建和管理虚拟环境
- 安装工具依赖
- 通过子进程在独立环境中加载工具
- 进程间通信获取工具元数据
- 智能缓存系统：基于文件修改时间自动缓存工具元数据

运行示例:
    # 基本使用
    python -c "
    from tool_env_manager import ToolEnvironmentManager
    manager = ToolEnvironmentManager('tools')
    result = manager.load_all_tools()
    print(f'加载了 {result[\"total_tools\"]} 个工具')
    print(f'缓存统计: {result[\"cache_stats\"]}')
    for name, meta in result['tools'].items():
        print(f'  - {name}: {meta[\"description\"][:50]}...')
    "
    
    # 测试特定工具环境
    python -c "
    from tool_env_manager import ToolEnvironmentManager
    from pathlib import Path
    manager = ToolEnvironmentManager('tools')
    info = manager.get_tool_environment_info(Path('tools/calculator'))
    print('Calculator环境信息:', info)
    "
    
    # 获取所有环境信息
    python -c "
    from tool_env_manager import ToolEnvironmentManager
    import json
    manager = ToolEnvironmentManager('tools')
    envs = manager.get_all_environments_info()
    print(json.dumps(envs, indent=2))
    "
    
    # 缓存管理
    python -c "
    from tool_env_manager import ToolEnvironmentManager
    import json
    manager = ToolEnvironmentManager('tools')
    cache_info = manager.get_cache_info()
    print('缓存信息:', json.dumps(cache_info, indent=2))
    manager.clear_cache()
    print('缓存已清理')
    "
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ToolEnvironmentManager:
    """工具环境管理器"""
    
    def __init__(self, tools_dir: str = "tools"):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(exist_ok=True)
        
        # 存储环境信息和缓存
        self.env_cache: Dict[str, Dict[str, Any]] = {}
        self.tools_cache: Dict[str, Dict[str, Any]] = {}
        
    def get_tool_directories(self) -> List[Path]:
        """获取所有工具目录"""
        tool_dirs = []
        for item in self.tools_dir.iterdir():
            if item.is_dir() and not item.name.startswith("_") and item.name != "__pycache__":
                tool_dirs.append(item)
        return tool_dirs
    
    def get_tool_file_info(self, tool_dir: Path) -> Dict[str, float]:
        """获取工具目录中关键文件的修改时间"""
        file_info = {}
        
        # 检查 tool.py
        tool_file = tool_dir / "tool.py"
        if tool_file.exists():
            file_info["tool.py"] = tool_file.stat().st_mtime
        
        # 检查 requirements.txt
        req_file = tool_dir / "requirements.txt"
        if req_file.exists():
            file_info["requirements.txt"] = req_file.stat().st_mtime
        
        return file_info
    
    def is_tool_cache_valid(self, tool_dir: Path) -> bool:
        """检查工具缓存是否有效"""
        tool_name = tool_dir.name
        
        if tool_name not in self.tools_cache:
            logger.debug(f"No cache found for tool {tool_name}")
            return False
        
        cached_info = self.tools_cache[tool_name]
        if "file_mtimes" not in cached_info:
            logger.debug(f"No file modification times in cache for tool {tool_name}")
            return False
        
        # 获取当前文件修改时间
        current_file_info = self.get_tool_file_info(tool_dir)
        cached_file_info = cached_info["file_mtimes"]
        
        # 比较文件修改时间
        for file_name, current_mtime in current_file_info.items():
            if file_name not in cached_file_info:
                logger.debug(f"New file {file_name} found in tool {tool_name}")
                return False
            
            if current_mtime > cached_file_info[file_name]:
                logger.debug(f"File {file_name} in tool {tool_name} has been modified")
                return False
        
        # 检查是否有文件被删除
        for file_name in cached_file_info:
            if file_name not in current_file_info:
                logger.debug(f"File {file_name} in tool {tool_name} has been deleted")
                return False
        
        logger.debug(f"Cache is valid for tool {tool_name}")
        return True
    
    def update_tool_cache(self, tool_dir: Path, tools_data: List[Dict[str, Any]]) -> None:
        """更新工具缓存"""
        tool_name = tool_dir.name
        file_info = self.get_tool_file_info(tool_dir)
        
        self.tools_cache[tool_name] = {
            "tools": tools_data,
            "file_mtimes": file_info,
            "last_loaded": time.time()
        }
        
        logger.info(f"Updated cache for tool {tool_name} with {len(tools_data)} tools")
    
    def ensure_virtual_environment(self, tool_dir: Path) -> Path:
        """确保工具目录有虚拟环境"""
        venv_path = tool_dir / "venv"
        
        if not venv_path.exists():
            logger.info(f"Creating virtual environment for {tool_dir.name}")
            try:
                subprocess.run([
                    sys.executable, "-m", "virtualenv", os.path.abspath(venv_path)
                ], capture_output=True, text=True, check=True, env=os.environ)
                logger.info(f"Virtual environment created at {venv_path}")
            except Exception as e:
                logger.error(f"Failed to create virtual environment for {tool_dir.name}: {e}")
                raise
        
        return venv_path
    
    def get_python_executable(self, tool_dir: Path) -> Path:
        """获取工具虚拟环境的Python可执行文件路径"""
        venv_path = self.ensure_virtual_environment(tool_dir)
        
        # Windows vs Unix路径差异
        if os.name == 'nt':
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        
        if not python_exe.exists():
            raise FileNotFoundError(f"Python executable not found at {python_exe}")
        
        return python_exe
    async def install_requirements(self, tool_dir: Path) -> Dict[str, Any]:
        """异步安装工具的依赖包，返回详细结果"""
        requirements_file = tool_dir / "requirements.txt"
        
        if not requirements_file.exists():
            logger.info(f"No requirements.txt found for {tool_dir.name}")
            return {
                "success": True,
                "message": "No requirements.txt found",
                "output": [],
                "error_output": []
            }
        
        try:
            python_exe = self.get_python_executable(tool_dir)

            print(f"[DEBUG] Installing requirements for {tool_dir.name}...")
            print(f"[DEBUG] Command: {python_exe} -m pip install -r {requirements_file}")
            
            # 使用异步子进程安装依赖
            process = await asyncio.create_subprocess_exec(
                str(python_exe), "-m", "pip", "install", "-r", str(requirements_file), 
                "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            # 收集输出和错误
            output_lines = []
            error_lines = []
            
            # 异步读取进程输出
            try:
                stdout = process.stdout
                if stdout is not None:
                    while True:
                        line = await stdout.readline()
                        if not line:
                            break
                        
                        decoded_line = line.decode().rstrip()
                        if decoded_line:
                            # 直接处理输出行
                            print(f"[{tool_dir.name}] {decoded_line}")
                            output_lines.append(decoded_line)
                            
                            # 检测错误行
                            if "ERROR:" in decoded_line or "FAILED:" in decoded_line or "Could not find" in decoded_line:
                                error_lines.append(decoded_line)
                
                # 等待进程结束
                return_code = await process.wait()
                
            except Exception as read_error:
                logger.error(f"Error reading process output: {read_error}")
                return_code = -1
            
            if return_code != 0:
                logger.error(f"Failed to install requirements for {tool_dir.name}, return code: {return_code}")
                return {
                    "success": False,
                    "message": f"Installation failed with return code {return_code}",
                    "output": output_lines,
                    "error_output": error_lines,
                    "return_code": return_code
                }
            
            logger.info(f"Requirements installed for {tool_dir.name}")
            return {
                "success": True,
                "message": "Requirements installed successfully",
                "output": output_lines,
                "error_output": error_lines,
                "return_code": return_code
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout installing requirements for {tool_dir.name}")
            return {
                "success": False,
                "message": "Installation timeout",
                "output": [],
                "error_output": ["Installation process timed out"]
            }
        except Exception as e:
            logger.error(f"Error installing requirements for {tool_dir.name}: {e}")
            return {
                "success": False,
                "message": f"Installation error: {str(e)}",
                "output": [],
                "error_output": [str(e)]
            }
    async def load_tools_from_environment(self, tool_dir: Path) -> Dict[str, Any]:
        """在独立环境中异步加载工具"""
        tool_file = tool_dir / "tool.py"
        if not tool_file.exists():
            return {"error": f"No tool.py found in {tool_dir}"}
        
        try:
            # 确保虚拟环境存在并安装依赖
            self.ensure_virtual_environment(tool_dir)
            install_result = await self.install_requirements(tool_dir)
            if not install_result["success"]:
                return {
                    "error": f"Failed to install requirements for {tool_dir.name}",
                    "install_details": install_result
                }
            
            # 获取Python可执行文件
            python_exe = self.get_python_executable(tool_dir)
            
            # 使用绝对路径调用工具加载脚本
            current_file = Path(__file__).resolve()  # tools/tool_env_manager.py
            project_root = current_file.parent.parent  # 回到项目根目录
            script_path = project_root / "tools" / "tool_loader_script.py"
            
            if not script_path.exists():
                return {"error": f"Tool loader script not found: {script_path}"}
            
            print(f"[DEBUG] Executing tool loading script for {tool_dir.name}...")
            print(f"[DEBUG] Command: {python_exe} {script_path} {tool_file} {tool_dir.name}")
            
            # 使用异步子进程执行脚本
            process = await asyncio.create_subprocess_exec(
                str(python_exe), str(script_path), str(tool_file), tool_dir.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 等待进程结束并收集输出
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
            
            # 解码输出
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
            
            if stderr_str:
                print(f"[{tool_dir.name}] STDERR: {stderr_str}")
            
            if process.returncode != 0:
                logger.error(f"Script execution failed with return code: {process.returncode}")
                logger.error(f"Script stderr: {stderr_str}")
                return {
                    "error": f"Script execution failed: {stderr_str}",
                    "stdout": stdout_str
                }
            
            # 解析结果
            try:
                print(f"[DEBUG] Raw stdout from {tool_dir.name}: {stdout_str[:200]}...")  # 打印前200字符
                output = json.loads(stdout_str)
                return output
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}, raw_output: {stdout_str}")
                return {
                    "error": f"Failed to parse JSON output: {e}",
                    "raw_output": stdout_str
                }

        except asyncio.TimeoutError:
            return {"error": f"Timeout loading tools from {tool_dir.name}"}
        except Exception as e:
            return {"error": f"Error loading tools from {tool_dir.name}: {str(e)}"}
    
    async def load_all_tools(self, request_tool_dir: Optional[str] = None) -> Dict[str, Any]:
        """异步加载所有工具目录中的工具"""
        all_tools = {}
        errors = []
        cache_hits = 0
        cache_misses = 0
        
        for tool_dir in self.get_tool_directories():
            if request_tool_dir is not None and request_tool_dir != tool_dir.name:
                continue
            logger.info(f"Processing tools from {tool_dir.name}")
            # 检查缓存
            if self.is_tool_cache_valid(tool_dir):
                cached_data = self.tools_cache[tool_dir.name]
                cached_tools = cached_data["tools"]
                
                # 将缓存的工具添加到结果中
                for tool_metadata in cached_tools:
                    tool_name = tool_metadata["name"]
                    all_tools[tool_name] = tool_metadata
                
                cache_hits += 1
                logger.info(f"Cache HIT: Loaded {len(cached_tools)} tools from cache for {tool_dir.name}")
                continue

            # 缓存无效，重新加载
            cache_misses += 1
            logger.info(f"Cache MISS: Loading tools from environment for {tool_dir.name}")
            
            result = await self.load_tools_from_environment(tool_dir)
            
            if "error" in result:
                errors.append({
                    "tool_dir": tool_dir.name,
                    "error": result["error"]
                })
                logger.error(f"Error loading tools from {tool_dir.name}: {result['error']}")
            elif "tools" in result:
                for tool_metadata in result["tools"]:
                    tool_name = tool_metadata["name"]
                    all_tools[tool_name] = tool_metadata
                
                logger.info(f"Loaded {len(result['tools'])} tools from environment for {tool_dir.name}")
                # 更新缓存
                self.update_tool_cache(tool_dir, result["tools"])
        
        logger.info(f"Tool loading summary: {cache_hits} cache hits, {cache_misses} cache misses")
        
        return {
            "tools": all_tools,
            "errors": errors,
            "total_tools": len(all_tools),
            "total_errors": len(errors),
            "cache_stats": {
                "cache_hits": cache_hits,
                "cache_misses": cache_misses
            }
        }
    
    def get_tool_environment_info(self, tool_dir: Path) -> Dict[str, Any]:
        """获取工具环境信息"""
        info = {
            "name": tool_dir.name,
            "path": str(tool_dir.absolute()),
            "has_venv": (tool_dir / "venv").exists(),
            "has_requirements": (tool_dir / "requirements.txt").exists(),
            "has_tool_py": (tool_dir / "tool.py").exists(),
            "venv_path": str((tool_dir / "venv").absolute()),
            "requirements_path": str((tool_dir / "requirements.txt").absolute()),
            "tool_entry_point": str((tool_dir / "tool.py").absolute())
        }
        
        if info["has_venv"]:
            try:
                python_exe = self.get_python_executable(tool_dir)
                info["python_executable"] = str(python_exe)
                info["venv_valid"] = python_exe.exists()
                
                # 获取pip路径
                venv_path = tool_dir / "venv"
                if os.name == 'nt':
                    pip_exe = venv_path / "Scripts" / "pip.exe"
                    scripts_dir = venv_path / "Scripts"
                else:
                    pip_exe = venv_path / "bin" / "pip"
                    scripts_dir = venv_path / "bin"
                
                info["pip_executable"] = str(pip_exe)
                info["scripts_directory"] = str(scripts_dir)
                info["pip_exists"] = pip_exe.exists()
                
                # 检查pip版本
                if pip_exe.exists():
                    try:
                        result = subprocess.run([
                            str(python_exe), "-m", "pip", "--version"
                        ], capture_output=True, text=True, timeout=10)
                        
                        if result.returncode == 0:
                            info["pip_version"] = result.stdout.strip()
                        else:
                            info["pip_version"] = f"Error: {result.stderr.strip()}"
                    except Exception as e:
                        info["pip_version"] = f"Error checking version: {str(e)}"
                
                # 检查已安装的包
                try:
                    result = subprocess.run([
                        str(python_exe), "-m", "pip", "list", "--format=json"
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        import json
                        packages = json.loads(result.stdout)
                        info["installed_packages"] = packages
                        info["packages_count"] = len(packages)
                    else:
                        info["installed_packages"] = []
                        info["packages_count"] = 0
                        info["packages_error"] = result.stderr.strip()
                except Exception as e:
                    info["installed_packages"] = []
                    info["packages_count"] = 0
                    info["packages_error"] = str(e)
                
            except Exception as e:
                info["venv_error"] = str(e)
                info["venv_valid"] = False
        
        # 读取requirements.txt内容
        if info["has_requirements"]:
            try:
                with open(tool_dir / "requirements.txt", 'r', encoding='utf-8') as f:
                    requirements_content = f.read().strip()
                    requirements_lines = [
                        line.strip() for line in requirements_content.split('\n') 
                        if line.strip() and not line.strip().startswith('#')
                    ]
                    info["requirements_content"] = requirements_content
                    info["requirements_list"] = requirements_lines
                    info["requirements_count"] = len(requirements_lines)
            except Exception as e:
                info["requirements_error"] = str(e)
        
        # 检查tool.py文件的大小和修改时间
        if info["has_tool_py"]:
            try:
                tool_py_path = tool_dir / "tool.py"
                stat = tool_py_path.stat()
                info["tool_py_size"] = stat.st_size
                info["tool_py_modified"] = time.ctime(stat.st_mtime)
                info["tool_py_modified_timestamp"] = stat.st_mtime
            except Exception as e:
                info["tool_py_error"] = str(e)
        
        return info
    
    def get_all_environments_info(self) -> List[Dict[str, Any]]:
        """获取所有工具环境信息"""
        envs_info = []
        for tool_dir in self.get_tool_directories():
            envs_info.append(self.get_tool_environment_info(tool_dir))
        return envs_info
    
    def cleanup_environment(self, tool_dir: Path) -> bool:
        """清理工具环境"""
        try:
            venv_path = tool_dir / "venv"
            if venv_path.exists():
                shutil.rmtree(venv_path)
                logger.info(f"Cleaned up environment for {tool_dir.name}")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up environment for {tool_dir.name}: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        cache_info = {
            "total_cached_tools": len(self.tools_cache),
            "cached_tools": {}
        }
        
        for tool_name, cache_data in self.tools_cache.items():
            cache_info["cached_tools"][tool_name] = {
                "tools_count": len(cache_data.get("tools", [])),
                "last_loaded": time.ctime(cache_data.get("last_loaded", 0)),
                "cached_files": list(cache_data.get("file_mtimes", {}).keys())
            }
        
        logger.debug(f"Cache info requested: {len(self.tools_cache)} tools in cache")
        return cache_info
    
    def clear_cache(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """清理缓存"""
        if tool_name:
            # 清理特定工具的缓存
            if tool_name in self.tools_cache:
                del self.tools_cache[tool_name]
                logger.info(f"Cleared cache for tool: {tool_name}")
                return {"status": "success", "message": f"Cache cleared for {tool_name}"}
            else:
                logger.warning(f"No cache found for tool: {tool_name}")
                return {"status": "warning", "message": f"No cache found for {tool_name}"}
        else:
            # 清理所有缓存
            cache_count = len(self.tools_cache)
            self.tools_cache.clear()
            logger.info(f"Cleared all cache ({cache_count} tools)")
            return {"status": "success", "message": f"Cleared cache for {cache_count} tools"}
    
    def invalidate_tool_cache(self, tool_name: str) -> bool:
        """手动失效特定工具的缓存"""
        if tool_name in self.tools_cache:
            del self.tools_cache[tool_name]
            logger.info(f"Invalidated cache for tool: {tool_name}")
            return True
        else:
            logger.warning(f"No cache found to invalidate for tool: {tool_name}")
            return False
