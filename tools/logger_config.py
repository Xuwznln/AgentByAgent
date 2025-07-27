#!/usr/bin/env python3
"""
Enhanced Logger Configuration - 增强日志配置

使用 loguru + rich 组合提供美观的彩色日志输出
支持 PyCharm 控制台颜色显示
"""

from loguru import logger
from rich.console import Console
from rich.traceback import install as install_rich_traceback
from rich.theme import Theme
from pathlib import Path
from typing import Optional

# 安装rich的traceback处理，使错误信息更美观
install_rich_traceback(show_locals=True)

# 创建自定义主题
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow", 
    "error": "red bold",
    "critical": "red on white bold",
    "success": "green bold",
    "debug": "dim blue",
    "timestamp": "dim white",
    "name": "magenta",
    "level": "bold",
    "message": "white"
})

# 创建控制台对象，自动检测终端宽度
console = Console(
    theme=custom_theme, 
    force_terminal=True, 
    width=None,
    soft_wrap=False
)

class DynamicMCPLogger:
    """动态MCP服务器专用日志器"""
    
    def __init__(self, logger_name: str = "DynamicMCP"):
        self.logger_name = logger_name
        self.console = console
        self._setup_logger()
    
    def _setup_logger(self):
        """设置loguru日志器"""
        # 移除默认处理器
        logger.remove()
        
        # 添加控制台处理器 - 使用简化的rich格式
        logger.add(
            self._rich_sink,
            level="DEBUG",
            format="{message}",  # 简化格式，由rich_sink处理
            colorize=False,      # 关闭loguru自带的颜色，使用rich
            backtrace=False,     # 简化回溯信息
            diagnose=False       # 简化诊断信息
        )
        
        # 添加文件处理器（保留详细格式用于文件记录）
        log_file = Path("logs") / "dynamic_mcp_server.log"
        log_file.parent.mkdir(exist_ok=True)
        
        logger.add(
            log_file,
            level="INFO",
            format="{time:HH:mm:ss} | {level: <7} | {message}",  # 文件日志也简化
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
    
    def _rich_sink(self, message):
        """Rich控制台输出处理器 - 简化版"""
        record = message.record
        
        # 根据日志级别选择样式和图标
        level_configs = {
            "TRACE": {"style": "dim blue", "icon": "🔍"},
            "DEBUG": {"style": "dim cyan", "icon": "🐛"}, 
            "INFO": {"style": "cyan", "icon": "ℹ️"},
            "SUCCESS": {"style": "green bold", "icon": "✅"},
            "WARNING": {"style": "yellow", "icon": "⚠️"},
            "ERROR": {"style": "red bold", "icon": "❌"},
            "CRITICAL": {"style": "red on white bold", "icon": "🚨"}
        }
        
        level = record["level"].name
        config = level_configs.get(level, {"style": "white", "icon": "📝"})
        
        # 格式化时间（更简洁）
        time_str = record["time"].strftime("%H:%M:%S")
        
        # 简化输出格式
        self.console.print(
            f"[dim white]{time_str}[/dim white] "
            f"{config['icon']} "
            f"[{config['style']}]{record['message']}[/{config['style']}]"
        )
    
    def get_logger(self, name: Optional[str] = None):
        """获取配置好的logger实例"""
        if name:
            return logger.bind(name=name)
        return logger.bind(name=self.logger_name)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        logger.bind(name=self.logger_name).info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        logger.bind(name=self.logger_name).debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        logger.bind(name=self.logger_name).warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        logger.bind(name=self.logger_name).error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        logger.bind(name=self.logger_name).critical(message, **kwargs)
    
    def success(self, message: str, **kwargs):
        """成功日志"""
        logger.bind(name=self.logger_name).success(message, **kwargs)
    
    def print_banner(self, title: str, subtitle: str = ""):
        """打印美观的横幅 - 自适应宽度"""
        # 获取终端宽度，如果获取失败则使用默认值
        try:
            width = self.console.size.width
            if width < 50:  # 最小宽度保护
                width = 70
        except:
            width = 70
            
        self.console.print("=" * width, style="cyan bold")
        self.console.print(f"{title:^{width}}", style="cyan bold")
        if subtitle:
            self.console.print(f"{subtitle:^{width}}", style="dim cyan")
        self.console.print("=" * width, style="cyan bold")
    
    def print_section(self, title: str, items: list, style: str = "green"):
        """打印分节信息"""
        self.console.print(f"\n{title}:", style=f"{style} bold")
        for item in items:
            self.console.print(f"  • {item}", style=style)
    
    def print_status(self, status: str, message: str, success: bool = True):
        """打印状态信息"""
        status_style = "green bold" if success else "red bold"
        self.console.print(f"[{status_style}]{status}[/{status_style}] {message}")

# 创建全局日志器实例
dynamic_logger = DynamicMCPLogger()