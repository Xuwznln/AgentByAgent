#!/usr/bin/env python3
"""
General Purpose Agent - Tool Calling Version
Uses Claude's native tool calling API with tools from FastMCP client

This Agent can:
1. Fetch tool definitions from FastMCP client
2. Use Claude's native tool calling API
3. Execute tools manually and return results
4. Provide intelligent problem-solving solutions
"""

import asyncio
import json
import traceback
import uuid
from typing import Dict, Any, List
from datetime import datetime

from anthropic.types.raw_message_delta_event import Delta
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp import Tool
from anthropic import Anthropic, BetaInputJsonEvent
from anthropic.types import MessageParam, StopReason, ToolParam
from anthropic import BetaMessageStream, BetaMessageStreamEvent, BetaTextEvent
from anthropic.lib.streaming._beta_types import BetaThinkingEvent, BetaSignatureEvent
from anthropic.types.beta import BetaRawMessageStartEvent, BetaRawMessageDeltaEvent, BetaRawContentBlockDeltaEvent, \
    BetaRawMessageStopEvent, BetaRawContentBlockStartEvent, BetaRawContentBlockStopEvent, BetaThinkingDelta, \
    BetaTextDelta, BetaInputJSONDelta
from rich.console import Console
from rich.panel import Panel

console = Console()


class TodoManager:
    """Todo管理器"""
    
    def __init__(self):
        self.todos: List[Dict[str, Any]] = []
        self.next_id = 1
    
    def add_todo(self, content: str, priority: str = "medium") -> Dict[str, Any]:
        """添加新的todo项"""
        todo = {
            "id": self.next_id,
            "content": content,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.todos.append(todo)
        self.next_id += 1
        return todo
    
    def finish_todo(self, todo_id: int) -> Dict[str, Any]:
        """完成todo项"""
        for todo in self.todos:
            if todo["id"] == todo_id:
                todo["status"] = "completed"
                todo["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return todo
        raise ValueError(f"Todo项 #{todo_id} 不存在")
    
    def get_todos(self) -> List[Dict[str, Any]]:
        """获取所有todo项"""
        return self.todos.copy()
    
    def get_pending_todos(self) -> List[Dict[str, Any]]:
        """获取未完成的todo项"""
        return [todo for todo in self.todos if todo["status"] == "pending"]


class ToolCallingAgent:
    """Tool Calling Agent using Claude's native API"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize Agent"""
        self.config = self._load_config(config_path)
        self.anthropic_client = Anthropic(
            api_key=self.config.get("agent", {}).get("api_key"),
            base_url=self.config.get("agent", {}).get("base_url")
        )
        console.print("✅ Anthropic client initialized successfully")
            
        self.conversation_history: List[MessageParam] = []
        self.session_id = str(uuid.uuid4())[:5]
        self.answer_times = 0
        self.tools: List[ToolParam] = []
        self.mcp_client = Client("http://localhost:3002/mcp/")
        console.print("✅ FastMCP client initialized successfully")
        self.mcp_tools: List[Tool] = []
        
        # 初始化Todo管理器
        self.todo_manager = TodoManager()
        
    async def async_init(self):
        """Async initialization for MCP client and tools"""
        await self.mcp_client._connect()
        await self._load_tools_from_mcp()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]Failed to load config file: {e}[/red]")
            return {}

    async def _load_tools_from_mcp(self):
        """Load tool definitions from FastMCP client"""
        try:
            self.mcp_tools = await self.mcp_client.list_tools()
            self.tools = []
            for tool in self.mcp_tools:
                claude_tool = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                self.tools.append(claude_tool)  # type: ignore
            
            console.print(f"[green]✅ Successfully loaded {len(self.tools)} tools[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to load tools from MCP: {e}[/red]")
            console.print("[yellow]⚠️ Continuing without tools...[/yellow]")
    
    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str | list[Dict[str, Any]]:
        """Execute a tool via FastMCP client"""
        try:
            console.print(f"[yellow]🔧 Executing tool: {tool_name}[/yellow]")
            
            # 处理内置Todo工具
            if tool_name == "todo_write":
                return await self._execute_todo_write(tool_input)
            elif tool_name == "todo_finish":
                return await self._execute_todo_finish(tool_input)
            elif tool_name == "todo_read":
                return await self._execute_todo_read(tool_input)
            
            # 处理MCP工具
            result: CallToolResult = await self.mcp_client.call_tool(tool_name, tool_input)
            result_content = []
            for i in result.content:
                result_content.append(i.model_dump(by_alias=True, mode="json", exclude_none=True))
            console.print(f"[green]✅ {tool_name} execution completed, result: {str(result_content)[:100]}[/green]")
            return result_content
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Tool execution error: {str(e)}"
            console.print(f"[red]❌ {error_msg}[/red]")
            return f"Tool execution error: {str(e)} \n {traceback.format_exc()}"
    
    async def _execute_todo_write(self, tool_input: Dict[str, Any]) -> str:
        """执行todo_write工具"""
        try:
            content = tool_input.get("content", "")
            priority = tool_input.get("priority", "medium")
            
            if not content:
                raise ValueError("Todo内容不能为空")
            
            todo = self.todo_manager.add_todo(content, priority)
            result_data = f"✅ 成功添加Todo项 #{todo['id']}: {content} (优先级: {priority})"
            
            console.print(f"[green]{result_data}[/green]")
            return result_data
            
        except Exception as e:
            error_msg = f"todo_write错误: {str(e)}"
            return error_msg
    
    async def _execute_todo_finish(self, tool_input: Dict[str, Any]) -> str:
        """执行todo_finish工具"""
        try:
            todo_id = tool_input.get("id")
            
            if todo_id is None:
                raise ValueError("请提供要完成的Todo编号")
            
            try:
                todo_id = int(todo_id)
            except ValueError:
                raise ValueError("Todo编号必须是数字")
            
            if todo_id < 1:
                raise ValueError("Todo编号必须从1开始")
            
            todo = self.todo_manager.finish_todo(todo_id)
            result_data = f"✅ Todo项 #{todo['id']} 已完成: {todo['content']}"
            
            console.print(f"[green]{result_data}[/green]")
            return result_data
            
        except Exception as e:
            error_msg = f"todo_finish错误: {str(e)}"
            return error_msg
    
    async def _execute_todo_read(self, tool_input: Dict[str, Any]) -> str:
        """执行todo_read工具"""
        try:
            show_all = tool_input.get("show_all", False)
            
            if show_all:
                todos = self.todo_manager.get_todos()
                title = "所有Todo项"
            else:
                todos = self.todo_manager.get_pending_todos()
                title = "待完成的Todo项"
            
            if not todos:
                result_data = f"📝 {title}: 无"
            else:
                result_lines = [f"📝 {title}:"]
                for todo in todos:
                    status_emoji = "✅" if todo["status"] == "completed" else "⏳"
                    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(todo["priority"], "⚪")
                    result_lines.append(f"  {status_emoji} #{todo['id']} {priority_emoji} {todo['content']}")
                    if todo["status"] == "completed" and "completed_at" in todo:
                        result_lines.append(f"    完成时间: {todo['completed_at']}")
                
                result_data = "\n".join(result_lines)
            
            console.print(f"[cyan]{result_data}[/cyan]")
            return result_data
            
        except Exception as e:
            error_msg = f"todo_read错误: {str(e)}"
            return error_msg
    
    def _build_system_prompt(self, user_question: str) -> str:
        """Build system prompt"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        with open("system_prompt.md", "r", encoding="utf-8") as f:
            system_prompt = f.read()
        return system_prompt + f"\n\n**Current session id**: {self.session_id}\n**Current answer times**: {self.answer_times}\n**Current time**: {current_time}\n**User question**: {user_question}"

    async def process_message_with_tool_calling(self, user_message: str) -> str:
        """Process user message using Claude's tool calling API"""
        try:
            console.print(Panel(f"📝 User Message: {user_message}", style="blue"))
            
            # Add current user message to conversation
            user_message = f"Help me solve this problem: {user_message}"
            current_messages = self.conversation_history.copy()
            current_messages.append({
                "role": "user",
                "content": user_message
            })
            
            console.print("[yellow]🤖 Processing with tool calling...[/yellow]")
            
            # Keep track of the conversation for this turn
            max_turns = 50  # Prevent infinite loops
            turn_count = 0
            response_content = ""  # 初始化response_content
            
            while turn_count < max_turns:
                turn_count += 1
                console.print(f"[dim]--- Turn {turn_count} ---[/dim]")
                
                # Make API call to Claude with streaming
                try:
                    if not self.anthropic_client:
                        raise Exception("Anthropic client not initialized")
                    await self._load_tools_from_mcp()
                    # Create the request parameters
                    create_params = {"model": self.config.get("agent", {}).get("model"),
                                     "max_tokens": 64000, "messages": current_messages,
                                     "system": self._build_system_prompt(user_message), "tools": self.tools.copy()}
                    
                    # 添加内置工具
                    # noinspection PyTypeChecker
                    builtin_tools = [
                        {
                            "type": "web_search_20250305",
                            "name": "web_search",
                            "max_uses": 5
                        },
                        {
                            "name": "todo_write",
                            "description": "Add new todo item",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "Todo content"
                                    },
                                    "priority": {
                                        "type": "string",
                                        "enum": ["high", "medium", "low"],
                                        "description": "Todo priority",
                                        "default": "medium"
                                    }
                                },
                                "required": ["content"]
                            }
                        },
                        {
                            "name": "todo_finish",
                            "description": "Mark todo item as completed",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "integer",
                                        "description": "Todo item ID",
                                        "minimum": 1
                                    }
                                },
                                "required": ["id"]
                            }
                        },
                        {
                            "name": "todo_read",
                            "description": "Read todo list",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "show_all": {
                                        "type": "boolean",
                                        "description": "Show all todos including completed",
                                        "default": False
                                    }
                                },
                                "required": []
                            }
                        }
                    ]
                    create_params["tools"].extend(builtin_tools)
                    
                    # Use streaming API
                    assistant_response_content = []
                    current_text = ""
                    tool_calls = []
                    
                    with self.anthropic_client.beta.messages.stream(**create_params) as stream:
                        self.answer_times += 1
                        is_thinking = False
                        is_json = False
                        stream: BetaMessageStream
                        event: BetaMessageStreamEvent
                        for event in stream:
                            if isinstance(event, (BetaThinkingEvent, BetaTextEvent, BetaRawMessageDeltaEvent, BetaRawContentBlockDeltaEvent)):
                                if isinstance(event, BetaThinkingEvent):
                                    is_thinking = True
                                    text = event.thinking
                                    # Thinking过程用灰色显示
                                    console.print(f"[dim bright_black]{text}[/dim bright_black]", end="")
                                elif isinstance(event, BetaTextEvent):
                                    if is_thinking:
                                        console.print()
                                        is_thinking = False
                                    text = event.text
                                    # 普通文本用默认颜色
                                    console.print(text, end="")
                                    current_text += text
                                elif isinstance(event, BetaRawMessageDeltaEvent):
                                    delta = event.delta
                                    if delta.stop_reason:
                                        pass
                                        # # 检查stop_reason是否为tool_use
                                        # if delta.stop_reason == "tool_use":
                                        #     console.print(f"[cyan]🛠️ Claude wants to use tools[/cyan]")
                                elif isinstance(event, BetaRawContentBlockDeltaEvent):
                                    delta = event.delta
                                    if isinstance(delta, BetaThinkingDelta):
                                        continue
                                    elif isinstance(delta, BetaTextDelta):  # 由BetaThinkingEvent+BetaTextEvent接管
                                        continue
                                    if isinstance(delta, BetaInputJSONDelta):
                                        console.print(f"[dim]{delta.partial_json}[/dim]", end="")
                                        # current_text += text  # 输入的json是否应该记录？似乎不应该
                                    elif isinstance(delta, Delta):
                                        if delta.stop_reason == StopReason:
                                            console.print(f"[cyan]🛠️ Claude wants to use tools[/cyan]")
                                    else:
                                        continue
                            elif isinstance(event, BetaRawContentBlockStartEvent):  # 段落开始
                                if event.content_block.type == "tool_use":
                                    console.print(f"[dim]-----{event.content_block.name} Start-----[/dim]")
                            elif isinstance(event, BetaRawContentBlockStopEvent):
                                if event.content_block.type == "text":
                                    # 一整段完成
                                    assistant_response_content.append(event.content_block)
                                elif event.content_block.type == "tool_use":
                                    tool_calls.append(event.content_block)
                                    console.print(f"\n[dim]-----{event.content_block.name} End-----[/dim]", end="")
                                    assistant_response_content.append(event.content_block)
                                else:
                                    pass
                                console.print()
                            elif isinstance(event, BetaRawMessageStopEvent):
                                pass  # 对BetaRawContentBlockStopEvent的汇总，content是个list
                            elif isinstance(event, (BetaInputJsonEvent, BetaSignatureEvent, BetaRawMessageStartEvent)):
                                continue
                            else:
                                pass
                    console.print()  # New line after streaming
                    
                    # 保存当前回合的文本响应
                    response_content += current_text
                    
                    # Add Claude's response to messages
                    if assistant_response_content:
                        current_messages.append({
                            "role": "assistant",
                            "content": assistant_response_content
                        })
                    
                    # Process tool calls if any
                    if tool_calls:
                        tool_results = []
                        for tool_call in tool_calls:
                            tool_name = tool_call.name
                            tool_input = tool_call.input
                            tool_use_id = tool_call.id
                            tool_result = await self._execute_tool(tool_name, tool_input)
                            # Format the tool result

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": tool_result
                            })
                        
                        # Add tool results to messages
                        if tool_results:
                            current_messages.append({
                                "role": "user",
                                "content": tool_results
                            })
                    else:
                        # No tool calls, conversation turn is complete
                        break
                        
                except Exception as e:
                    traceback.print_exc()
                    break
            
            if turn_count >= max_turns:
                response_content += "\n\n[Note: Reached maximum conversation turns]"
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_content})
            
            # Keep history length within reasonable range
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return response_content
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Tool calling message processing failed: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg


# ================================
# Test Scenarios (same as original)
# ================================

class TestScenarios:
    """Test Scenario Manager"""
    
    @staticmethod
    def get_youtube_scenario() -> str:
        """YouTube video analysis scenario"""
        return """In the YouTube 360 VR video from March 2018 narrated by the voice actor of Lord of the Rings' Gollum, what number was mentioned by the narrator directly after dinosaurs were first shown in the video?"""
    
    @staticmethod
    def get_stock_scenario() -> str:
        """Stock price query scenario"""
        return """What is the latest stock price of NVIDIA (NVDA)?"""
    
    @staticmethod
    def get_tools_list_scenario() -> str:
        """Tool calling scenario"""
        return """Please list all available tools you have now. And create another environment function as a calculator. And test it."""
    
    @staticmethod
    def get_todo_scenario() -> str:
        """Todo管理测试场景"""
        return """请帮我测试todo管理功能：
1. 添加3个不同优先级的todo项
2. 查看当前的todo列表
3. 完成其中一个todo项
4. 再次查看todo列表（包含已完成的项目）"""


# ================================
# Main Program
# ================================

async def main():
    """Main program"""
    console.print(Panel.fit("🤖 Tool Calling Agent System", style="bold cyan"))
    
    try:
        # Initialize Agent
        agent = ToolCallingAgent()
        await agent.async_init()
        
        console.print("\n" + "="*60)
        console.print("🎯 Select Test Mode:")
        console.print("1. YouTube Video Analysis (Tool Calling)")
        console.print("2. Stock Price Query (Tool Calling)") 
        console.print("3. List Available Tools (Tool Calling)")
        console.print("4. Direct Chat (Tool Calling)")
        console.print("5. Show Available Tools")
        console.print("6. Exit")
        console.print("="*60)
        
        # Initial mode selection
        choice = input("\nPlease select mode (1-6): ").strip()

        if choice == "1":
            message = TestScenarios.get_youtube_scenario()
            console.print(f"\n[yellow]📺 YouTube Analysis Scenario (Tool Calling)[/yellow]")
            await agent.process_message_with_tool_calling(message)
            
        elif choice == "2":
            message = TestScenarios.get_stock_scenario()
            console.print(f"\n[yellow]📈 Stock Query Scenario (Tool Calling)[/yellow]")
            await agent.process_message_with_tool_calling(message)
            
        elif choice == "3":
            message = TestScenarios.get_tools_list_scenario()
            console.print(f"\n[yellow]🛠️ Tools List Scenario (Tool Calling)[/yellow]")
            await agent.process_message_with_tool_calling(message)
            
        elif choice == "4":
            console.print(f"\n[yellow]💬 Direct Chat Mode (Tool Calling)[/yellow]")
            console.print("[cyan]You can start chatting directly. Type 'quit' to exit.[/cyan]")
            
            # 直接进入对话循环，不需要预设消息
            while True:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() == 'quit':
                    console.print("[green]👋 Thank you for using![/green]")
                    break
                if user_input:  # 只有当用户输入不为空时才处理
                    await agent.process_message_with_tool_calling(user_input)
            return  # 直接返回，不进入下面的连续对话模式
            
        elif choice == "5":
            console.print(f"\n[yellow]🛠️ Available Tools List[/yellow]")
            tools = await agent.mcp_client.list_tools()
            console.print(f"[green]✅ Successfully loaded {len(tools)} tools[/green]")
            for i, tool in enumerate(tools, 1):
                console.print(f"{i}. {tool.name}: {tool.description}")
            return
            
        elif choice == "6":
            console.print("[green]👋 Goodbye![/green]")
            return

        # Continuous conversation loop (for modes 1, 2, 3)
        if choice in ["1", "2", "3"]:
            console.print("\n" + "="*60)
            console.print("💬 Continuous Conversation Mode")
            console.print("[cyan]Type your message to continue the conversation, or 'quit' to exit[/cyan]")
            console.print("="*60)
            
            while True:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() == 'quit':
                    console.print("[green]👋 Thank you for using![/green]")
                    break
                if user_input:  # 只有当用户输入不为空时才处理
                    await agent.process_message_with_tool_calling(user_input)
                    
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Program interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Program execution error: {e}[/red]")

if __name__ == "__main__":
    asyncio.run(main()) 