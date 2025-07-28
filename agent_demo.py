#!/usr/bin/env python3
"""
General Purpose Agent - Intelligent Agent System

This Agent can:
1. Connect to MCP server
2. Analyze task requirements
3. Use available tools to solve problems
4. Provide intelligent problem-solving solutions
"""

import asyncio
import json
import os
import traceback
from typing import Dict, Any, List

import anthropic
from anthropic import BetaMessageStream, BetaMessageStreamEvent, BetaTextEvent
from anthropic.lib.streaming._beta_types import BetaThinkingEvent, BetaSignatureEvent
from anthropic.types.beta import BetaRawMessageStartEvent, BetaRawMessageDeltaEvent, BetaRawContentBlockDeltaEvent, \
    BetaRawMessageStopEvent, BetaRawContentBlockStartEvent, BetaRawContentBlockStopEvent, BetaThinkingDelta, \
    BetaSignatureDelta, BetaTextDelta
from anthropic.types.beta.beta_raw_message_delta_event import Delta
from rich.console import Console
from rich.panel import Panel

console = Console()

class GeneralAgent:
    """General Purpose Intelligent Agent"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize Agent"""
        self.config = self._load_config(config_path)
        self.anthropic_client = None
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = self._build_system_prompt()
        
        # Initialize Anthropic client
        self._initialize_client()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]Failed to load config file: {e}[/red]")
            return {}
    
    def _initialize_client(self):
        """Initialize Anthropic client"""
        try:
            # Get API key from config file or environment variable
            api_key = (
                self.config.get("agent", {}).get("api_key") or 
                os.getenv("ANTHROPIC_API_KEY")
            )
            
            if not api_key:
                raise ValueError("Anthropic API key not found")
            
            # Get base_url if available
            base_url = self.config.get("agent", {}).get("base_url")
            
            if base_url:
                self.anthropic_client = anthropic.Anthropic(
                    api_key=api_key,
                    base_url=base_url
                )
            else:
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                
            console.print("✅ Anthropic client initialized successfully")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to initialize Anthropic client: {e}[/red]")
            raise
    
    def _build_system_prompt(self) -> str:
        """Build system prompt"""
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        return f"""

You are an autonomous developer agent with access to a suite of tools. For each user request, you must:

1. Identify the goal and plan a sequence of tool calls—combining them as needed—to solve the problem end‑to‑end.  
2. Persistently iterate: if the first attempt doesn’t work, retry with a different combination of tools until the problem is resolved.  
3. Use `create_tool_environment` whenever you need to scaffold a new tool module or project environment; then write and test `tool.py` code in that environment.  
4. To leverage existing solutions, first run `search_github` to look for Python libraries or code examples you can adopt directly.  
5. Always refresh your tool list via the MCP server’s auto‑refresh mechanism before beginning new work, so you have the latest capabilities.  
6. For any external data or factual lookups, call only well‑documented APIs—never scrape arbitrary web pages. Record and trace every API response (strict provenance). If a source proves unreliable or returns unexpected data, abort and reset your context before trying a different API.  
7. Log each step of your reasoning and tool usage, so it’s clear why you chose that sequence and when you decide to pivot strategies.  
8. Continue this process—environment setup, GitHub search, code generation, tool invocation, API validation—until you deliver a complete, working solution.  
If you dont have relevant tools, you can create them using `create_tool_environment` and edit tool file.
"""

    async def process_message_with_mcp(self, user_message: str) -> str:
        """Process user message (using MCP server)"""
        try:
            console.print(Panel(f"📝 User Message: {user_message}", style="blue"))
            
            # Build message history
            messages = []
            
            # Add conversation history
            for msg in self.conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            user_message = f"Help me solve this problem: {user_message}"
            # Add current user message
            messages.append({
                "role": "user", 
                "content": user_message
            })
            
            # Ensure client is initialized
            if not self.anthropic_client:
                raise ValueError("Anthropic client not initialized")
            
            console.print("[yellow]🤖 Processing ...[/yellow]")
            
            # Use MCP server streaming API
            result = ""
            with self.anthropic_client.beta.messages.stream(
                model=self.config.get("agent", {}).get("model"),
                max_tokens=64000,
                messages=messages,
                system=self.system_prompt,
                mcp_servers=[
                    {
                        "type": "url", 
                        "url": "https://mcp.wznln.com/mcp/",
                        "name": "tools-server",
                    }
                ],
                thinking={
                    "type": "enabled",
                    "budget_tokens": 16000
                },
                extra_headers={
                    "anthropic-beta": "mcp-client-2025-04-04"
                }
            ) as stream:
                console.print("[green]📡 Starting to receive MCP response...[/green]")
                # 处理普通文本流
                stream: BetaMessageStream
                event: BetaMessageStreamEvent
                is_thinking = False
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
                        else:
                            delta = event.delta
                            if isinstance(delta, BetaThinkingDelta):
                                continue
                                # text = delta.thinking
                                # # Thinking过程用灰色显示
                                # console.print(f"[dim bright_black]{text}[/dim bright_black]", end="")
                            elif isinstance(delta, BetaTextDelta):
                                continue
                                # text = delta.text
                                # # 普通文本用默认颜色
                                # console.print(text, end="")
                            elif isinstance(delta, (BetaSignatureDelta, Delta)):
                                console.print()
                                continue
                            else:
                                continue
                        result += str(text)
                    elif isinstance(event, (BetaRawMessageStartEvent, BetaRawMessageStopEvent, BetaRawContentBlockStartEvent, BetaRawContentBlockStopEvent, BetaSignatureEvent)):
                        pass
                    else:
                        pass
                
                console.print()  # New line
            
            # Save conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": result})
            
            # Keep history length within reasonable range
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return result
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"MCP message processing failed: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        console.print("✅ Conversation history cleared")

# ================================
# Test Scenarios
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
        """MCP tools list scenario"""
        return """Please list all available MCP tools you have now. And create another environment function as a calculator. And test it."""

# ================================
# Main Program
# ================================

async def main():
    """Main program"""
    console.print(Panel.fit("🤖 General Purpose Intelligent Agent System", style="bold cyan"))
    
    try:
        # Initialize Agent
        agent = GeneralAgent()
        
        console.print("\n" + "="*60)
        console.print("🎯 Select Test Mode:")
        console.print("1. YouTube Video Analysis (MCP)")
        console.print("2. Stock Price Query (MCP)") 
        console.print("3. List Available MCP Tools (MCP)")
        console.print("4. Direct Chat (MCP)")
        console.print("5. Exit")
        console.print("="*60)
        
        # Initial mode selection
        choice = input("\nPlease select mode (1-5): ").strip()
        
        if choice == "1":
            message = TestScenarios.get_youtube_scenario()
            console.print(f"\n[yellow]📺 YouTube Analysis Scenario (MCP)[/yellow]")
            await agent.process_message_with_mcp(message)
            
        elif choice == "2":
            message = TestScenarios.get_stock_scenario()
            console.print(f"\n[yellow]📈 Stock Query Scenario (MCP)[/yellow]")
            await agent.process_message_with_mcp(message)
            
        elif choice == "3":
            message = TestScenarios.get_tools_list_scenario()
            console.print(f"\n[yellow]🛠️ MCP Tools List Scenario (MCP)[/yellow]")
            await agent.process_message_with_mcp(message)
            
        elif choice == "4":
            console.print(f"\n[yellow]💬 Direct Chat Mode (MCP)[/yellow]")
            console.print("[cyan]You can start chatting directly. Type 'quit' to exit.[/cyan]")
            
        elif choice == "5":
            console.print("[green]👋 Thank you for using![/green]")
            return
            
        else:
            console.print("[red]Invalid choice, starting direct chat mode[/red]")
            console.print(f"\n[yellow]💬 Direct Chat Mode (MCP)[/yellow]")
            console.print("[cyan]You can start chatting directly. Type 'quit' to exit.[/cyan]")
        
        # Continuous conversation loop
        if choice in ["1", "2", "3", "4"] or choice not in ["5"]:
            console.print("\n" + "="*60)
            console.print("💬 Continuous Conversation Mode")
            console.print("[cyan]Type your message to continue the conversation, or 'quit' to exit[/cyan]")
            console.print("="*60)
            
            while True:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'quit':
                    console.print("[green]👋 Thank you for using![/green]")
                    break
                    
                if user_input:
                    console.print(f"\n[yellow]🤖 Processing your message...[/yellow]")
                    await agent.process_message_with_mcp(user_input)
                else:
                    console.print("[yellow]Please enter a message or 'quit' to exit[/yellow]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Program interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Program execution error: {e}[/red]")

if __name__ == "__main__":
    asyncio.run(main()) 