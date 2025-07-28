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
from typing import Dict, Any, List

import anthropic
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
        return """You are an intelligent AI assistant with the following core capabilities:

## 🎯 Core Responsibilities
Your mission is to understand and solve various problems posed by users by analyzing task requirements and using available tools multiple times to provide optimal solutions.

You need to build new tools:
1.  **Understand task requirements**: Analyze tasks and determine if new capabilities/tools are needed to complete them.
2.  **Search for solutions**: Search for relevant libraries or APIs in the open source world (like GitHub) to implement required functionality.
3.  **Learn and integrate**: Read documentation or code examples, learn how to use found libraries/APIs, and dynamically generate code to call them, thus "creating" new tools.
4.  **Execute tasks**: Use newly created tools to solve problems.

## 🛠️ Tool Usage Principles
- Carefully analyze user requirements and determine the capabilities needed to solve the problem
- Reasonably select and use available tools to complete tasks
- Analyze and verify tool execution results
- Provide clear and accurate final answers

## 🔄 Workflow
1. **Requirement Understanding**: Carefully analyze user questions and understand real needs
2. **Solution Planning**: Develop strategies and steps to solve problems
3. **Tool Execution**: Reasonably use available tools to complete each step
4. **Result Integration**: Integrate tool execution results into complete answers
5. **Quality Verification**: Ensure accuracy and completeness of answers

## 📝 Response Principles
- Provide accurate and useful information
- Maintain professional and objective attitude
- Acknowledge uncertainty, do not fabricate information
- Prioritize reliable information sources and methods

Now please assist users in solving problems."""

    async def process_message(self, user_message: str) -> str:
        """Process user message (using streaming response)"""
        try:
            console.print(Panel(f"📝 User Message: {user_message}", style="blue"))
            
            # Build message history - use correct message format
            messages = []
            
            # Add conversation history
            for msg in self.conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Ensure client is initialized
            if not self.anthropic_client:
                raise ValueError("Anthropic client not initialized")
            
            # Use streaming API call
            console.print("[yellow]🤖 AI thinking...[/yellow]")
            result = ""
            
            # Create streaming response
            with self.anthropic_client.messages.stream(
                model=self.config.get("agent", {}).get("model"),
                max_tokens=32000,
                messages=messages,
                system=self.system_prompt
            ) as stream:
                console.print("[green]📡 Starting to receive response...[/green]")
                
                for text in stream.text_stream:
                    result += text
                    # Display response content in real time
                    console.print(text, end="")
                
                console.print()  # New line
            
            # Save conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": result})
            
            # Keep history length within reasonable range
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to process message: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

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
            
            # Add current user message
            messages.append({
                "role": "user", 
                "content": user_message
            })
            
            # Ensure client is initialized
            if not self.anthropic_client:
                raise ValueError("Anthropic client not initialized")
            
            console.print("[yellow]🤖 Processing with MCP server...[/yellow]")
            
            # Use MCP server streaming API
            result = ""
            with self.anthropic_client.beta.messages.stream(
                model=self.config.get("agent", {}).get("model"),
                max_tokens=32000,
                messages=messages,
                mcp_servers=[
                    {
                        "type": "url", 
                        "url": "https://mcp.wznln.com/mcp/",
                        "name": "tools-server",
                    }
                ],
                extra_headers={
                    "anthropic-beta": "mcp-client-2025-04-04"
                }
            ) as stream:
                console.print("[green]📡 Starting to receive MCP response...[/green]")
                
                for text in stream.text_stream:
                    result += text
                    console.print(text, end="")
                
                console.print()  # New line
            
            # Save conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": result})
            
            # Keep history length within reasonable range
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return result
            
        except Exception as e:
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
        return """In the YouTube 360 VR video from March 2018 narrated by the voice actor of Lord of the Rings' Gollum, what number was mentioned by the narrator directly after dinosaurs were first shown in the video?

Please help me find this answer."""
    
    @staticmethod
    def get_stock_scenario() -> str:
        """Stock price query scenario"""
        return """What is the latest stock price of NVIDIA (NVDA)?"""
    
    @staticmethod
    def get_tools_list_scenario() -> str:
        """MCP tools list scenario"""
        return """Please list all available MCP tools you have now. And select two of them to use."""

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
        console.print("4. Exit")
        console.print("="*60)
        
        while True:
            choice = input("\nPlease select mode (1-4): ").strip()
            
            if choice == "1":
                message = TestScenarios.get_youtube_scenario()
                console.print(f"\n[yellow]📺 YouTube Analysis Scenario (MCP)[/yellow]")
                result = await agent.process_message_with_mcp(message)
                
            elif choice == "2":
                message = TestScenarios.get_stock_scenario()
                console.print(f"\n[yellow]📈 Stock Query Scenario (MCP)[/yellow]")
                result = await agent.process_message_with_mcp(message)
                
            elif choice == "3":
                message = TestScenarios.get_tools_list_scenario()
                console.print(f"\n[yellow]🛠️ MCP Tools List Scenario (MCP)[/yellow]")
                result = await agent.process_message_with_mcp(message)
                
            elif choice == "4":
                console.print("[green]👋 Thank you for using![/green]")
                break
                
            else:
                console.print("[red]Invalid choice, please try again[/red]")
                continue
            
            # Ask if continue
            if choice in ["1", "2", "3"]:
                if input("\nContinue testing other features? (y/n): ").strip().lower() != 'y':
                    break
    
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Program interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Program execution error: {e}[/red]")

if __name__ == "__main__":
    asyncio.run(main()) 