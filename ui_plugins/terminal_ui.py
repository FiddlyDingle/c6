import sys
import logging
from typing import Dict, Any
from core.ui_interface import UIInterface

class TerminalUI(UIInterface):
    """
    Simple terminal interface for Phase 1
    Basic command-line interaction with the agent
    """
    
    def __init__(self, agent, config: Dict[str, Any]):
        super().__init__(agent, config)
        self.logger = logging.getLogger(__name__)
        self.prompt = "Cerb> "
        
    def start(self) -> None:
        """Start the terminal interface"""
        self.running = True
        self.display_message("Cerb AI Assistant - Phase 1", "info")
        self.display_message("Type 'quit' or 'exit' to stop, 'help' for commands", "info")
        print("-" * 50)
        
        while self.running:
            try:
                user_input = self.get_user_input()
                
                if not user_input.strip():
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.stop()
                    break
                elif user_input.lower() in ['help', 'h']:
                    self._show_help()
                    continue
                elif user_input.lower().startswith('status'):
                    self._show_status()
                    continue
                
                # Process input through agent
                self.display_message("Processing...", "thinking")
                result = self.agent.process_input(user_input)
                
                if result['success']:
                    # Show tool execution if any
                    if result['tools_used']:
                        self.display_message(f"Tools used: {', '.join(result['tools_used'])}", "info")
                    
                    # Show response
                    self.display_message(result['response'], "success")
                else:
                    self.display_message(result['response'], "error")
                    
                print()  # Add spacing
                
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit properly")
            except Exception as e:
                self.display_message(f"Unexpected error: {e}", "error")
                
    def stop(self) -> None:
        """Stop the terminal interface"""
        self.running = False
        self.display_message("Goodbye!", "info")
    
    def display_message(self, message: str, message_type: str = "info") -> None:
        """Display a message to the user with color coding"""
        colors = {
            "info": "\033[94m",      # Blue
            "success": "\033[92m",   # Green
            "warning": "\033[93m",   # Yellow
            "error": "\033[91m",     # Red
            "thinking": "\033[95m",  # Magenta
        }
        
        reset_color = "\033[0m"
        color = colors.get(message_type, "")
        
        prefix_map = {
            "info": "[INFO]",
            "success": "[CERB]",
            "warning": "[WARN]",
            "error": "[ERROR]",
            "thinking": "[THINKING]"
        }
        
        prefix = prefix_map.get(message_type, "")
        print(f"{color}{prefix}{reset_color} {message}")
    
    def get_user_input(self) -> str:
        """Get input from the user"""
        try:
            return input(self.prompt)
        except EOFError:
            return "quit"
    
    def display_tool_execution(self, tool_name: str, status: str, result: Any = None) -> None:
        """Show tool execution status"""
        if status == "starting":
            self.display_message(f"Starting tool: {tool_name}", "info")
        elif status == "running":
            self.display_message(f"Running: {tool_name}...", "thinking")
        elif status == "completed":
            self.display_message(f"Completed: {tool_name}", "success")
            if result:
                self.display_message(f"Result: {str(result)[:100]}...", "info")
        elif status == "error":
            self.display_message(f"Failed: {tool_name}", "error")
            if result:
                self.display_message(f"Error: {result}", "error")
    
    def _show_help(self) -> None:
        """Show available commands"""
        help_text = """
Available Commands:
  help, h          - Show this help message
  status           - Show system status
  quit, exit, q    - Exit the program
  
You can also:
  - Ask questions
  - Request file operations (read file.txt, list directory)
  - Search for files or web content
  - Execute system commands (in safe mode)
  - Create or write files
  
Examples:
  "What is the weather like?"
  "Read the file config.txt"
  "List the current directory"
  "Search for Python files"
  "Create a file called test.txt with hello world"
        """
        print(help_text)
    
    def _show_status(self) -> None:
        """Show system status"""
        try:
            # Get available tools
            tools = self.agent.tools.get_available_tools()
            tools_enabled = len([t for t in tools if t['usage_count'] >= 0])
            
            # Get memory stats
            recent_convs = self.agent.memory.get_recent_conversations(1)
            memory_active = len(recent_convs) > 0
            
            self.display_message("=== System Status ===", "info")
            self.display_message(f"Tools available: {tools_enabled}", "info")
            self.display_message(f"Memory system: {'Active' if memory_active else 'Empty'}", "info")
            
            # Test LM Studio connection
            try:
                test_response = self.agent._call_lm_studio("Test connection")
                self.display_message("LM Studio: Connected", "success")
            except Exception as e:
                self.display_message(f"LM Studio: Disconnected ({str(e)[:50]})", "error")
                
        except Exception as e:
            self.display_message(f"Status check failed: {e}", "error")
