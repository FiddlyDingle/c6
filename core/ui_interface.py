from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class UIInterface(ABC):
    """
    Abstract base class for all UI plugins.
    Defines the contract between the core AI system and user interfaces.
    """
    
    def __init__(self, agent, config: Dict[str, Any]):
        self.agent = agent
        self.config = config
        self.running = False
    
    @abstractmethod
    def start(self) -> None:
        """Initialize and start the UI interface"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the UI interface and cleanup"""
        pass
    
    @abstractmethod
    def display_message(self, message: str, message_type: str = "info") -> None:
        """
        Display a message to the user
        message_type: "info", "error", "warning", "success", "thinking"
        """
        pass
    
    @abstractmethod
    def get_user_input(self) -> str:
        """Get input from the user"""
        pass
    
    @abstractmethod
    def display_tool_execution(self, tool_name: str, status: str, result: Any = None) -> None:
        """
        Show tool execution status
        status: "starting", "running", "completed", "error"
        """
        pass
    
    def format_response(self, response: str) -> str:
        """Format AI response for display (can be overridden)"""
        return response
    
    def show_thinking(self, thinking: str) -> None:
        """Show AI reasoning process (optional override)"""
        self.display_message(f"Thinking: {thinking}", "thinking")
