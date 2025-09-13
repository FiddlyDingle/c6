#!/usr/bin/env python3
"""
Cerb AI Assistant - Enhanced Main Entry Point
Loads configuration, initializes the enhanced agent, and starts the selected UI
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Try to import enhanced agent, fall back to original
try:
    from core.enhanced_agent import EnhancedAgent as Agent
    print("Using enhanced agent with plugin support")
except ImportError:
    from core.agent import Agent
    print("Using original agent (enhanced features not available)")


def setup_logging(config: Dict[str, Any]) -> None:
    """Configure enhanced logging system"""
    log_config = config.get('logging', {})
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('file', 'logs/agent.log')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging with enhanced format
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config() -> Dict[str, Any]:
    """Load configuration from settings.json"""
    config_path = Path(__file__).parent / 'config' / 'settings.json'
    
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and add missing config keys for backward compatibility"""
    # Add missing sections with defaults
    if 'plugins' not in config:
        config['plugins'] = {
            "enabled": True,
            "auto_load": True,
            "plugin_directories": ["plugins"],
            "max_plugins": 20
        }
    
    if 'features' not in config:
        config['features'] = {
            "auto_save_conversations": True,
            "context_window_size": 10,
            "enable_tool_chaining": True,
            "enable_plugin_hooks": True
        }
    
    # Enhance existing sections
    if 'timeout' not in config.get('lm_studio', {}):
        config['lm_studio']['timeout'] = 30
    if 'max_retries' not in config.get('lm_studio', {}):
        config['lm_studio']['max_retries'] = 3
    
    if 'max_parallel_tools' not in config.get('tools', {}):
        config['tools']['max_parallel_tools'] = 3
    if 'categories_enabled' not in config.get('tools', {}):
        config['tools']['categories_enabled'] = ["file_ops", "web_ops", "system_ops"]
    
    return config


def load_ui_plugin(plugin_name: str, agent: Agent, config: Dict[str, Any]):
    """Load and initialize the specified UI plugin"""
    try:
        if plugin_name == 'terminal_ui':
            from ui_plugins.terminal_ui import TerminalUI
            return TerminalUI(agent, config)
        else:
            raise ImportError(f"Unknown UI plugin: {plugin_name}")
            
    except ImportError as e:
        print(f"Error loading UI plugin '{plugin_name}': {e}")
        print("Available UI plugins: terminal_ui")
        sys.exit(1)


def main():
    """Enhanced main entry point"""
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config()
        
        # Validate and enhance config for compatibility
        config = validate_config(config)
        
        # Setup logging
        setup_logging(config)
        logger = logging.getLogger(__name__)
        logger.info("Starting Cerb AI Assistant - Enhanced Version")
        
        # Initialize the agent
        print("Initializing AI agent...")
        agent = Agent(config)
        logger.info("Agent initialized successfully")
        
        # Show system status
        if hasattr(agent, 'plugin_manager'):
            plugins = agent.plugin_manager.get_loaded_plugins()
            if plugins:
                print(f"Loaded {len(plugins)} plugins:")
                for plugin in plugins:
                    print(f"  - {plugin['name']} v{plugin['version']}")
        
        tools = agent.tools.get_available_tools()
        if tools:
            categories = set(tool.get('category', 'general') for tool in tools)
            print(f"Available tools: {len(tools)} tools in {len(categories)} categories")
        
        # Load UI plugin
        ui_plugin_name = config.get('ui_plugin', 'terminal_ui')
        print(f"Loading UI plugin: {ui_plugin_name}")
        ui = load_ui_plugin(ui_plugin_name, agent, config)
        
        # Start the interface
        print("Starting user interface...")
        print("Enhanced features: Plugin support, improved error handling, tool discovery")
        logger.info(f"Starting UI plugin: {ui_plugin_name}")
        ui.start()
        
        logger.info("Cerb AI Assistant shutting down")
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.exception("Fatal error occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
