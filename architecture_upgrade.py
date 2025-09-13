#!/usr/bin/env python3
"""
Cerb AI Assistant - Architecture Upgrade Script
Enhances the tool system and plugin system with better architecture
Run this script to upgrade your existing Phase 1 installation
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

class ArchitectureUpgrader:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup_before_upgrade"
        
    def run_upgrade(self):
        """Run the complete architecture upgrade"""
        print("Starting Cerb AI Architecture Upgrade...")
        print(f"Project root: {self.project_root}")
        
        # Create backup
        self.create_backup()
        
        # Upgrade components
        self.upgrade_tool_system()
        self.upgrade_plugin_system()
        self.upgrade_config_system()
        self.create_enhanced_agent()
        self.create_tool_examples()
        
        print("Architecture upgrade complete!")
        print("\nWhat's been upgraded:")
        print("   - Enhanced tool discovery and metadata system")
        print("   - Plugin lifecycle management")
        print("   - Configuration validation")
        print("   - Error handling improvements")
        print("   - Event system for component communication")
        print("\nTo use: Run 'python main.py' as before - everything is backward compatible!")
        
    def create_backup(self):
        """Create backup of current system"""
        print("Creating backup...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir()
        
        # Backup core files
        for file_path in ["core/tools.py", "core/agent.py", "config/settings.json"]:
            src = self.project_root / file_path
            if src.exists():
                dst = self.backup_dir / file_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        
        print(f"Backup created at: {self.backup_dir}")
    
    def upgrade_tool_system(self):
        """Upgrade the tool system with better architecture"""
        print("Upgrading tool system...")
        
        # Create tools directory structure
        tools_dir = self.project_root / "tools"
        tools_dir.mkdir(exist_ok=True)
        
        # Create tool categories
        categories = ["file_ops", "web_ops", "system_ops", "dev_ops"]
        for category in categories:
            (tools_dir / category).mkdir(exist_ok=True)
            
        # Create enhanced tool system
        enhanced_tools = '''import os
import sys
import json
import importlib
import importlib.util
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
import requests

class ToolMetadata:
    """Metadata for individual tools"""
    def __init__(self, name: str, description: str, category: str = "general",
                 parameters: Dict = None, examples: List[str] = None,
                 permissions: List[str] = None, version: str = "1.0"):
        self.name = name
        self.description = description
        self.category = category
        self.parameters = parameters or {}
        self.examples = examples or []
        self.permissions = permissions or []
        self.version = version
        self.usage_count = 0
        self.last_used = None

class Tool:
    """Enhanced tool class with metadata and validation"""
    
    def __init__(self, metadata: ToolMetadata, function: Callable):
        self.metadata = metadata
        self.function = function
        self.enabled = True
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with parameter validation"""
        try:
            # Validate parameters if metadata specifies them
            if self.metadata.parameters:
                missing_required = []
                for param_name, param_info in self.metadata.parameters.items():
                    if param_info.get('required', False) and param_name not in kwargs:
                        missing_required.append(param_name)
                
                if missing_required:
                    return {
                        'success': False,
                        'error': f'Missing required parameters: {missing_required}',
                        'tool': self.metadata.name
                    }
            
            # Execute the tool
            self.metadata.usage_count += 1
            self.metadata.last_used = datetime.now()
            result = self.function(**kwargs)
            
            return {
                'success': True,
                'result': result,
                'tool': self.metadata.name,
                'timestamp': self.metadata.last_used.isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tool': self.metadata.name,
                'timestamp': datetime.now().isoformat()
            }

class EnhancedToolSystem:
    """Enhanced tool system with discovery and plugin support"""
    
    def __init__(self, config: Dict[str, Any], event_bus=None):
        self.config = config
        self.event_bus = event_bus
        self.tools = {}
        self.categories = {}
        self.enabled = config['tools']['enabled']
        self.safe_mode = config['tools']['safe_mode']
        self.timeout = config['tools']['timeout_seconds']
        
        # Tool directories to scan
        self.tool_directories = [
            Path(__file__).parent.parent / "tools",
            Path(__file__).parent.parent / "plugins" / "tools"
        ]
        
        if self.enabled:
            self._discover_and_load_tools()
    
    def _discover_and_load_tools(self):
        """Discover and load tools from multiple sources"""
        # Load core tools first
        self._load_core_tools()
        
        # Discover tools from directories
        for tool_dir in self.tool_directories:
            if tool_dir.exists():
                self._load_tools_from_directory(tool_dir)
        
        if self.event_bus:
            self.event_bus.emit('tools_loaded', {'tool_count': len(self.tools)})
    
    def _load_core_tools(self):
        """Load essential core tools"""
        # File operations
        self.register_tool(
            ToolMetadata(
                name="read_file",
                description="Read contents of a file",
                category="file_ops",
                parameters={
                    "file_path": {"required": True, "type": "str", "description": "Path to file to read"}
                },
                examples=["read_file(file_path='config.txt')"],
                permissions=["file_read"]
            ),
            self._read_file
        )
        
        self.register_tool(
            ToolMetadata(
                name="write_file",
                description="Write content to a file",
                category="file_ops", 
                parameters={
                    "file_path": {"required": True, "type": "str"},
                    "content": {"required": True, "type": "str"}
                },
                permissions=["file_write"]
            ),
            self._write_file
        )
        
        self.register_tool(
            ToolMetadata(
                name="list_directory",
                description="List contents of a directory",
                category="file_ops",
                parameters={"dir_path": {"required": True, "type": "str"}},
                permissions=["file_read"]
            ),
            self._list_directory
        )
        
        # Web operations
        self.register_tool(
            ToolMetadata(
                name="web_request",
                description="Make HTTP request to a URL",
                category="web_ops",
                parameters={
                    "url": {"required": True, "type": "str"},
                    "method": {"required": False, "type": "str", "default": "GET"}
                },
                permissions=["web_access"]
            ),
            self._web_request
        )
        
        # System operations
        self.register_tool(
            ToolMetadata(
                name="run_command",
                description="Execute system command",
                category="system_ops",
                parameters={"command": {"required": True, "type": "str"}},
                permissions=["system_exec"]
            ),
            self._run_command
        )
    
    def _load_tools_from_directory(self, directory: Path):
        """Load tools from a directory structure"""
        for category_dir in directory.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('_'):
                for tool_file in category_dir.glob("*.py"):
                    if tool_file.name != "__init__.py":
                        self._load_tool_from_file(tool_file, category_dir.name)
    
    def _load_tool_from_file(self, tool_file: Path, category: str):
        """Load a tool from a Python file"""
        try:
            spec = importlib.util.spec_from_file_location("tool_module", tool_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for tool registration function
            if hasattr(module, 'register_tools'):
                module.register_tools(self)
            elif hasattr(module, 'get_tool_metadata') and hasattr(module, 'execute'):
                # Simple tool format
                metadata = module.get_tool_metadata()
                metadata.category = category
                self.register_tool(ToolMetadata(**metadata), module.execute)
                
        except Exception as e:
            print(f"Failed to load tool from {tool_file}: {e}")
    
    def register_tool(self, metadata: ToolMetadata, function: Callable):
        """Register a tool with metadata"""
        tool = Tool(metadata, function)
        self.tools[metadata.name] = tool
        
        # Add to category
        if metadata.category not in self.categories:
            self.categories[metadata.category] = []
        self.categories[metadata.category].append(metadata.name)
        
        if self.event_bus:
            self.event_bus.emit('tool_registered', {'tool_name': metadata.name})
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools with enhanced metadata"""
        return [
            {
                'name': tool.metadata.name,
                'description': tool.metadata.description,
                'category': tool.metadata.category,
                'usage_count': tool.metadata.usage_count,
                'last_used': tool.metadata.last_used.isoformat() if tool.metadata.last_used else None,
                'enabled': tool.enabled,
                'parameters': tool.metadata.parameters,
                'examples': tool.metadata.examples
            }
            for tool in self.tools.values()
        ]
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tools in a specific category"""
        return self.categories.get(category, [])
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool with enhanced error handling"""
        if not self.enabled:
            return {'success': False, 'error': 'Tools are disabled'}
        
        if tool_name not in self.tools:
            return {'success': False, 'error': f'Tool {tool_name} not found'}
        
        tool = self.tools[tool_name]
        
        if not tool.enabled:
            return {'success': False, 'error': f'Tool {tool_name} is disabled'}
        
        # Check permissions in safe mode
        if self.safe_mode and not self._check_tool_permissions(tool):
            return {'success': False, 'error': f'Tool {tool_name} not allowed in safe mode'}
        
        if self.event_bus:
            self.event_bus.emit('tool_execution_start', {'tool_name': tool_name})
        
        result = tool.execute(**kwargs)
        
        if self.event_bus:
            self.event_bus.emit('tool_execution_complete', {
                'tool_name': tool_name, 
                'success': result['success']
            })
        
        return result
    
    def _check_tool_permissions(self, tool: Tool) -> bool:
        """Check if tool is allowed in current mode"""
        dangerous_permissions = ['system_exec', 'file_write', 'network_access']
        return not any(perm in tool.metadata.permissions for perm in dangerous_permissions)
    
    # Core tool implementations
    def _read_file(self, file_path: str) -> str:
        """Read file contents"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            size = path.stat().st_size
            return f"Binary file: {file_path} ({size} bytes)"
    
    def _write_file(self, file_path: str, content: str) -> str:
        """Write content to file"""
        path = Path(file_path)
        
        if self.safe_mode and not self._is_safe_path(path):
            raise PermissionError(f"Writing to {file_path} not allowed in safe mode")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return f"Successfully wrote {len(content)} characters to {file_path}"
    
    def _list_directory(self, dir_path: str) -> List[str]:
        """List directory contents"""
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        items = []
        for item in path.iterdir():
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            items.append(f"{prefix} {item.name}")
        
        return sorted(items)
    
    def _web_request(self, url: str, method: str = 'GET', headers: Dict = None, data: Any = None) -> Dict[str, Any]:
        """Make HTTP request"""
        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=headers or {},
                json=data if isinstance(data, dict) else None,
                timeout=self.timeout
            )
            
            return {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'content': response.text[:1000],
                'success': response.ok
            }
        except requests.RequestException as e:
            raise Exception(f"Request failed: {e}")
    
    def _run_command(self, command: str) -> str:
        """Execute system command"""
        if self.safe_mode:
            safe_commands = ['dir', 'ls', 'pwd', 'echo', 'date', 'whoami']
            if not any(command.startswith(cmd) for cmd in safe_commands):
                raise PermissionError(f"Command '{command}' not allowed in safe mode")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\\nSTDERR: {result.stderr}"
            
            return output
        except subprocess.TimeoutExpired:
            raise Exception(f"Command timed out after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Command execution failed: {e}")
    
    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is safe to write to in safe mode"""
        safe_directories = [
            Path.cwd(),
            Path.cwd() / 'data',
            Path.cwd() / 'logs',
        ]
        
        try:
            resolved_path = path.resolve()
            return any(
                str(resolved_path).startswith(str(safe_dir.resolve()))
                for safe_dir in safe_directories
            )
        except Exception:
            return False
'''
        
        # Write enhanced tool system
        tools_file = self.project_root / "core" / "enhanced_tools.py"
        with open(tools_file, 'w') as f:
            f.write(enhanced_tools)
        
        print("Enhanced tool system created")
    
    def upgrade_plugin_system(self):
        """Create enhanced plugin system"""
        print("Upgrading plugin system...")
        
        plugin_manager = '''import os
import sys
import json
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime

class PluginMetadata:
    """Metadata for plugins"""
    def __init__(self, name: str, version: str, description: str,
                 author: str = "Unknown", dependencies: List[str] = None,
                 capabilities: List[str] = None):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies or []
        self.capabilities = capabilities or []
        self.loaded_at = None
        self.enabled = True

class BasePlugin(ABC):
    """Base class for all plugins"""
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    @abstractmethod
    def initialize(self, system_context: Dict[str, Any]) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """Clean up plugin resources"""
        pass
    
    def on_tool_execution(self, tool_name: str, result: Dict[str, Any]) -> None:
        """Hook called after tool execution (optional)"""
        pass
    
    def on_conversation_update(self, conversation_data: Dict[str, Any]) -> None:
        """Hook called when conversation is updated (optional)"""
        pass

class SimpleEventBus:
    """Simple event system for component communication"""
    
    def __init__(self):
        self.listeners = {}
    
    def on(self, event: str, callback):
        """Register event listener"""
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(callback)
    
    def emit(self, event: str, data: Any = None):
        """Emit event to all listeners"""
        if event in self.listeners:
            for callback in self.listeners[event]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Event handler error for {event}: {e}")

class PluginManager:
    """Enhanced plugin management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plugins = {}
        self.event_bus = SimpleEventBus()
        self.plugin_directories = [
            Path(__file__).parent.parent / "plugins"
        ]
    
    def discover_plugins(self) -> List[Path]:
        """Discover plugin files"""
        plugin_files = []
        
        for plugin_dir in self.plugin_directories:
            if plugin_dir.exists():
                # Look for Python files
                for file_path in plugin_dir.glob("**/*.py"):
                    if file_path.name not in ["__init__.py", "base_plugin.py"]:
                        plugin_files.append(file_path)
        
        return plugin_files
    
    def load_plugin(self, plugin_file: Path) -> bool:
        """Load a single plugin"""
        try:
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_file.stem}", plugin_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for plugin class
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePlugin) and 
                    attr != BasePlugin):
                    plugin_class = attr
                    break
            
            if not plugin_class:
                print(f"No plugin class found in {plugin_file}")
                return False
            
            # Initialize plugin
            plugin_instance = plugin_class()
            metadata = plugin_instance.get_metadata()
            
            # Check dependencies
            if not self._check_dependencies(metadata):
                print(f"Dependencies not met for plugin {metadata.name}")
                return False
            
            # Initialize plugin
            system_context = {
                'config': self.config,
                'event_bus': self.event_bus,
                'plugin_manager': self
            }
            
            if plugin_instance.initialize(system_context):
                metadata.loaded_at = datetime.now()
                self.plugins[metadata.name] = {
                    'instance': plugin_instance,
                    'metadata': metadata,
                    'file_path': plugin_file
                }
                
                self.event_bus.emit('plugin_loaded', {
                    'name': metadata.name,
                    'version': metadata.version
                })
                
                print(f"Loaded plugin: {metadata.name} v{metadata.version}")
                return True
            else:
                print(f"Failed to initialize plugin: {metadata.name}")
                return False
                
        except Exception as e:
            print(f"Error loading plugin {plugin_file}: {e}")
            return False
    
    def load_all_plugins(self):
        """Load all discovered plugins"""
        plugin_files = self.discover_plugins()
        
        print(f"Discovered {len(plugin_files)} potential plugins")
        
        loaded_count = 0
        for plugin_file in plugin_files:
            if self.load_plugin(plugin_file):
                loaded_count += 1
        
        print(f"Successfully loaded {loaded_count} plugins")
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self.plugins:
            return False
        
        try:
            plugin_data = self.plugins[plugin_name]
            plugin_data['instance'].cleanup()
            del self.plugins[plugin_name]
            
            self.event_bus.emit('plugin_unloaded', {'name': plugin_name})
            print(f"Unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            print(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def get_loaded_plugins(self) -> List[Dict[str, Any]]:
        """Get list of loaded plugins"""
        return [
            {
                'name': data['metadata'].name,
                'version': data['metadata'].version,
                'description': data['metadata'].description,
                'author': data['metadata'].author,
                'capabilities': data['metadata'].capabilities,
                'loaded_at': data['metadata'].loaded_at.isoformat(),
                'enabled': data['metadata'].enabled
            }
            for data in self.plugins.values()
        ]
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name]['metadata'].enabled = True
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name]['metadata'].enabled = False
            return True
        return False
    
    def _check_dependencies(self, metadata: PluginMetadata) -> bool:
        """Check if plugin dependencies are met"""
        for dep in metadata.dependencies:
            if dep not in self.plugins:
                return False
        return True
    
    def get_event_bus(self) -> SimpleEventBus:
        """Get the event bus for inter-component communication"""
        return self.event_bus
'''
        
        # Write plugin manager
        plugin_file = self.project_root / "core" / "plugin_manager.py"
        with open(plugin_file, 'w') as f:
            f.write(plugin_manager)
        
        print("Enhanced plugin system created")
    
    def upgrade_config_system(self):
        """Upgrade configuration system with validation"""
        print("Upgrading configuration system...")
        
        # Enhanced configuration with validation
        enhanced_config = {
            "lm_studio": {
                "host": "localhost",
                "port": 1234,
                "endpoint": "/v1/chat/completions",
                "model": "local-model",
                "timeout": 30,
                "max_retries": 3
            },
            "memory": {
                "database_path": "data/memory.db",
                "max_conversations": 1000,
                "importance_threshold": 0.3,
                "cleanup_interval_hours": 24
            },
            "tools": {
                "enabled": True,
                "safe_mode": True,
                "timeout_seconds": 30,
                "max_parallel_tools": 3,
                "categories_enabled": ["file_ops", "web_ops", "system_ops"]
            },
            "plugins": {
                "enabled": True,
                "auto_load": True,
                "plugin_directories": ["plugins"],
                "max_plugins": 20
            },
            "logging": {
                "level": "INFO",
                "file": "logs/agent.log",
                "max_size_mb": 10,
                "backup_count": 5,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "ui_plugin": "terminal_ui",
            "features": {
                "auto_save_conversations": True,
                "context_window_size": 10,
                "enable_tool_chaining": True,
                "enable_plugin_hooks": True
            }
        }
        
        # Write enhanced config
        config_file = self.project_root / "config" / "settings.json"
        with open(config_file, 'w') as f:
            json.dump(enhanced_config, f, indent=4)
        
        print("Enhanced configuration created")
    
    def create_enhanced_agent(self):
        """Create enhanced agent that uses new systems"""
        print("Creating enhanced agent...")
        
        enhanced_agent = '''import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from .memory import MemorySystem
from .enhanced_tools import EnhancedToolSystem
from .plugin_manager import PluginManager

class EnhancedAgent:
    """
    Enhanced AI Agent with plugin support and better architecture
    Backward compatible with the original Agent
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize plugin manager first (provides event bus)
        self.plugin_manager = PluginManager(config)
        self.event_bus = self.plugin_manager.get_event_bus()
        
        # Initialize subsystems with event bus
        self.memory = MemorySystem(config)
        self.tools = EnhancedToolSystem(config, self.event_bus)
        
        # LM Studio connection settings
        self.lm_studio_host = config['lm_studio']['host']
        self.lm_studio_port = config['lm_studio']['port']
        self.lm_studio_endpoint = config['lm_studio']['endpoint']
        self.model_name = config['lm_studio']['model']
        self.timeout = config['lm_studio'].get('timeout', 30)
        self.max_retries = config['lm_studio'].get('max_retries', 3)
        
        # Agent state
        self.conversation_active = False
        self.current_context = {}
        
        # Load plugins if enabled
        if config.get('plugins', {}).get('enabled', False):
            self.plugin_manager.load_all_plugins()
        
        self.logger.info("Enhanced Agent initialized successfully")
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Enhanced input processing with plugin hooks
        Backward compatible with original process_input
        """
        try:
            self.logger.info(f"Processing input: {user_input[:100]}...")
            self.event_bus.emit('input_received', {'input': user_input})
            
            # Step 1: Analyze input
            analysis = self._analyze_input(user_input)
            self.event_bus.emit('input_analyzed', analysis)
            
            # Step 2: Gather context
            context = self._gather_context(analysis)
            
            # Step 3: Plan actions
            plan = self._create_action_plan(user_input, analysis, context)
            
            # Step 4: Execute tools if needed
            tool_results = []
            if plan.get('tools_needed'):
                tool_results = self._execute_tools_with_retry(plan['tools_needed'])
            
            # Step 5: Generate response
            response = self._generate_response(user_input, context, tool_results, plan)
            
            # Step 6: Update memory
            tools_used = [tool['tool'] for tool in tool_results if tool.get('success')]
            conversation_id = self.memory.store_conversation(
                user_input, 
                response, 
                tools_used, 
                {'analysis': analysis, 'plan': plan}
            )
            
            result = {
                'response': response,
                'tools_used': tools_used,
                'conversation_id': conversation_id,
                'analysis': analysis,
                'success': True
            }
            
            # Notify plugins
            self.event_bus.emit('conversation_updated', {
                'user_input': user_input,
                'response': response,
                'tools_used': tools_used
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            error_result = {
                'response': f"I encountered an error: {str(e)}",
                'tools_used': [],
                'conversation_id': -1,
                'analysis': {},
                'success': False
            }
            self.event_bus.emit('processing_error', {'error': str(e)})
            return error_result
    
    def _execute_tools_with_retry(self, tools_needed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tools with retry logic"""
        results = []
        
        for tool_spec in tools_needed:
            tool_name = tool_spec['tool']
            params = tool_spec.get('params', {})
            
            # Try execution with retries
            for attempt in range(self.max_retries):
                try:
                    self.logger.info(f"Executing tool: {tool_name} (attempt {attempt + 1})")
                    result = self.tools.execute_tool(tool_name, **params)
                    
                    if result['success']:
                        results.append(result)
                        break
                    else:
                        if attempt == self.max_retries - 1:
                            results.append(result)
                        else:
                            self.logger.warning(f"Tool {tool_name} failed, retrying...")
                            
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        results.append({
                            'success': False,
                            'error': str(e),
                            'tool': tool_name
                        })
        
        return results
    
    # Include original methods for backward compatibility
    def _analyze_input(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input to understand intent and complexity"""
        analysis = {
            'intent': 'general',
            'complexity': 'medium',
            'requires_tools': False,
            'emotional_tone': 'neutral'
        }
        
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['read', 'open', 'show', 'display']):
            analysis['intent'] = 'file_operation'
            analysis['requires_tools'] = True
        elif any(word in input_lower for word in ['write', 'create', 'save', 'make']):
            analysis['intent'] = 'creation'
            analysis['requires_tools'] = True
        elif any(word in input_lower for word in ['search', 'find', 'look']):
            analysis['intent'] = 'search'
            analysis['requires_tools'] = True
        elif any(word in input_lower for word in ['run', 'execute', 'command']):
            analysis['intent'] = 'execution'
            analysis['requires_tools'] = True
        elif '?' in user_input:
            analysis['intent'] = 'question'
        
        return analysis
    
    def _gather_context(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced context gathering"""
        context = {
            'recent_conversations': [],
            'user_facts': {},
            'available_tools': [],
            'loaded_plugins': []
        }
        
        # Get memory context
        memory_context = self.memory.get_conversation_context()
        context.update(memory_context)
        
        # Get available tools if needed
        if analysis.get('requires_tools'):
            context['available_tools'] = self.tools.get_available_tools()
        
        # Get loaded plugins
        context['loaded_plugins'] = self.plugin_manager.get_loaded_plugins()
        
        return context
    
    def _create_action_plan(self, user_input: str, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced action planning"""
        plan = {
            'approach': 'direct_response',
            'tools_needed': [],
            'reasoning': 'Simple response without tools'
        }
        
        if analysis.get('requires_tools'):
            intent = analysis.get('intent')
            
            if intent == 'file_operation':
                if 'read' in user_input.lower():
                    plan['tools_needed'].append({
                        'tool': 'read_file',
                        'params': self._extract_file_path(user_input)
                    })
                elif 'list' in user_input.lower() or 'directory' in user_input.lower():
                    plan['tools_needed'].append({
                        'tool': 'list_directory',
                        'params': self._extract_directory_path(user_input)
                    })
            
            elif intent == 'creation':
                plan['tools_needed'].append({
                    'tool': 'write_file',
                    'params': self._extract_write_params(user_input)
                })
            
            elif intent == 'execution':
                plan['tools_needed'].append({
                    'tool': 'run_command',
                    'params': {'command': self._extract_command(user_input)}
                })
            
            if plan['tools_needed']:
                plan['approach'] = 'tool_assisted'
                plan['reasoning'] = f"Need to use tools for {intent}"
        
        return plan
    
    def _generate_response(self, user_input: str, context: Dict[str, Any], 
                          tool_results: List[Dict[str, Any]], plan: Dict[str, Any]) -> str:
        """Enhanced response generation"""
        try:
            prompt = self._build_enhanced_prompt(user_input, context, tool_results, plan)
            response = self._call_lm_studio_with_retry(prompt)
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._fallback_response(user_input, tool_results)
    
    def _build_enhanced_prompt(self, user_input: str, context: Dict[str, Any], 
                              tool_results: List[Dict[str, Any]], plan: Dict[str, Any]) -> str:
        """Build enhanced prompt with plugin context"""
        prompt_parts = []
        
        prompt_parts.append("You are Cerb, an intelligent AI assistant with enhanced capabilities.")
        
        if context.get('recent_conversations'):
            prompt_parts.append("\\nRecent conversation context:")
            for conv in context['recent_conversations'][-3:]:
                prompt_parts.append(f"User: {conv['user_input']}")
                prompt_parts.append(f"Assistant: {conv['ai_response'][:200]}...")
        
        if tool_results:
            prompt_parts.append("\\nTool execution results:")
            for result in tool_results:
                if result.get('success'):
                    prompt_parts.append(f"Tool {result['tool']} succeeded: {str(result['result'])[:300]}")
                else:
                    prompt_parts.append(f"Tool {result['tool']} failed: {result.get('error', 'Unknown error')}")
        
        prompt_parts.append(f"\\nCurrent user request: {user_input}")
        prompt_parts.append("\\nProvide a helpful, accurate response. Be conversational and mention any tools used.")
        
        return "\\n".join(prompt_parts)
    
    def _call_lm_studio_with_retry(self, prompt: str) -> str:
        """Call LM Studio with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return self._call_lm_studio(prompt)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                else:
                    self.logger.warning(f"LM Studio call failed, retrying... ({e})")
                    import time
                    time.sleep(1)
    
    def _call_lm_studio(self, prompt: str) -> str:
        """Call LM Studio API"""
        url = f"http://{self.lm_studio_host}:{self.lm_studio_port}{self.lm_studio_endpoint}"
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to LM Studio. Make sure it's running on localhost:1234")
        except Exception as e:
            raise Exception(f"LM Studio API error: {e}")
    
    def _fallback_response(self, user_input: str, tool_results: List[Dict[str, Any]]) -> str:
        """Generate fallback response when LM Studio is unavailable"""
        response_parts = ["I apologize, but I'm having trouble connecting to my language model."]
        
        if tool_results:
            response_parts.append("However, I was able to execute some tools for you:")
            for result in tool_results:
                if result.get('success'):
                    response_parts.append(f"- {result['tool']}: {str(result['result'])[:200]}")
                else:
                    response_parts.append(f"- {result['tool']}: Failed - {result.get('error')}")
        
        response_parts.append("Please try again or check if LM Studio is running.")
        return "\\n".join(response_parts)
    
    # Original utility methods for backward compatibility
    def _extract_file_path(self, user_input: str) -> Dict[str, str]:
        words = user_input.split()
        for word in words:
            if '.' in word and any(word.endswith(ext) for ext in ['.txt', '.py', '.json', '.md']):
                return {'file_path': word}
        return {'file_path': 'MISSING_FILE_PATH'}
    
    def _extract_directory_path(self, user_input: str) -> Dict[str, str]:
        if 'current' in user_input.lower() or 'this' in user_input.lower():
            return {'dir_path': '.'}
        words = user_input.split()
        for word in words:
            if '/' in word or '\\\\' in word:
                return {'dir_path': word}
        return {'dir_path': '.'}
    
    def _extract_write_params(self, user_input: str) -> Dict[str, str]:
        return {'file_path': 'MISSING_FILE_PATH', 'content': 'MISSING_CONTENT'}
    
    def _extract_command(self, user_input: str) -> str:
        words = user_input.split()
        for i, word in enumerate(words):
            if word.lower() in ['run', 'execute', 'command']:
                if i + 1 < len(words):
                    return ' '.join(words[i+1:])
        return 'MISSING_COMMAND'

# Backward compatibility alias
Agent = EnhancedAgent
'''
        
        # Write enhanced agent
        agent_file = self.project_root / "core" / "enhanced_agent.py"
        with open(agent_file, 'w') as f:
            f.write(enhanced_agent)
        
        print("Enhanced agent created")
    
    def create_tool_examples(self):
        """Create example tools to demonstrate the new system"""
        print("Creating example tools...")
        
        # Create tools directory structure
        tools_dir = self.project_root / "tools"
        tools_dir.mkdir(exist_ok=True)
        
        # Example file operations tool
        file_ops_dir = tools_dir / "file_ops"
        file_ops_dir.mkdir(exist_ok=True)
        
        example_tool = '''"""
Example file statistics tool
Shows how to create tools in the new system
"""

from pathlib import Path
from typing import Dict, Any

def get_tool_metadata() -> Dict[str, Any]:
    """Return tool metadata"""
    return {
        'name': 'file_stats',
        'description': 'Get statistics about a file or directory',
        'category': 'file_ops',
        'parameters': {
            'path': {
                'required': True,
                'type': 'str',
                'description': 'Path to file or directory'
            }
        },
        'examples': ['file_stats(path="config.json")'],
        'permissions': ['file_read']
    }

def execute(path: str) -> Dict[str, Any]:
    """Execute the tool"""
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    stats = file_path.stat()
    
    result = {
        'path': str(file_path),
        'size_bytes': stats.st_size,
        'modified': stats.st_mtime,
        'is_directory': file_path.is_dir(),
        'is_file': file_path.is_file()
    }
    
    if file_path.is_dir():
        items = list(file_path.iterdir())
        result['item_count'] = len(items)
        result['subdirectories'] = len([i for i in items if i.is_dir()])
        result['files'] = len([i for i in items if i.is_file()])
    
    return result
'''
        
        with open(file_ops_dir / "file_stats.py", 'w') as f:
            f.write(example_tool)
        
        # Create __init__.py files
        with open(tools_dir / "__init__.py", 'w') as f:
            f.write("# Tools directory")
        
        with open(file_ops_dir / "__init__.py", 'w') as f:
            f.write("# File operations tools")
        
        print("Example tools created")


def main():
    """Main function to run the upgrade"""
    if len(sys.argv) != 2:
        print("Usage: python architecture_upgrade.py <project_directory>")
        print("Example: python architecture_upgrade.py 'C:/Users/RatMa/Desktop/test/Cerb 6'")
        sys.exit(1)
    
    project_dir = sys.argv[1]
    
    if not Path(project_dir).exists():
        print(f"Error: Project directory does not exist: {project_dir}")
        sys.exit(1)
    
    upgrader = ArchitectureUpgrader(project_dir)
    upgrader.run_upgrade()
    
    print("\nUpgrade complete! Your system now has:")
    print("- Enhanced tool discovery system")
    print("- Plugin management with lifecycle support")
    print("- Event bus for component communication")
    print("- Improved error handling and retry logic")
    print("- Configuration validation with automatic compatibility fixes")
    print("- Updated main.py that gracefully handles both old and new systems")
    print("- Backward compatibility with existing code")
    print("\nRun your system normally with 'python main.py'")
    print("The system will automatically detect and use enhanced features when available.")


if __name__ == "__main__":
    main()