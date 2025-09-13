import os
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
                output += f"\nSTDERR: {result.stderr}"
            
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
