import os
import sys
import subprocess
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
import requests
from datetime import datetime

class Tool:
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str, function: Callable):
        self.name = name
        self.description = description
        self.function = function
        self.last_used = None
        self.usage_count = 0
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        try:
            self.last_used = datetime.now()
            self.usage_count += 1
            result = self.function(**kwargs)
            
            return {
                'success': True,
                'result': result,
                'tool': self.name,
                'timestamp': self.last_used.isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tool': self.name,
                'timestamp': datetime.now().isoformat()
            }

class ToolSystem:
    """
    Tool management system for Phase 1
    Provides essential tools for file operations, web requests, and basic code execution
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tools = {}
        self.enabled = config['tools']['enabled']
        self.safe_mode = config['tools']['safe_mode']
        self.timeout = config['tools']['timeout_seconds']
        self.logger = logging.getLogger(__name__)
        
        if self.enabled:
            self._register_core_tools()
    
    def _register_core_tools(self) -> None:
        """Register essential tools for Phase 1"""
        
        # File operations
        self.register_tool('read_file', 'Read contents of a file', self._read_file)
        self.register_tool('write_file', 'Write content to a file', self._write_file)
        self.register_tool('list_directory', 'List contents of a directory', self._list_directory)
        self.register_tool('search_files', 'Search for files by name pattern', self._search_files)
        
        # Web operations
        self.register_tool('web_search', 'Search the web for information', self._web_search)
        self.register_tool('web_request', 'Make HTTP request to a URL', self._web_request)
        
        # System operations
        self.register_tool('run_command', 'Execute system command', self._run_command)
        self.register_tool('get_system_info', 'Get basic system information', self._get_system_info)
        
        # Code execution
        self.register_tool('execute_python', 'Execute Python code safely', self._execute_python)
        
        self.logger.info(f"Registered {len(self.tools)} core tools")
    
    def register_tool(self, name: str, description: str, function: Callable) -> None:
        """Register a new tool"""
        tool = Tool(name, description, function)
        self.tools[name] = tool
        self.logger.debug(f"Registered tool: {name}")
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools with descriptions"""
        return [
            {
                'name': name,
                'description': tool.description,
                'usage_count': tool.usage_count,
                'last_used': tool.last_used.isoformat() if tool.last_used else None
            }
            for name, tool in self.tools.items()
        ]
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool"""
        if not self.enabled:
            return {'success': False, 'error': 'Tools are disabled'}
        
        if tool_name not in self.tools:
            return {'success': False, 'error': f'Tool {tool_name} not found'}
        
        tool = self.tools[tool_name]
        self.logger.info(f"Executing tool: {tool_name}")
        
        try:
            result = tool.execute(**kwargs)
            self.logger.debug(f"Tool {tool_name} completed: {result['success']}")
            return result
        except Exception as e:
            self.logger.error(f"Tool {tool_name} failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # Core tool implementations
    def _read_file(self, file_path: str) -> str:
        """Read file contents"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try reading as binary and return info about the file
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
    
    def _search_files(self, directory: str, pattern: str) -> List[str]:
        """Search for files matching pattern"""
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        matches = []
        for item in path.rglob(pattern):
            matches.append(str(item))
        
        return matches
    
    def _web_search(self, query: str) -> str:
        """Simple web search placeholder (would integrate with search API)"""
        return f"Web search for '{query}' - This would integrate with a search API in full implementation"
    
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
                'content': response.text[:1000],  # Limit response size
                'success': response.ok
            }
        except requests.RequestException as e:
            raise Exception(f"Request failed: {e}")
    
    def _run_command(self, command: str) -> str:
        """Execute system command"""
        if self.safe_mode:
            # Only allow safe commands in safe mode
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
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        return {
            'platform': sys.platform,
            'python_version': sys.version,
            'current_directory': os.getcwd(),
            'environment_variables': dict(os.environ),
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_python(self, code: str) -> str:
        """Execute Python code in safe environment"""
        if self.safe_mode:
            # Basic safety checks
            dangerous_imports = ['os', 'sys', 'subprocess', 'importlib']
            if any(f"import {module}" in code for module in dangerous_imports):
                raise PermissionError("Dangerous imports not allowed in safe mode")
        
        try:
            # Create a restricted environment
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'range': range,
                    'sum': sum,
                    'max': max,
                    'min': min,
                }
            }
            
            # Capture output
            import io
            import contextlib
            
            output_buffer = io.StringIO()
            
            with contextlib.redirect_stdout(output_buffer):
                exec(code, safe_globals)
            
            return output_buffer.getvalue()
            
        except Exception as e:
            raise Exception(f"Python execution failed: {e}")
    
    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is safe to write to in safe mode"""
        safe_directories = [
            Path.cwd(),
            Path.cwd() / 'data',
            Path.cwd() / 'logs',
            Path('/tmp') if sys.platform != 'win32' else Path('C:/temp')
        ]
        
        try:
            resolved_path = path.resolve()
            return any(
                str(resolved_path).startswith(str(safe_dir.resolve()))
                for safe_dir in safe_directories
            )
        except Exception:
            return False
