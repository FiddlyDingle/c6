import os
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
