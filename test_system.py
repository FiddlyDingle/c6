#!/usr/bin/env python3
"""
Test script for Cerb AI Assistant Phase 1
Verifies all components can be imported and initialized
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all core modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test core imports
        from core.memory import MemorySystem
        from core.tools import ToolSystem
        from core.agent import Agent
        from core.ui_interface import UIInterface
        
        # Test UI plugin import
        from ui_plugins.terminal_ui import TerminalUI
        
        print("✓ All imports successful")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during imports: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    try:
        print("Testing configuration loading...")
        
        import json
        config_path = Path(__file__).parent / 'config' / 'settings.json'
        
        if not config_path.exists():
            print("✗ Configuration file not found")
            return False
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        required_keys = ['lm_studio', 'memory', 'tools', 'logging']
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing required config key: {key}")
                return False
        
        print("✓ Configuration loading successful")
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_component_initialization():
    """Test that components can be initialized"""
    try:
        print("Testing component initialization...")
        
        # Load config
        import json
        config_path = Path(__file__).parent / 'config' / 'settings.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Test memory system
        from core.memory import MemorySystem
        memory = MemorySystem(config)
        print("✓ Memory system initialized")
        
        # Test tool system
        from core.tools import ToolSystem
        tools = ToolSystem(config)
        print("✓ Tool system initialized")
        
        # Test agent (but don't make actual API calls)
        from core.agent import Agent
        agent = Agent(config)
        print("✓ Agent initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Component initialization error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Cerb AI Assistant Phase 1 Test Suite ===\n")
    
    tests = [
        test_imports,
        test_config_loading,
        test_component_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("✓ All tests passed! System is ready to run.")
        print("Run 'python main.py' to start the assistant.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
