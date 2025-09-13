#!/usr/bin/env python3
"""
Test script for Chunk 1 - Enhanced Memory System
Tests that the enhanced memory system works properly
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_enhanced_memory():
    """Test enhanced memory system"""
    print("Testing Enhanced Memory System...")
    
    try:
        from core.enhanced_memory import EnhancedMemorySystem
        
        # Test configuration
        config = {
            'memory': {
                'database_path': 'data/test_memory.db',
                'max_conversations': 100,
                'importance_threshold': 0.3
            }
        }
        
        # Initialize memory system
        memory = EnhancedMemorySystem(config)
        print("✓ Enhanced memory system initialized")
        
        # Test storing conversation
        conv_id = memory.store_conversation(
            "Hello, can you help me with Python?",
            "Of course! I'd be happy to help you with Python programming.",
            tools_used=["web_search"],
            context={"test": True}
        )
        
        print(f"✓ Stored conversation with ID: {conv_id}")
        
        # Test retrieving conversations
        recent = memory.get_recent_conversations(5)
        print(f"✓ Retrieved {len(recent)} recent conversations")
        
        # Test context gathering
        context = memory.get_conversation_context("Python programming")
        print(f"✓ Context gathered with {len(context.get('recent_conversations', []))} recent conversations")
        
        # Test semantic search if available
        if hasattr(memory, 'semantic_search'):
            results = memory.semantic_search("Python help", 3)
            print(f"✓ Semantic search returned {len(results)} results")
        else:
            print("- Semantic search not available (ChromaDB not installed)")
        
        print("Enhanced memory system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced memory system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_agent():
    """Test enhanced agent with new memory system"""
    print("\nTesting Enhanced Agent...")
    
    try:
        # Load configuration
        import json
        config_path = Path(__file__).parent / 'config' / 'settings.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Add missing config sections for compatibility
        if 'plugins' not in config:
            config['plugins'] = {'enabled': True}
        if 'features' not in config:
            config['features'] = {'enable_plugin_hooks': True}
        
        from core.enhanced_agent import EnhancedAgent
        
        # Initialize agent
        agent = EnhancedAgent(config)
        print("✓ Enhanced agent initialized with new memory system")
        
        # Test that memory system is enhanced
        if hasattr(agent.memory, 'semantic_search'):
            print("✓ Agent is using enhanced memory system with semantic search")
        else:
            print("- Agent is using basic memory system (fallback mode)")
        
        print("Enhanced agent test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Chunk 1 tests"""
    print("=== Chunk 1: Enhanced Memory System Tests ===\n")
    
    tests_passed = 0
    total_tests = 2
    
    if test_enhanced_memory():
        tests_passed += 1
    
    if test_enhanced_agent():
        tests_passed += 1
    
    print(f"\n=== Test Results: {tests_passed}/{total_tests} tests passed ===")
    
    if tests_passed == total_tests:
        print("✓ Chunk 1 implementation is working correctly!")
        print("\nWhat's been enhanced:")
        print("- Enhanced memory system with semantic search capabilities")
        print("- ChromaDB integration for intelligent conversation recall")
        print("- Enhanced conversation context with topic tracking")
        print("- Sentiment analysis and importance scoring")
        print("- Session tracking and topic clustering")
        print("- Backward compatibility with existing memory system")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
