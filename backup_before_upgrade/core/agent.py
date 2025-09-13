import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from .memory import MemorySystem
from .tools import ToolSystem

class Agent:
    """
    Main AI Agent for Phase 1
    Handles reasoning, tool execution, and memory management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize subsystems
        self.memory = MemorySystem(config)
        self.tools = ToolSystem(config)
        
        # LM Studio connection settings
        self.lm_studio_host = config['lm_studio']['host']
        self.lm_studio_port = config['lm_studio']['port']
        self.lm_studio_endpoint = config['lm_studio']['endpoint']
        self.model_name = config['lm_studio']['model']
        
        # Agent state
        self.conversation_active = False
        self.current_context = {}
        
        self.logger.info("Agent initialized successfully")
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Main input processing pipeline
        1. Analyze input
        2. Gather context
        3. Plan actions
        4. Execute tools if needed
        5. Generate response
        6. Update memory
        """
        try:
            self.logger.info(f"Processing input: {user_input[:100]}...")
            
            # Step 1: Analyze input
            analysis = self._analyze_input(user_input)
            
            # Step 2: Gather context
            context = self._gather_context(analysis)
            
            # Step 3: Plan actions
            plan = self._create_action_plan(user_input, analysis, context)
            
            # Step 4: Execute tools if needed
            tool_results = []
            if plan.get('tools_needed'):
                tool_results = self._execute_tools(plan['tools_needed'])
            
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
            
            return {
                'response': response,
                'tools_used': tools_used,
                'conversation_id': conversation_id,
                'analysis': analysis,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            return {
                'response': f"I encountered an error: {str(e)}",
                'tools_used': [],
                'conversation_id': -1,
                'analysis': {},
                'success': False
            }
    
    def _analyze_input(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input to understand intent and complexity"""
        analysis = {
            'intent': 'general',
            'complexity': 'medium',
            'requires_tools': False,
            'emotional_tone': 'neutral',
            'topics': [],
            'entities': []
        }
        
        input_lower = user_input.lower()
        
        # Detect intent
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
        
        # Assess complexity
        if len(user_input.split()) > 20:
            analysis['complexity'] = 'high'
        elif len(user_input.split()) < 5:
            analysis['complexity'] = 'low'
        
        # Detect emotional tone
        if any(word in input_lower for word in ['please', 'help', 'thanks']):
            analysis['emotional_tone'] = 'polite'
        elif any(word in input_lower for word in ['urgent', 'asap', 'quickly']):
            analysis['emotional_tone'] = 'urgent'
        
        return analysis
    
    def _gather_context(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Gather relevant context from memory and current state"""
        context = {
            'recent_conversations': [],
            'user_facts': {},
            'available_tools': [],
            'system_state': {}
        }
        
        # Get memory context
        memory_context = self.memory.get_conversation_context()
        context.update(memory_context)
        
        # Get available tools if needed
        if analysis.get('requires_tools'):
            context['available_tools'] = self.tools.get_available_tools()
        
        return context
    
    def _create_action_plan(self, user_input: str, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create action plan based on input analysis and context"""
        plan = {
            'approach': 'direct_response',
            'tools_needed': [],
            'reasoning': 'Simple response without tools',
            'steps': []
        }
        
        # Determine if tools are needed based on analysis
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
            
            elif intent == 'search':
                if 'web' in user_input.lower() or 'internet' in user_input.lower():
                    plan['tools_needed'].append({
                        'tool': 'web_search',
                        'params': {'query': self._extract_search_query(user_input)}
                    })
                else:
                    plan['tools_needed'].append({
                        'tool': 'search_files',
                        'params': self._extract_file_search_params(user_input)
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
    
    def _execute_tools(self, tools_needed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute required tools"""
        results = []
        
        for tool_spec in tools_needed:
            tool_name = tool_spec['tool']
            params = tool_spec.get('params', {})
            
            self.logger.info(f"Executing tool: {tool_name}")
            result = self.tools.execute_tool(tool_name, **params)
            results.append(result)
        
        return results
    
    def _generate_response(self, user_input: str, context: Dict[str, Any], 
                          tool_results: List[Dict[str, Any]], plan: Dict[str, Any]) -> str:
        """Generate AI response using LM Studio"""
        try:
            # Build prompt with context
            prompt = self._build_prompt(user_input, context, tool_results, plan)
            
            # Call LM Studio API
            response = self._call_lm_studio(prompt)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._fallback_response(user_input, tool_results)
    
    def _build_prompt(self, user_input: str, context: Dict[str, Any], 
                     tool_results: List[Dict[str, Any]], plan: Dict[str, Any]) -> str:
        """Build comprehensive prompt for LM Studio"""
        prompt_parts = []
        
        # System context
        prompt_parts.append("You are Cerb, an intelligent AI assistant. You have access to tools and memory.")
        
        # Recent conversation context
        if context.get('recent_conversations'):
            prompt_parts.append("\nRecent conversation context:")
            for conv in context['recent_conversations'][-3:]:  # Last 3 conversations
                prompt_parts.append(f"User: {conv['user_input']}")
                prompt_parts.append(f"Assistant: {conv['ai_response'][:200]}...")
        
        # User facts
        if context.get('user_facts'):
            prompt_parts.append("\nKnown user information:")
            for fact_type, facts in context['user_facts'].items():
                for key, value in facts.items():
                    prompt_parts.append(f"- {fact_type}.{key}: {value['value']}")
        
        # Tool results
        if tool_results:
            prompt_parts.append("\nTool execution results:")
            for result in tool_results:
                if result.get('success'):
                    prompt_parts.append(f"Tool {result['tool']} succeeded: {str(result['result'])[:300]}")
                else:
                    prompt_parts.append(f"Tool {result['tool']} failed: {result.get('error', 'Unknown error')}")
        
        # Current user input
        prompt_parts.append(f"\nCurrent user request: {user_input}")
        
        # Response instruction
        prompt_parts.append("\nProvide a helpful, accurate response based on the context and tool results above. Be conversational and natural.")
        
        return "\n".join(prompt_parts)
    
    def _call_lm_studio(self, prompt: str) -> str:
        """Call LM Studio API"""
        url = f"http://{self.lm_studio_host}:{self.lm_studio_port}{self.lm_studio_endpoint}"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to LM Studio. Make sure it's running on localhost:1234")
        except requests.exceptions.Timeout:
            raise Exception("LM Studio request timed out")
        except Exception as e:
            raise Exception(f"LM Studio API error: {e}")
    
    def _fallback_response(self, user_input: str, tool_results: List[Dict[str, Any]]) -> str:
        """Generate fallback response when LM Studio is unavailable"""
        response_parts = ["I apologize, but I'm having trouble connecting to my language model."]
        
        # Include tool results if available
        if tool_results:
            response_parts.append("However, I was able to execute some tools for you:")
            for result in tool_results:
                if result.get('success'):
                    response_parts.append(f"- {result['tool']}: {str(result['result'])[:200]}")
                else:
                    response_parts.append(f"- {result['tool']}: Failed - {result.get('error')}")
        
        response_parts.append("Please try again or check if LM Studio is running.")
        return "\n".join(response_parts)
    
    # Utility methods for parameter extraction
    def _extract_file_path(self, user_input: str) -> Dict[str, str]:
        """Extract file path from user input"""
        # Simple extraction - look for quoted strings or common file extensions
        words = user_input.split()
        for word in words:
            if '.' in word and any(word.endswith(ext) for ext in ['.txt', '.py', '.json', '.md']):
                return {'file_path': word}
        
        # Fallback: ask for file path
        return {'file_path': 'MISSING_FILE_PATH'}
    
    def _extract_directory_path(self, user_input: str) -> Dict[str, str]:
        """Extract directory path from user input"""
        # Look for directory indicators
        if 'current' in user_input.lower() or 'this' in user_input.lower():
            return {'dir_path': '.'}
        
        # Look for quoted paths
        words = user_input.split()
        for word in words:
            if '/' in word or '\\' in word:
                return {'dir_path': word}
        
        return {'dir_path': '.'}
    
    def _extract_write_params(self, user_input: str) -> Dict[str, str]:
        """Extract write parameters from user input"""
        return {
            'file_path': 'MISSING_FILE_PATH',
            'content': 'MISSING_CONTENT'
        }
    
    def _extract_search_query(self, user_input: str) -> str:
        """Extract search query from user input"""
        # Remove common command words
        query_words = []
        skip_words = ['search', 'find', 'look', 'for', 'web', 'internet']
        
        for word in user_input.split():
            if word.lower() not in skip_words:
                query_words.append(word)
        
        return ' '.join(query_words) if query_words else 'MISSING_QUERY'
    
    def _extract_file_search_params(self, user_input: str) -> Dict[str, str]:
        """Extract file search parameters"""
        return {
            'directory': '.',
            'pattern': '*.*'
        }
    
    def _extract_command(self, user_input: str) -> str:
        """Extract command from user input"""
        # Simple extraction - everything after 'run' or 'execute'
        words = user_input.split()
        for i, word in enumerate(words):
            if word.lower() in ['run', 'execute', 'command']:
                if i + 1 < len(words):
                    return ' '.join(words[i+1:])
        
        return 'MISSING_COMMAND'
