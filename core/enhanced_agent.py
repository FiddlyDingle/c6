import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
try:
    from .enhanced_memory import EnhancedMemorySystem as MemorySystem
except ImportError:
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
            context = self._gather_context(analysis, user_input)
            
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
    
    def _gather_context(self, analysis: Dict[str, Any], user_input: str = None) -> Dict[str, Any]:
        """Enhanced context gathering with semantic search"""
        context = {
            'recent_conversations': [],
            'user_facts': {},
            'available_tools': [],
            'loaded_plugins': [],
            'semantic_matches': []
        }
        
        # Get enhanced memory context with semantic search
        if hasattr(self.memory, 'get_conversation_context'):
            # Use enhanced memory system
            memory_context = self.memory.get_conversation_context(query=user_input)
            context.update(memory_context)
        else:
            # Fallback to basic memory system
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
        """Build enhanced prompt with semantic search and plugin context"""
        prompt_parts = []
        
        prompt_parts.append("You are Cerb, an intelligent AI assistant with enhanced capabilities and memory.")
        
        # Add semantic matches if available
        if context.get('semantic_matches'):
            prompt_parts.append("\nRelevant past conversations:")
            for match in context['semantic_matches'][:2]:  # Top 2 most relevant
                similarity = match.get('similarity_score', 0)
                if similarity > 0.7:  # Only include highly relevant matches
                    prompt_parts.append(f"Previous context (similarity: {similarity:.2f}):")
                    prompt_parts.append(f"User: {match['user_input'][:150]}...")
                    prompt_parts.append(f"Assistant: {match['ai_response'][:150]}...")
        
        # Add recent conversation context
        if context.get('recent_conversations'):
            prompt_parts.append("\nRecent conversation context:")
            for conv in context['recent_conversations'][-3:]:
                prompt_parts.append(f"User: {conv['user_input']}")
                prompt_parts.append(f"Assistant: {conv['ai_response'][:200]}...")
        
        # Add session topics if available
        if context.get('session_topics'):
            prompt_parts.append(f"\nCurrent session topics: {', '.join(context['session_topics'])}")
        
        # Add user facts
        if context.get('user_facts'):
            prompt_parts.append("\nKnown user information:")
            fact_count = 0
            for fact_type, facts in context['user_facts'].items():
                for key, value in facts.items():
                    if fact_count < 5:  # Limit to 5 most important facts
                        prompt_parts.append(f"- {fact_type}.{key}: {value['value']}")
                        fact_count += 1
        
        # Add tool results
        if tool_results:
            prompt_parts.append("\nTool execution results:")
            for result in tool_results:
                if result.get('success'):
                    prompt_parts.append(f"Tool {result['tool']} succeeded: {str(result['result'])[:300]}")
                else:
                    prompt_parts.append(f"Tool {result['tool']} failed: {result.get('error', 'Unknown error')}")
        
        # Current user input
        prompt_parts.append(f"\nCurrent user request: {user_input}")
        
        # Enhanced instruction
        prompt_parts.append("\nProvide a helpful, accurate response using the context above. Be conversational, reference relevant past conversations when helpful, and mention any tools used.")
        
        return "\n".join(prompt_parts)
    
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
        return "\n".join(response_parts)
    
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
            if '/' in word or '\\' in word:
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
