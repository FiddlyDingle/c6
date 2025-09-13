import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class MemorySystem:
    """
    Simple memory system using SQLite for Phase 1
    Stores conversations, user facts, and basic context
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config['memory']['database_path']
        self.max_conversations = config['memory']['max_conversations']
        self.importance_threshold = config['memory']['importance_threshold']
        self.logger = logging.getLogger(__name__)
        
        # Ensure data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Conversations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        user_input TEXT NOT NULL,
                        ai_response TEXT NOT NULL,
                        importance_score REAL DEFAULT 0.5,
                        context TEXT,
                        tools_used TEXT
                    )
                ''')
                
                # User facts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_facts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fact_type TEXT NOT NULL,
                        fact_key TEXT NOT NULL,
                        fact_value TEXT NOT NULL,
                        confidence REAL DEFAULT 1.0,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(fact_type, fact_key)
                    )
                ''')
                
                # Goals and contexts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS goals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        goal_type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        status TEXT DEFAULT 'active',
                        priority INTEGER DEFAULT 1,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def store_conversation(self, user_input: str, ai_response: str, 
                          tools_used: List[str] = None, context: Dict = None) -> int:
        """Store a conversation in memory"""
        try:
            importance_score = self._calculate_importance(user_input, ai_response)
            tools_json = json.dumps(tools_used) if tools_used else None
            context_json = json.dumps(context) if context else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations 
                    (user_input, ai_response, importance_score, context, tools_used)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_input, ai_response, importance_score, context_json, tools_json))
                
                conversation_id = cursor.lastrowid
                conn.commit()
                
                self.logger.debug(f"Stored conversation {conversation_id} with importance {importance_score}")
                return conversation_id
                
        except Exception as e:
            self.logger.error(f"Failed to store conversation: {e}")
            return -1
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent conversations for context"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_input, ai_response, timestamp, importance_score, context, tools_used
                    FROM conversations
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
                results = []
                for row in cursor.fetchall():
                    context = json.loads(row[4]) if row[4] else {}
                    tools_used = json.loads(row[5]) if row[5] else []
                    
                    results.append({
                        'user_input': row[0],
                        'ai_response': row[1],
                        'timestamp': row[2],
                        'importance_score': row[3],
                        'context': context,
                        'tools_used': tools_used
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get recent conversations: {e}")
            return []
    
    def store_user_fact(self, fact_type: str, fact_key: str, fact_value: str, confidence: float = 1.0) -> bool:
        """Store a fact about the user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_facts
                    (fact_type, fact_key, fact_value, confidence)
                    VALUES (?, ?, ?, ?)
                ''', (fact_type, fact_key, fact_value, confidence))
                
                conn.commit()
                self.logger.debug(f"Stored user fact: {fact_type}.{fact_key} = {fact_value}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store user fact: {e}")
            return False
    
    def get_user_facts(self, fact_type: str = None) -> Dict[str, Any]:
        """Retrieve user facts, optionally filtered by type"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if fact_type:
                    cursor.execute('''
                        SELECT fact_key, fact_value, confidence
                        FROM user_facts
                        WHERE fact_type = ?
                        ORDER BY timestamp DESC
                    ''', (fact_type,))
                else:
                    cursor.execute('''
                        SELECT fact_type, fact_key, fact_value, confidence
                        FROM user_facts
                        ORDER BY timestamp DESC
                    ''')
                
                facts = {}
                for row in cursor.fetchall():
                    if fact_type:
                        facts[row[0]] = {'value': row[1], 'confidence': row[2]}
                    else:
                        if row[0] not in facts:
                            facts[row[0]] = {}
                        facts[row[0]][row[1]] = {'value': row[2], 'confidence': row[3]}
                
                return facts
                
        except Exception as e:
            self.logger.error(f"Failed to get user facts: {e}")
            return {}
    
    def _calculate_importance(self, user_input: str, ai_response: str) -> float:
        """Calculate importance score for a conversation (simple version for Phase 1)"""
        score = 0.5  # Base score
        
        # Longer conversations tend to be more important
        if len(user_input) > 100:
            score += 0.1
        if len(ai_response) > 200:
            score += 0.1
        
        # Questions tend to be more important
        if '?' in user_input:
            score += 0.1
        
        # Code or technical content
        if any(keyword in user_input.lower() for keyword in ['code', 'function', 'error', 'debug', 'build']):
            score += 0.2
        
        # Personal information
        if any(keyword in user_input.lower() for keyword in ['my', 'i am', 'i like', 'i need', 'i want']):
            score += 0.1
        
        return min(1.0, score)
    
    def cleanup_old_conversations(self) -> None:
        """Remove old conversations to keep database size manageable"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Keep only the most recent conversations above the limit
                cursor.execute('''
                    DELETE FROM conversations
                    WHERE id NOT IN (
                        SELECT id FROM conversations
                        ORDER BY importance_score DESC, timestamp DESC
                        LIMIT ?
                    )
                ''', (self.max_conversations,))
                
                deleted = cursor.rowcount
                conn.commit()
                
                if deleted > 0:
                    self.logger.info(f"Cleaned up {deleted} old conversations")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup conversations: {e}")
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context for the agent"""
        recent = self.get_recent_conversations(5)
        facts = self.get_user_facts()
        
        return {
            'recent_conversations': recent,
            'user_facts': facts,
            'conversation_count': len(recent)
        }
