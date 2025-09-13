import sqlite3
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available - falling back to SQLite-only memory")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Sentence-transformers not available - using basic text matching")

class EnhancedMemorySystem:
    """
    Enhanced memory system with ChromaDB for semantic search
    Falls back gracefully to SQLite-only if ChromaDB unavailable
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config['memory']['database_path']
        self.max_conversations = config['memory']['max_conversations']
        self.importance_threshold = config['memory']['importance_threshold']
        self.logger = logging.getLogger(__name__)
        
        # Ensure data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database
        self._initialize_database()
        
        # Initialize ChromaDB if available
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        
        if CHROMADB_AVAILABLE:
            self._initialize_chromadb()
        
        if EMBEDDINGS_AVAILABLE and CHROMADB_AVAILABLE:
            self._initialize_embeddings()
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database with enhanced tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Enhanced conversations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        user_input TEXT NOT NULL,
                        ai_response TEXT NOT NULL,
                        importance_score REAL DEFAULT 0.5,
                        context TEXT,
                        tools_used TEXT,
                        embeddings_id TEXT,
                        topic_tags TEXT,
                        sentiment REAL DEFAULT 0.0,
                        session_id TEXT
                    )
                ''')
                
                # Enhanced user facts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_facts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fact_type TEXT NOT NULL,
                        fact_key TEXT NOT NULL,
                        fact_value TEXT NOT NULL,
                        confidence REAL DEFAULT 1.0,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        source_conversation_id INTEGER,
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
                        progress REAL DEFAULT 0.0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        target_date DATETIME
                    )
                ''')
                
                # Topic clustering table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS topics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic_name TEXT NOT NULL UNIQUE,
                        conversation_count INTEGER DEFAULT 0,
                        last_mentioned DATETIME DEFAULT CURRENT_TIMESTAMP,
                        importance REAL DEFAULT 0.5
                    )
                ''')
                
                # Session tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                        end_time DATETIME,
                        conversation_count INTEGER DEFAULT 0,
                        topics TEXT
                    )
                ''')
                
                conn.commit()
                self.logger.info("Enhanced database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB for semantic search"""
        try:
            # Create ChromaDB data directory
            chroma_path = Path(self.db_path).parent / "chroma"
            chroma_path.mkdir(exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection("conversations")
                self.logger.info("Connected to existing ChromaDB collection")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name="conversations",
                    metadata={"description": "Conversation embeddings for semantic search"}
                )
                self.logger.info("Created new ChromaDB collection")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    def _initialize_embeddings(self) -> None:
        """Initialize sentence transformer model for embeddings"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def store_conversation(self, user_input: str, ai_response: str, 
                          tools_used: List[str] = None, context: Dict = None,
                          session_id: str = None) -> int:
        """Store a conversation with enhanced metadata and embeddings"""
        try:
            # Calculate enhanced importance score
            importance_score = self._calculate_enhanced_importance(user_input, ai_response, tools_used)
            
            # Extract topics and sentiment
            topic_tags = self._extract_topics(user_input, ai_response)
            sentiment = self._analyze_sentiment(user_input)
            
            # Generate session ID if not provided
            if not session_id:
                session_id = self._get_current_session_id()
            
            # Prepare data for storage
            tools_json = json.dumps(tools_used) if tools_used else None
            context_json = json.dumps(context) if context else None
            topic_tags_json = json.dumps(topic_tags) if topic_tags else None
            
            # Store in SQLite
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations 
                    (user_input, ai_response, importance_score, context, tools_used,
                     topic_tags, sentiment, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_input, ai_response, importance_score, context_json, 
                      tools_json, topic_tags_json, sentiment, session_id))
                
                conversation_id = cursor.lastrowid
                conn.commit()
            
            # Store embeddings in ChromaDB if available
            if self.collection and self.embedding_model:
                self._store_embeddings(conversation_id, user_input, ai_response)
            
            # Update topic tracking
            self._update_topics(topic_tags)
            
            # Extract and store user facts
            self._extract_and_store_facts(user_input, ai_response, conversation_id)
            
            self.logger.debug(f"Stored enhanced conversation {conversation_id} with importance {importance_score}")
            return conversation_id
            
        except Exception as e:
            self.logger.error(f"Failed to store conversation: {e}")
            return -1
    
    def _store_embeddings(self, conversation_id: int, user_input: str, ai_response: str) -> None:
        """Store conversation embeddings in ChromaDB"""
        try:
            # Combine input and response for embedding
            combined_text = f"User: {user_input}\nAssistant: {ai_response}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(combined_text).tolist()
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[combined_text],
                ids=[str(conversation_id)],
                metadatas=[{
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "user_input": user_input[:500],  # Truncate for metadata
                    "ai_response": ai_response[:500]
                }]
            )
            
            # Update SQLite with embedding ID
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE conversations SET embeddings_id = ? WHERE id = ?
                ''', (str(conversation_id), conversation_id))
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to store embeddings for conversation {conversation_id}: {e}")
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search conversations using semantic similarity"""
        if not (self.collection and self.embedding_model):
            self.logger.warning("Semantic search not available - falling back to text search")
            return self.text_search(query, limit)
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to standard format
            conversations = []
            for i in range(len(results['ids'][0])):
                conversation_id = int(results['ids'][0][i])
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                conversations.append({
                    'conversation_id': conversation_id,
                    'user_input': metadata['user_input'],
                    'ai_response': metadata['ai_response'],
                    'timestamp': metadata['timestamp'],
                    'similarity_score': 1 - distance,  # Convert distance to similarity
                    'search_type': 'semantic'
                })
            
            return conversations
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return self.text_search(query, limit)
    
    def text_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback text-based search"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, user_input, ai_response, timestamp, importance_score
                    FROM conversations
                    WHERE user_input LIKE ? OR ai_response LIKE ?
                    ORDER BY importance_score DESC, timestamp DESC
                    LIMIT ?
                ''', (f'%{query}%', f'%{query}%', limit))
                
                conversations = []
                for row in cursor.fetchall():
                    conversations.append({
                        'conversation_id': row[0],
                        'user_input': row[1],
                        'ai_response': row[2],
                        'timestamp': row[3],
                        'importance_score': row[4],
                        'search_type': 'text'
                    })
                
                return conversations
                
        except Exception as e:
            self.logger.error(f"Text search failed: {e}")
            return []
    
    def get_recent_conversations(self, limit: int = 10, session_id: str = None) -> List[Dict[str, Any]]:
        """Enhanced recent conversations retrieval"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if session_id:
                    cursor.execute('''
                        SELECT user_input, ai_response, timestamp, importance_score, 
                               context, tools_used, topic_tags, sentiment
                        FROM conversations
                        WHERE session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (session_id, limit))
                else:
                    cursor.execute('''
                        SELECT user_input, ai_response, timestamp, importance_score, 
                               context, tools_used, topic_tags, sentiment
                        FROM conversations
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (limit,))
                
                results = []
                for row in cursor.fetchall():
                    context = json.loads(row[4]) if row[4] else {}
                    tools_used = json.loads(row[5]) if row[5] else []
                    topic_tags = json.loads(row[6]) if row[6] else []
                    
                    results.append({
                        'user_input': row[0],
                        'ai_response': row[1],
                        'timestamp': row[2],
                        'importance_score': row[3],
                        'context': context,
                        'tools_used': tools_used,
                        'topic_tags': topic_tags,
                        'sentiment': row[7]
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get recent conversations: {e}")
            return []
    
    def _calculate_enhanced_importance(self, user_input: str, ai_response: str, tools_used: List[str] = None) -> float:
        """Enhanced importance calculation"""
        score = 0.5  # Base score
        
        # Length indicators
        if len(user_input) > 100:
            score += 0.1
        if len(ai_response) > 200:
            score += 0.1
        
        # Question indicators
        if '?' in user_input:
            score += 0.1
        
        # Technical content
        technical_keywords = ['code', 'function', 'error', 'debug', 'build', 'install', 'configure']
        if any(keyword in user_input.lower() for keyword in technical_keywords):
            score += 0.2
        
        # Personal information
        personal_keywords = ['my', 'i am', 'i like', 'i need', 'i want', 'remember']
        if any(keyword in user_input.lower() for keyword in personal_keywords):
            score += 0.15
        
        # Tool usage increases importance
        if tools_used:
            score += min(0.2, len(tools_used) * 0.05)
        
        # Learning/teaching content
        learning_keywords = ['learn', 'teach', 'explain', 'understand', 'how to']
        if any(keyword in user_input.lower() for keyword in learning_keywords):
            score += 0.1
        
        # Project/work related
        work_keywords = ['project', 'work', 'task', 'deadline', 'meeting', 'client']
        if any(keyword in user_input.lower() for keyword in work_keywords):
            score += 0.15
        
        return min(1.0, score)
    
    def _extract_topics(self, user_input: str, ai_response: str) -> List[str]:
        """Extract topics from conversation"""
        topics = []
        
        # Simple keyword-based topic extraction
        topic_keywords = {
            'programming': ['code', 'function', 'programming', 'python', 'javascript', 'software'],
            'project_management': ['project', 'task', 'deadline', 'milestone', 'planning'],
            'learning': ['learn', 'study', 'understand', 'explain', 'tutorial'],
            'work': ['work', 'job', 'career', 'professional', 'business'],
            'personal': ['personal', 'life', 'family', 'hobby', 'interest'],
            'technical_support': ['error', 'bug', 'fix', 'troubleshoot', 'debug'],
            'file_management': ['file', 'folder', 'directory', 'save', 'open'],
            'web_research': ['search', 'web', 'internet', 'website', 'research']
        }
        
        combined_text = f"{user_input} {ai_response}".lower()
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _analyze_sentiment(self, user_input: str) -> float:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'happy', 'pleased', 'thanks', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'angry', 'frustrated', 'error', 'problem']
        
        user_lower = user_input.lower()
        positive_count = sum(1 for word in positive_words if word in user_lower)
        negative_count = sum(1 for word in negative_words if word in user_lower)
        
        if positive_count == negative_count:
            return 0.0  # Neutral
        elif positive_count > negative_count:
            return min(1.0, (positive_count - negative_count) * 0.2)
        else:
            return max(-1.0, (positive_count - negative_count) * 0.2)
    
    def _get_current_session_id(self) -> str:
        """Generate or get current session ID"""
        # Simple session ID based on date
        return datetime.now().strftime("%Y%m%d_%H")
    
    def _update_topics(self, topic_tags: List[str]) -> None:
        """Update topic tracking"""
        if not topic_tags:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for topic in topic_tags:
                    cursor.execute('''
                        INSERT OR IGNORE INTO topics (topic_name) VALUES (?)
                    ''', (topic,))
                    
                    cursor.execute('''
                        UPDATE topics 
                        SET conversation_count = conversation_count + 1,
                            last_mentioned = CURRENT_TIMESTAMP
                        WHERE topic_name = ?
                    ''', (topic,))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update topics: {e}")
    
    def _extract_and_store_facts(self, user_input: str, ai_response: str, conversation_id: int) -> None:
        """Extract and store user facts from conversation"""
        facts = []
        
        # Simple fact extraction patterns
        user_lower = user_input.lower()
        
        # Name extraction
        if 'my name is' in user_lower or 'i am' in user_lower:
            # This is simplified - in practice you'd want more sophisticated NLP
            pass
        
        # Preference extraction
        if 'i like' in user_lower:
            # Extract preferences
            pass
        
        # For now, just store that a conversation happened
        # You can enhance this with more sophisticated NLP later
        
    def get_conversation_context(self, query: str = None, session_id: str = None) -> Dict[str, Any]:
        """Get enhanced conversation context"""
        context = {
            'recent_conversations': [],
            'user_facts': {},
            'conversation_count': 0,
            'session_topics': [],
            'semantic_matches': []
        }
        
        # Get recent conversations
        recent = self.get_recent_conversations(5, session_id)
        context['recent_conversations'] = recent
        context['conversation_count'] = len(recent)
        
        # Get user facts
        facts = self.get_user_facts()
        context['user_facts'] = facts
        
        # If query provided, get semantic matches
        if query and self.collection:
            semantic_matches = self.semantic_search(query, 3)
            context['semantic_matches'] = semantic_matches
        
        # Get session topics
        if recent:
            all_topics = []
            for conv in recent:
                all_topics.extend(conv.get('topic_tags', []))
            context['session_topics'] = list(set(all_topics))
        
        return context
    
    def get_user_facts(self, fact_type: str = None) -> Dict[str, Any]:
        """Enhanced user facts retrieval"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if fact_type:
                    cursor.execute('''
                        SELECT fact_key, fact_value, confidence, timestamp
                        FROM user_facts
                        WHERE fact_type = ?
                        ORDER BY timestamp DESC
                    ''', (fact_type,))
                else:
                    cursor.execute('''
                        SELECT fact_type, fact_key, fact_value, confidence, timestamp
                        FROM user_facts
                        ORDER BY timestamp DESC
                    ''')
                
                facts = {}
                for row in cursor.fetchall():
                    if fact_type:
                        facts[row[0]] = {
                            'value': row[1], 
                            'confidence': row[2],
                            'timestamp': row[3]
                        }
                    else:
                        if row[0] not in facts:
                            facts[row[0]] = {}
                        facts[row[0]][row[1]] = {
                            'value': row[2], 
                            'confidence': row[3],
                            'timestamp': row[4]
                        }
                
                return facts
                
        except Exception as e:
            self.logger.error(f"Failed to get user facts: {e}")
            return {}
    
    def store_user_fact(self, fact_type: str, fact_key: str, fact_value: str, 
                       confidence: float = 1.0, source_conversation_id: int = None) -> bool:
        """Enhanced user fact storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_facts
                    (fact_type, fact_key, fact_value, confidence, source_conversation_id)
                    VALUES (?, ?, ?, ?, ?)
                ''', (fact_type, fact_key, fact_value, confidence, source_conversation_id))
                
                conn.commit()
                self.logger.debug(f"Stored enhanced user fact: {fact_type}.{fact_key} = {fact_value}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store user fact: {e}")
            return False
    
    def cleanup_old_conversations(self) -> None:
        """Enhanced cleanup with importance preservation"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Keep high-importance conversations regardless of age
                cursor.execute('''
                    SELECT COUNT(*) FROM conversations
                ''')
                total_conversations = cursor.fetchone()[0]
                
                if total_conversations <= self.max_conversations:
                    return
                
                # Get conversations to delete (low importance, old)
                cursor.execute('''
                    SELECT id FROM conversations
                    WHERE importance_score < ? 
                    ORDER BY importance_score ASC, timestamp ASC
                    LIMIT ?
                ''', (self.importance_threshold, total_conversations - self.max_conversations))
                
                conversations_to_delete = [row[0] for row in cursor.fetchall()]
                
                if conversations_to_delete:
                    # Delete from ChromaDB first
                    if self.collection:
                        try:
                            self.collection.delete(ids=[str(cid) for cid in conversations_to_delete])
                        except Exception as e:
                            self.logger.warning(f"Failed to delete from ChromaDB: {e}")
                    
                    # Delete from SQLite
                    placeholders = ','.join(['?' for _ in conversations_to_delete])
                    cursor.execute(f'''
                        DELETE FROM conversations WHERE id IN ({placeholders})
                    ''', conversations_to_delete)
                    
                    conn.commit()
                    self.logger.info(f"Cleaned up {len(conversations_to_delete)} old conversations")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup conversations: {e}")

# Backward compatibility - alias the new class to the old name
MemorySystem = EnhancedMemorySystem
