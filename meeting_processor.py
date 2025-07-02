import os
import json
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from pathlib import Path
import httpx
# Remove Azure imports and replace with OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import sqlite3
import faiss
from collections import defaultdict
import threading
import time
from dotenv import load_dotenv

# Force reload of environment variables
load_dotenv(override=True)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check API key availability at startup
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
logger.info(f"OpenAI API key loaded: {openai_api_key[:15]}...{openai_api_key[-10:]}")

project_id = "openai-meeting-processor"  # Simple project ID for personal use
tiktoken_cache_dir = os.path.abspath("tiktoken_cache")
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# --- Dummy Auth Function (for compatibility) ---
def get_access_token():
    """
    Dummy function to maintain compatibility with existing code.
    OpenAI API uses API key authentication, not tokens.
    """
    return "dummy_token_for_compatibility"

# --- OpenAI LLM Client ---
def get_llm(access_token: str = None):
    """
    Get OpenAI LLM client. access_token parameter is kept for compatibility
    but not used since OpenAI uses API key authentication.
    """
    # Get fresh API key each time to avoid caching issues
    current_api_key = os.getenv("OPENAI_API_KEY")
    if not current_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return ChatOpenAI(
        model="gpt-4o",  # Using GPT-4o model
        openai_api_key=current_api_key,
        temperature=0,
        max_tokens=4000,  # Adjust as needed
        request_timeout=60
    )

# --- OpenAI Embedding Model ---
def get_embedding_model(access_token: str = None):
    """
    Get OpenAI embedding model. access_token parameter is kept for compatibility
    but not used since OpenAI uses API key authentication.
    """
    # Get fresh API key each time to avoid caching issues
    current_api_key = os.getenv("OPENAI_API_KEY")
    if not current_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return OpenAIEmbeddings(
        model="text-embedding-3-large",  # Using text-embedding-3-large
        openai_api_key=current_api_key,
        dimensions=3072  # text-embedding-3-large dimension
    )

# Initialize global variables (keeping same structure)
access_token = get_access_token()
embedding_model = get_embedding_model(access_token)
llm = get_llm(access_token)

@dataclass
class DocumentChunk:
    """Structure to hold document chunk information"""
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    embedding: Optional[np.ndarray] = None

@dataclass
class MeetingDocument:
    """Structure to hold meeting document information"""
    document_id: str
    filename: str
    date: datetime
    title: str
    content: str
    content_summary: str  # Condensed summary for metadata
    main_topics: List[str]
    past_events: List[str]
    future_actions: List[str]
    participants: List[str]
    chunk_count: int = 0
    file_size: int = 0

class VectorDatabase:
    """Vector database using FAISS for similarity search and SQLite for metadata"""
    
    def __init__(self, db_path: str = "meeting_documents.db", index_path: str = "vector_index.faiss"):
        self.db_path = db_path
        self.index_path = index_path
        self.dimension = 3072  # text-embedding-3-large dimension
        self.index = None
        self.chunk_metadata = {}
        self.document_metadata = {}
        self._init_database()
        self._load_or_create_index()
    
    def _init_database(self):
        """Initialize SQLite database for metadata storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                date TIMESTAMP NOT NULL,
                title TEXT,
                content_summary TEXT,
                main_topics TEXT,
                past_events TEXT,
                future_actions TEXT,
                participants TEXT,
                chunk_count INTEGER,
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_char INTEGER,
                end_char INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (document_id)
            )
        ''')
        
        # Create indexes for faster searches
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_date ON documents(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_filename ON documents(filename)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_document ON chunks(document_id)')
        
        conn.commit()
        conn.close()
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Created new FAISS index")
    
    def add_document(self, document: MeetingDocument, chunks: List[DocumentChunk]):
        """Add document and its chunks to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert document metadata
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (document_id, filename, date, title, content_summary, main_topics, 
                 past_events, future_actions, participants, chunk_count, file_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document.document_id,
                document.filename,
                document.date,
                document.title,
                document.content_summary,
                json.dumps(document.main_topics),
                json.dumps(document.past_events),
                json.dumps(document.future_actions),
                json.dumps(document.participants),
                document.chunk_count,
                document.file_size
            ))
            
            # Prepare vectors and chunk data for batch insertion
            vectors = []
            chunk_data = []
            
            for chunk in chunks:
                if chunk.embedding is not None:
                    vectors.append(chunk.embedding)
                    chunk_data.append((
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.filename,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.start_char,
                        chunk.end_char
                    ))
            
            # Insert chunks in batch
            cursor.executemany('''
                INSERT OR REPLACE INTO chunks 
                (chunk_id, document_id, filename, chunk_index, content, start_char, end_char)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', chunk_data)
            
            # Add vectors to FAISS index
            if vectors:
                vectors_array = np.array(vectors).astype('float32')
                # Normalize vectors for cosine similarity
                faiss.normalize_L2(vectors_array)
                self.index.add(vectors_array)
                
                # Store chunk metadata mapping
                start_idx = self.index.ntotal - len(vectors)
                for i, chunk in enumerate(chunks):
                    if chunk.embedding is not None:
                        self.chunk_metadata[start_idx + i] = chunk.chunk_id
            
            conn.commit()
            self.document_metadata[document.document_id] = document
            logger.info(f"Added document {document.filename} with {len(chunks)} chunks")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error adding document {document.filename}: {e}")
            raise
        finally:
            conn.close()
    
    def search_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        """Search for similar chunks using FAISS"""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query vector
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search in FAISS index
        similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx in self.chunk_metadata:
                chunk_id = self.chunk_metadata[idx]
                similarity = float(similarities[0][i])
                results.append((chunk_id, similarity))
        
        return results
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[DocumentChunk]:
        """Retrieve chunks by their IDs"""
        if not chunk_ids:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join(['?' for _ in chunk_ids])
        cursor.execute(f'''
            SELECT chunk_id, document_id, filename, chunk_index, content, start_char, end_char
            FROM chunks WHERE chunk_id IN ({placeholders})
        ''', chunk_ids)
        
        chunks = []
        for row in cursor.fetchall():
            chunk = DocumentChunk(
                chunk_id=row[0],
                document_id=row[1],
                filename=row[2],
                chunk_index=row[3],
                content=row[4],
                start_char=row[5],
                end_char=row[6]
            )
            chunks.append(chunk)
        
        conn.close()
        return chunks
    
    def get_documents_by_timeframe(self, timeframe: str) -> List[MeetingDocument]:
        """Get documents filtered by timeframe"""
        now = datetime.now()
        timeframe_map = {
            'last_week': timedelta(weeks=1),
            'last_month': timedelta(days=30),
            'last_3_months': timedelta(days=90),
            'last_quarter': timedelta(days=90),
            'last_6_months': timedelta(days=180),
            'last_year': timedelta(days=365),
            'recent': timedelta(days=30)
        }
        
        if timeframe not in timeframe_map:
            return list(self.document_metadata.values())
        
        cutoff = now - timeframe_map[timeframe]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT document_id, filename, date, title, content_summary, main_topics,
                   past_events, future_actions, participants, chunk_count, file_size
            FROM documents WHERE date >= ?
            ORDER BY date DESC
        ''', (cutoff,))
        
        documents = []
        for row in cursor.fetchall():
            doc = MeetingDocument(
                document_id=row[0],
                filename=row[1],
                date=datetime.fromisoformat(row[2]),
                title=row[3],
                content="",  # We don't load full content for filtering
                content_summary=row[4],
                main_topics=json.loads(row[5]) if row[5] else [],
                past_events=json.loads(row[6]) if row[6] else [],
                future_actions=json.loads(row[7]) if row[7] else [],
                participants=json.loads(row[8]) if row[8] else [],
                chunk_count=row[9],
                file_size=row[10]
            )
            documents.append(doc)
        
        conn.close()
        return documents
    
    def keyword_search_chunks(self, keywords: List[str], limit: int = 50) -> List[str]:
        """Perform keyword search on chunk content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build search query
        search_conditions = []
        params = []
        for keyword in keywords:
            search_conditions.append("content LIKE ?")
            params.append(f"%{keyword}%")
        
        query = f'''
            SELECT chunk_id FROM chunks 
            WHERE {" OR ".join(search_conditions)}
            LIMIT ?
        '''
        params.append(limit)
        
        cursor.execute(query, params)
        chunk_ids = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return chunk_ids
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents with metadata for document selection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT document_id, filename, date, title, content_summary, file_size, chunk_count
            FROM documents
            ORDER BY date DESC
        ''')
        
        documents = []
        for row in cursor.fetchall():
            doc = {
                'document_id': row[0],
                'filename': row[1],
                'date': row[2],
                'title': row[3],
                'content_summary': row[4],
                'file_size': row[5],
                'chunk_count': row[6]
            }
            documents.append(doc)
        
        conn.close()
        return documents
    
    def save_index(self):
        """Save FAISS index to disk"""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")

class EnhancedMeetingDocumentProcessor:
    """Enhanced Meeting Document Processor with Vector Database Support"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the enhanced processor"""
        global llm, embedding_model, access_token
        self.llm = llm
        self.embedding_model = embedding_model
        self.access_token = access_token
        self.token_expiry = datetime.now() + timedelta(hours=1)
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
        
        # Vector database
        self.vector_db = VectorDatabase()
        
        logger.info("Enhanced Meeting Document Processor initialized with OpenAI")
    
    def refresh_clients(self):
        """Refresh clients - simplified for OpenAI (no token refresh needed)"""
        try:
            global access_token, llm, embedding_model
            
            # Force reload environment variables
            load_dotenv(override=True)
            
            # Check API key
            current_api_key = os.getenv("OPENAI_API_KEY")
            if not current_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            logger.info(f"Refreshing with API key: {current_api_key[:15]}...{current_api_key[-10:]}")
            
            # For OpenAI, we don't need to refresh tokens since we use API keys
            # But we can recreate the clients if needed
            access_token = get_access_token()  # This just returns a dummy token
            llm = get_llm(access_token)
            embedding_model = get_embedding_model(access_token)
            
            self.llm = llm
            self.embedding_model = embedding_model
            self.access_token = access_token
            
            logger.info("OpenAI clients refreshed successfully")
            
        except Exception as e:
            logger.error(f"Failed to refresh clients: {e}")
            raise
    
    def extract_date_from_filename(self, filename: str) -> datetime:
        """Extract date from filename pattern with multiple formats"""
        patterns = [
            r'(\d{8})_(\d{6})',  # YYYYMMDD_HHMMSS
            r'(\d{8})',          # YYYYMMDD
            r'(\d{4}-\d{2}-\d{2})', # YYYY-MM-DD
            r'(\d{2}/\d{2}/\d{4})'  # MM/DD/YYYY
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    if '_' in match.group(0):
                        return datetime.strptime(match.group(0), "%Y%m%d_%H%M%S")
                    elif '-' in match.group(0):
                        return datetime.strptime(match.group(0), "%Y-%m-%d")
                    elif '/' in match.group(0):
                        return datetime.strptime(match.group(0), "%m/%d/%Y")
                    else:
                        return datetime.strptime(match.group(0), "%Y%m%d")
                except ValueError:
                    continue
        
        logger.warning(f"Could not extract date from filename: {filename}, using current date")
        return datetime.now()
    
    def read_document_content(self, file_path: str) -> str:
        """Read document content from file"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
            elif file_ext == '.docx':
                try:
                    from docx import Document
                    doc = Document(file_path)
                    content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                except ImportError:
                    logger.error("python-docx not installed. Cannot process .docx files.")
                    return ""
                    
            elif file_ext == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        content = ""
                        for page in pdf_reader.pages:
                            content += page.extract_text() + "\n"
                except ImportError:
                    logger.error("PyPDF2 not installed. Cannot process .pdf files.")
                    return ""
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                return ""
            
            return content if content.strip() else ""
                
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {e}")
            return ""
    
    def create_content_summary(self, content: str, max_length: int = 500) -> str:
        """Create a condensed summary of the content"""
        try:
            summary_prompt = f"""
            Create a concise summary of this meeting document in 2-3 sentences (max {max_length} characters).
            Focus on the main purpose, key decisions, and outcomes.
            
            Content: {content[:2000]}...
            
            Summary:
            """
            
            messages = [
                SystemMessage(content="You are a meeting summarization expert. Create concise, informative summaries."),
                HumanMessage(content=summary_prompt)
            ]
            
            response = self.llm.invoke(messages)
            summary = response.content.strip()
            
            return summary[:max_length] if len(summary) > max_length else summary
            
        except Exception as e:
            logger.error(f"Error creating content summary: {e}")
            # Fallback to truncated content
            return content[:max_length] + "..." if len(content) > max_length else content
    
    def parse_document_content(self, content: str, filename: str) -> MeetingDocument:
        """Parse a meeting document and extract structured information"""
        doc_date = self.extract_date_from_filename(filename)
        document_id = f"{filename}_{doc_date.strftime('%Y%m%d_%H%M%S')}"
        
        parsing_prompt = f"""
        You are an expert document analyst. Analyze this meeting document and extract information in valid JSON format.
        
        Document content:
        {content[:4000]}{"..." if len(content) > 4000 else ""}
        
        Extract:
        1. "title": Clear, descriptive meeting title (2-8 words)
        2. "main_topics": Array of main topics discussed
        3. "past_events": Array of past events/completed items mentioned
        4. "future_actions": Array of upcoming actions/planned activities
        5. "participants": Array of participant names mentioned
        
        Return only valid JSON with no additional text.
        """
        
        try:
            messages = [
                SystemMessage(content="You are a document parsing assistant. Always return valid JSON format with no additional text."),
                HumanMessage(content=parsing_prompt)
            ]
            
            response = self.llm.invoke(messages)
            content_str = response.content.strip()
            
            # Clean JSON response
            if content_str.startswith('```json'):
                content_str = content_str[7:-3].strip()
            elif content_str.startswith('```'):
                content_str = content_str[3:-3].strip()
            
            parsed_data = json.loads(content_str)
            
            # Create content summary
            content_summary = self.create_content_summary(content)
            
            return MeetingDocument(
                document_id=document_id,
                filename=filename,
                date=doc_date,
                title=parsed_data.get('title', f'Meeting - {doc_date.strftime("%Y-%m-%d")}'),
                content=content,
                content_summary=content_summary,
                main_topics=parsed_data.get('main_topics', []),
                past_events=parsed_data.get('past_events', []),
                future_actions=parsed_data.get('future_actions', []),
                participants=parsed_data.get('participants', []),
                file_size=len(content)
            )
            
        except Exception as e:
            logger.error(f"Error parsing document {filename}: {e}")
            return self._create_fallback_document(content, filename, doc_date, document_id)
    
    def _create_fallback_document(self, content: str, filename: str, doc_date: datetime, document_id: str) -> MeetingDocument:
        """Create a fallback document when parsing fails"""
        content_summary = self.create_content_summary(content)
        
        return MeetingDocument(
            document_id=document_id,
            filename=filename,
            date=doc_date,
            title=f"Meeting - {doc_date.strftime('%Y-%m-%d')}",
            content=content,
            content_summary=content_summary,
            main_topics=[],
            past_events=[],
            future_actions=[],
            participants=[],
            file_size=len(content)
        )
    
    def chunk_document(self, document: MeetingDocument) -> List[DocumentChunk]:
        """Split document into chunks with embeddings"""
        # Split content into chunks
        chunks = self.text_splitter.split_text(document.content)
        
        document_chunks = []
        current_pos = 0
        
        for i, chunk_content in enumerate(chunks):
            chunk_id = f"{document.document_id}_chunk_{i}"
            
            # Find start position in original content
            start_char = document.content.find(chunk_content, current_pos)
            if start_char == -1:
                start_char = current_pos
            
            end_char = start_char + len(chunk_content)
            current_pos = end_char
            
            # Generate embedding for chunk
            try:
                embedding = self.embedding_model.embed_query(chunk_content)
                embedding_array = np.array(embedding)
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {chunk_id}: {e}")
                embedding_array = np.zeros(3072)
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                filename=document.filename,
                chunk_index=i,
                content=chunk_content,
                start_char=start_char,
                end_char=end_char,
                embedding=embedding_array
            )
            
            document_chunks.append(chunk)
        
        document.chunk_count = len(document_chunks)
        return document_chunks
    
    def process_documents(self, document_folder: str) -> None:
        """Process all documents in a folder with chunking and vector storage"""
        folder_path = Path(document_folder)
        
        if not folder_path.exists():
            logger.error(f"Folder {document_folder} does not exist")
            return
        
        # Get supported files
        supported_extensions = ['.docx', '.txt', '.pdf']
        doc_files = []
        for ext in supported_extensions:
            doc_files.extend(folder_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(doc_files)} documents to process...")
        
        processed_count = 0
        for doc_file in doc_files:
            try:
                logger.info(f"Processing: {doc_file.name}")
                
                # Read document content
                content = self.read_document_content(str(doc_file))
                if not content.strip():
                    logger.warning(f"No content extracted from {doc_file.name}")
                    continue
                
                # Parse document
                meeting_doc = self.parse_document_content(content, doc_file.name)
                
                # Create chunks with embeddings
                logger.info(f"Creating chunks for {doc_file.name}")
                chunks = self.chunk_document(meeting_doc)
                
                # Store in vector database
                self.vector_db.add_document(meeting_doc, chunks)
                
                processed_count += 1
                logger.info(f"Successfully processed: {doc_file.name} ({len(chunks)} chunks)")
                
            except Exception as e:
                logger.error(f"Error processing {doc_file.name}: {e}")
        
        # Save vector index
        if processed_count > 0:
            self.vector_db.save_index()
            logger.info(f"Successfully processed {processed_count} documents")
    
    def hybrid_search(self, query: str, top_k: int = 15, semantic_weight: float = 0.7) -> List[DocumentChunk]:
        """Perform hybrid search combining semantic and keyword search"""
        
        # Extract keywords from query
        keywords = [word.lower().strip() for word in query.split() if len(word) > 2]
        
        # Semantic search
        try:
            query_embedding = np.array(self.embedding_model.embed_query(query))
            semantic_results = self.vector_db.search_similar_chunks(query_embedding, top_k * 2)
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            semantic_results = []
        
        # Keyword search
        keyword_chunk_ids = self.vector_db.keyword_search_chunks(keywords, top_k)
        
        # Combine and score results
        chunk_scores = defaultdict(float)
        
        # Add semantic scores
        for chunk_id, similarity in semantic_results:
            chunk_scores[chunk_id] += similarity * semantic_weight
        
        # Add keyword scores
        keyword_weight = 1.0 - semantic_weight
        for chunk_id in keyword_chunk_ids:
            chunk_scores[chunk_id] += keyword_weight * 0.5  # Base keyword score
        
        # Get top chunks
        top_chunk_ids = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)[:top_k]
        
        # Retrieve chunk details
        return self.vector_db.get_chunks_by_ids(top_chunk_ids)
    
    def answer_query(self, query: str, document_ids: List[str] = None, context_limit: int = 10, include_context: bool = False) -> Union[str, Tuple[str, str]]:
        """Answer user query using hybrid search and intelligent context selection"""
        
        # Detect timeframe filtering
        timeframe_keywords = {
            'last week': 'last_week', 'past week': 'last_week', 'this week': 'last_week',
            'last month': 'last_month', 'past month': 'last_month', 'this month': 'last_month',
            'recent': 'recent', 'recently': 'recent',
            'last quarter': 'last_quarter', 'past quarter': 'last_quarter',
            'last 3 months': 'last_3_months', 'past 3 months': 'last_3_months',
            'last 6 months': 'last_6_months', 'last year': 'last_year'
        }
        
        detected_timeframe = None
        for keyword, timeframe in timeframe_keywords.items():
            if keyword in query.lower():
                detected_timeframe = timeframe
                break
        
        # Get relevant documents by timeframe if specified
        if detected_timeframe:
            timeframe_docs = self.vector_db.get_documents_by_timeframe(detected_timeframe)
            if not timeframe_docs:
                error_msg = f"I don't have any meeting documents from the {detected_timeframe.replace('_', ' ')} timeframe."
                return (error_msg, "") if include_context else error_msg
        
        # Perform hybrid search
        relevant_chunks = self.hybrid_search(query, top_k=context_limit * 3)
        
        # Filter chunks by document IDs if specified
        if document_ids:
            relevant_chunks = [chunk for chunk in relevant_chunks if chunk.document_id in document_ids]
            if not relevant_chunks:
                error_msg = "I don't have any relevant information in the specified documents for your question."
                return (error_msg, "") if include_context else error_msg
        elif not relevant_chunks:
            error_msg = "I don't have any relevant meeting documents to answer your question."
            return (error_msg, "") if include_context else error_msg
        
        # Group chunks by document and select best representatives
        document_chunks = defaultdict(list)
        for chunk in relevant_chunks:
            document_chunks[chunk.document_id].append(chunk)
        
        # Select best chunks from each document (max 2 per document)
        selected_chunks = []
        for doc_id, chunks in document_chunks.items():
            # Sort chunks by position to maintain context
            chunks.sort(key=lambda x: x.chunk_index)
            selected_chunks.extend(chunks[:2])  # Max 2 chunks per document
        
        # Limit total chunks
        selected_chunks = selected_chunks[:context_limit]
        
        # Build context without chunk references
        context_parts = []
        current_doc = None
        
        for chunk in selected_chunks:
            if chunk.document_id != current_doc:
                # Add document header when switching documents
                if current_doc is not None:
                    context_parts.append("\n" + "="*60 + "\n")
                
                context_parts.append(f"Document: {chunk.filename}")
                current_doc = chunk.document_id
            
            # Add content without chunk numbering
            context_parts.append(chunk.content)
        
        context = "\n".join(context_parts)
        
        # Generate answer using OpenAI GPT-4o
        answer_prompt = f"""
You are an expert AI assistant specializing in analyzing meeting documents. Based on the provided meeting document content, give a comprehensive and detailed answer to the user's question.

Meeting Document Context:
{context}

User Question: {query}

Instructions for your response:
- Provide a thorough, well-structured, and informative answer based on the meeting information
- Use specific details, dates, names, and facts from the documents
- Cite specific document names when referencing information from those documents
- DO NOT reference any "chunks", "chunk numbers", or technical document sections
- Present information naturally as if reading from complete meeting documents
- If the question asks for summaries: Provide a consolidated summary across relevant meetings with key points
- If timeline questions: Organize information chronologically with specific dates when available
- If about future plans: Focus on upcoming actions, decisions, deadlines, and planned activities
- If about past events: Highlight what has been discussed, completed, decided, or resolved
- If about specific topics: Extract and synthesize all relevant information about those topics
- If about participants: Include their roles, contributions, and responsibilities mentioned
- If information spans multiple documents, clearly indicate which information comes from which document
- Be precise about what information is available vs. what might be missing
- Use bullet points or numbered lists when appropriate for clarity
- If the answer requires information not available in the documents, clearly state what additional information would be helpful

Provide a comprehensive answer:
"""
        
        try:
            messages = [
                SystemMessage(content="You are a helpful AI assistant specializing in analyzing meeting documents. Provide comprehensive, accurate, and well-structured answers based on the meeting content provided. Be thorough and cite specific information from the documents."),
                HumanMessage(content=answer_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return (response.content, context) if include_context else response.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Try refreshing clients and retry
            try:
                self.refresh_clients()
                messages = [
                    SystemMessage(content="You are a helpful AI assistant specializing in analyzing meeting documents. Provide comprehensive, accurate, and well-structured answers based on the meeting content provided."),
                    HumanMessage(content=answer_prompt)
                ]
                response = self.llm.invoke(messages)
                return (response.content, context) if include_context else response.content
            except Exception as retry_error:
                error_msg = f"I encountered an error while processing your question. Please try again later. Error details: {str(retry_error)}"
                return (error_msg, "") if include_context else error_msg
    
    def generate_follow_up_questions(self, user_query: str, ai_response: str, context: str) -> List[str]:
        """Generate follow-up questions based on the user query, AI response, and document context"""
        try:
            follow_up_prompt = f"""
Based on the user's question, the AI response provided, and the meeting document context, generate 4-5 relevant follow-up questions that the user might want to ask next.

User's Question: {user_query}

AI Response: {ai_response}

Meeting Context: {context[:2000]}...

Generate follow-up questions that:
- Build upon the current conversation topic
- Explore related aspects mentioned in the documents
- Ask for deeper details about key points
- Inquire about timelines, next steps, or implications
- Connect to other relevant topics in the meetings

Return exactly 4-5 questions, each on a new line, without numbers or bullet points. Make them natural and conversational.
"""

            messages = [
                SystemMessage(content="You are an expert at generating relevant follow-up questions for meeting document analysis. Create questions that help users explore the meeting content more deeply."),
                HumanMessage(content=follow_up_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse the response into individual questions
            questions = [q.strip() for q in response.content.split('\n') if q.strip() and len(q.strip()) > 10]
            
            # Ensure we have 4-5 questions
            if len(questions) > 5:
                questions = questions[:5]
            elif len(questions) < 4:
                # Add generic questions if we don't have enough
                generic_questions = [
                    "What were the key decisions made in these meetings?",
                    "Who are the main stakeholders involved?",
                    "What are the next steps mentioned?",
                    "Are there any deadlines or milestones discussed?",
                    "What challenges or issues were identified?"
                ]
                questions.extend(generic_questions[:5-len(questions)])
            
            return questions[:5]
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            # Return default follow-up questions
            return [
                "What were the main decisions made in the meetings?",
                "Who are the key participants mentioned?",
                "What are the upcoming deadlines or milestones?",
                "Are there any action items assigned?",
                "What challenges were discussed?"
            ]
    
    def get_meeting_statistics(self) -> Dict[str, Any]:
        """Get simplified statistics about processed meetings"""
        try:
            conn = sqlite3.connect(self.vector_db.db_path)
            cursor = conn.cursor()
            
            # Get document counts
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            if total_docs == 0:
                return {"error": "No documents processed"}
            
            # Get date range
            cursor.execute("SELECT MIN(date), MAX(date) FROM documents")
            date_range = cursor.fetchone()
            
            # Get chunk statistics
            cursor.execute("SELECT COUNT(*), AVG(LENGTH(content)) FROM chunks")
            chunk_stats = cursor.fetchone()
            
            # Get monthly distribution (optional - can keep for internal tracking)
            cursor.execute("SELECT strftime('%Y-%m', date) as month, COUNT(*) FROM documents GROUP BY month ORDER BY month")
            monthly_counts = dict(cursor.fetchall())
            
            conn.close()
            
            stats = {
                "total_meetings": total_docs,
                "total_chunks": chunk_stats[0] if chunk_stats[0] else 0,
                "average_chunk_length": int(chunk_stats[1]) if chunk_stats[1] else 0,
                "vector_index_size": self.vector_db.index.ntotal if self.vector_db.index else 0,
                "date_range": {
                    "earliest": date_range[0] if date_range[0] else "N/A",
                    "latest": date_range[1] if date_range[1] else "N/A"
                },
                "meetings_per_month": monthly_counts,
                "openai_integration": {
                    "environment": "OpenAI API",
                    "llm_model": "gpt-4o",
                    "embedding_model": "text-embedding-3-large",
                    "vector_database": "FAISS + SQLite",
                    "search_type": "Hybrid (Semantic + Keyword)"
                },
                "processing_summary": {
                    "chunk_size": self.text_splitter._chunk_size,
                    "chunk_overlap": self.text_splitter._chunk_overlap,
                    "database_path": self.vector_db.db_path,
                    "index_path": self.vector_db.index_path
                }
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {"error": f"Failed to generate statistics: {e}"}

def main():
    """Main function for Meeting Document AI System with OpenAI"""
    
    print("🚀 Meeting Document AI System v3.0 (OpenAI Edition)")
    print("📊 Features: Vector Database, Hybrid Search, Chunking Support")
    print("🔑 Using OpenAI API (GPT-4o + text-embedding-3-large)")
    print("=" * 60)
    
    try:
        # Check OpenAI API key
        if not openai_api_key:
            print("❌ Error: OPENAI_API_KEY environment variable not set")
            print("💡 Please set your OpenAI API key:")
            print("   export OPENAI_API_KEY='your-api-key-here'")
            return
        
        print("🔐 OpenAI API key found")
        print(f"🏢 Project ID: {project_id}")
        print("✅ OpenAI authentication configured")
        
        processor = EnhancedMeetingDocumentProcessor(chunk_size=1000, chunk_overlap=200)
        
        # Check if documents folder exists
        docs_folder = Path("meeting_documents")
        if not docs_folder.exists():
            docs_folder.mkdir(exist_ok=True)
            print("📁 Created 'meeting_documents' folder. Please add your meeting documents to this folder.")
            return
        
        # Check for documents
        doc_files = []
        for ext in ['.txt', '.docx', '.pdf']:
            doc_files.extend(docs_folder.glob(f"*{ext}"))
        
        if not doc_files:
            print("📁 No meeting documents found in the 'meeting_documents' folder.")
            print("📋 Supported formats: .txt, .docx, .pdf")
            print("📂 Please add your meeting documents to the 'meeting_documents' folder and run again.")
            return
        
        print(f"📄 Found {len(doc_files)} documents to process")
        
        # Check if vector database exists
        print("🔍 Checking existing vector database...")
        if processor.vector_db.index.ntotal == 0:
            print("🔄 Processing documents and building vector database...")
            processor.process_documents("meeting_documents")
        else:
            print(f"✅ Loaded existing vector database with {processor.vector_db.index.ntotal} vectors")
        
        # Show comprehensive statistics
        print("\n📊 System Statistics:")
        print("-" * 50)
        stats = processor.get_meeting_statistics()
        if "error" not in stats:
            print(f"📋 Total meetings: {stats.get('total_meetings', 0)}")
            print(f"🧩 Total chunks: {stats.get('total_chunks', 0)}")
            print(f"🔢 Vector index size: {stats.get('vector_index_size', 0)}")
            print(f"📏 Average chunk length: {stats.get('average_chunk_length', 0)} characters")
            
            if 'date_range' in stats:
                print(f"📅 Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
            
            openai_info = stats.get('openai_integration', {})
            print(f"🤖 LLM Model: {openai_info.get('llm_model', 'Unknown')}")
            print(f"🔗 Embedding Model: {openai_info.get('embedding_model', 'Unknown')}")
            print(f"🔍 Search Type: {openai_info.get('search_type', 'Unknown')}")
            
            processing_info = stats.get('processing_summary', {})
            print(f"📦 Chunk size: {processing_info.get('chunk_size', 0)} chars")
            print(f"🔄 Chunk overlap: {processing_info.get('chunk_overlap', 0)} chars")
        
        # Interactive query loop
        print("\n🎯 Interactive Query Session")
        print("-" * 50)
        print("💡 Now supports hundreds of documents with hybrid search!")
        print("🔍 Combines semantic similarity + keyword matching for better results")
        
        example_queries = [
            "What are the main topics from recent meetings?",
            "Tell me about the AI integration progress across all meetings",
            "What are our upcoming deadlines and action items?",
            "Who are the key participants in recent discussions?",
            "What challenges or blockers have been identified?",
            "Summarize all migration plans discussed"
        ]
        
        for i, example in enumerate(example_queries[:3], 1):
            print(f"   {i}. {example}")
        print("   ... or ask anything about your meetings!")
        
        print(f"\n💬 Commands: 'quit'/'exit' to stop, 'stats' for statistics, 'help' for examples")
        
        while True:
            print("\n" + "-"*60)
            query = input("🤔 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the Meeting Document AI System!")
                break
            elif query.lower() == 'stats':
                stats = processor.get_meeting_statistics()
                print("\n📊 Detailed Statistics:")
                print(json.dumps(stats, indent=2, default=str))
                continue
            elif query.lower() == 'help':
                print("\n📚 Example Queries (enhanced with hybrid search):")
                for i, example in enumerate(example_queries, 1):
                    print(f"   {i}. {example}")
                print("\n💭 More ideas:")
                print("   • Search across hundreds of documents instantly")
                print("   • Find specific keywords + semantic meaning")
                print("   • Get comprehensive answers from multiple meetings")
                print("   • Timeline analysis across large document sets")
                continue
            elif query.lower() == 'refresh':
                try:
                    processor.refresh_clients()
                    print("✅ OpenAI clients refreshed successfully")
                except Exception as e:
                    print(f"❌ Failed to refresh clients: {e}")
                continue
            elif not query:
                print("❓ Please enter a valid question.")
                continue
            
            print(f"\n🔍 Processing with hybrid search: '{query}'...")
            start_time = datetime.now()
            
            try:
                answer = processor.answer_query(query)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                print(f"\n💬 Answer (processed in {processing_time:.2f}s):")
                print("-" * 60)
                print(answer)
                
            except Exception as query_error:
                print(f"❌ Error processing query: {query_error}")
                print("🔄 You can try rephrasing your question or check your OpenAI API key.")
    
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        print(f"❌ Critical Error: {e}")
        print("🔧 Please check your OpenAI API key and configuration.")
        print("💡 Make sure you have set OPENAI_API_KEY environment variable")

if __name__ == "__main__":
    main()