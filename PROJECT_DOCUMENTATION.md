# UHG Meeting Document AI System

## Overview

A sophisticated AI-powered web application designed to process, analyze, and provide intelligent insights from meeting documents. Built with Flask backend and modern JavaScript frontend, the system leverages OpenAI's GPT-4o model and advanced vector search capabilities for comprehensive meeting document analysis.

## Project Structure

```
AI assistant old/
├── flask_app.py              # Main Flask application server
├── meeting_processor.py      # Core document processing and AI logic
├── requirements.txt          # Python dependencies
├── templates/
│   └── chat.html             # Main web interface template
├── static/
│   ├── script.js             # Frontend JavaScript logic
│   ├── styles.css            # Application styling
│   └── icons/
│       ├── Optum-logo.png    # UHG/Optum logo (main branding)
│       ├── favicon.png       # Application favicon
│       └── paperclip.svg     # File attachment icon
├── meeting_documents/        # Storage for uploaded documents
├── uploads/                  # Temporary file upload directory
├── temp/                     # Temporary processing directory
├── logs/
│   └── flask_app.log         # Application logs
├── backups/                  # Backup storage
├── tiktoken_cache/           # Token caching for OpenAI
├── meeting_documents.db      # SQLite database for metadata
└── vector_index.faiss        # FAISS vector database index
```

## Core Components

### 1. Flask Application (`flask_app.py`)

**Main Features:**
- RESTful API endpoints for file upload, chat, and system management
- Comprehensive error handling and logging
- File validation and security measures
- Integration with the meeting processor

**Key Endpoints:**
- `GET /` - Main chat interface
- `POST /api/upload` - File upload and processing
- `POST /api/chat` - Chat message processing
- `GET /api/stats` - System statistics
- `POST /api/refresh` - System refresh
- `GET /api/test` - System health check

**Security Features:**
- File type validation (.docx, .txt, .pdf)
- File size limits (50MB max)
- Secure filename handling
- Input sanitization

### 2. Meeting Processor (`meeting_processor.py`)

**Core Architecture:**
- **OpenAI Integration**: Uses GPT-4o for text analysis and text-embedding-3-large for vector embeddings
- **Vector Database**: FAISS + SQLite hybrid storage for efficient similarity search
- **Document Processing**: Supports DOCX, PDF, and TXT files
- **Hybrid Search**: Combines semantic similarity and keyword matching

**Key Classes:**

#### `MeetingDocument`
```python
@dataclass
class MeetingDocument:
    document_id: str
    filename: str
    date: datetime
    title: str
    content: str
    content_summary: str
    main_topics: List[str]
    past_events: List[str]
    future_actions: List[str]
    participants: List[str]
    chunk_count: int
    file_size: int
```

#### `DocumentChunk`
```python
@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    embedding: Optional[np.ndarray]
```

#### `VectorDatabase`
- FAISS index for high-performance vector similarity search
- SQLite for metadata storage and keyword search
- Automatic vector normalization for cosine similarity
- Batch processing for efficient insertions

#### `EnhancedMeetingDocumentProcessor`
- Document parsing with AI-powered content extraction
- Intelligent chunking with configurable overlap
- Hybrid search combining semantic and keyword approaches
- Comprehensive error handling and retry mechanisms

### 3. Frontend Interface

#### HTML Template (`chat.html`)
- Modern chat interface with ChatGPT-like design
- Responsive sidebar with conversation history
- Modal dialogs for file upload and system statistics
- Mobile-optimized responsive design

#### JavaScript (`script.js`)
**Key Features:**
- Real-time conversation management with local storage persistence
- Advanced Markdown rendering with security sanitization
- File upload with drag-and-drop support
- Mobile-responsive sidebar with touch optimization
- Auto-save functionality for conversation persistence

**Notable Functions:**
- `formatMarkdownToHTML()`: Enhanced Markdown parsing with DOMPurify sanitization
- `hybrid_search()`: Client-side search result handling
- `persistAllData()`: Conversation state management
- `initializeMobileFixes()`: Mobile optimization

#### CSS Styling (`styles.css`)
- Modern design system with UHG brand colors
- Responsive grid layouts
- Advanced Markdown styling for AI responses
- Mobile-first responsive design
- Smooth animations and transitions

## Technical Architecture

### AI & Machine Learning Stack
- **LLM Model**: OpenAI GPT-4o (4000 max tokens)
- **Embedding Model**: text-embedding-3-large (3072 dimensions)
- **Vector Search**: FAISS with Inner Product similarity
- **Text Processing**: LangChain with RecursiveCharacterTextSplitter

### Database Architecture
- **Vector Storage**: FAISS index for embeddings
- **Metadata Storage**: SQLite with optimized schemas
- **Caching**: TikToken cache for tokenization
- **Persistence**: Local file storage with backup support

### Search Strategy
1. **Semantic Search**: Vector similarity using OpenAI embeddings
2. **Keyword Search**: SQL-based text matching
3. **Hybrid Scoring**: Weighted combination (70% semantic, 30% keyword)
4. **Context Selection**: Smart chunk selection across documents

## Configuration & Environment

### Required Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Python Dependencies
```
Flask                 # Web framework
openai               # OpenAI API client
langchain            # LLM framework
langchain_openai     # OpenAI integration
faiss-cpu            # Vector similarity search
numpy                # Numerical computing
scikit-learn         # Machine learning utilities
python-docx          # DOCX file processing
PyPDF2               # PDF file processing
pandas               # Data manipulation
httpx                # HTTP client
tiktoken             # Tokenization
python-dotenv        # Environment variables
pydantic             # Data validation
```

## Key Features

### Document Processing
1. **Multi-format Support**: DOCX, PDF, TXT files
2. **Intelligent Parsing**: AI-powered content extraction
3. **Metadata Extraction**: Topics, participants, action items, dates
4. **Chunking Strategy**: Configurable size (1000 chars) with overlap (200 chars)
5. **Vector Embeddings**: High-dimensional semantic representations

### Search & Retrieval
1. **Hybrid Search**: Semantic + keyword matching
2. **Timeframe Filtering**: Date-based document filtering
3. **Context Ranking**: Intelligent chunk selection and scoring
4. **Multi-document Synthesis**: Cross-document information aggregation

### Web Interface
1. **Chat Interface**: Real-time conversational AI
2. **File Management**: Drag-and-drop upload with progress tracking
3. **Conversation History**: Persistent conversation management
4. **Responsive Sidebar**: Collapsible sidebar with smooth animations and content adjustment
5. **Mobile Optimization**: Touch-friendly responsive design with overlay sidebar
6. **Statistics Dashboard**: System insights and metrics

### Security & Performance
1. **Input Validation**: Comprehensive file and input checking
2. **XSS Protection**: DOMPurify sanitization for Markdown
3. **Error Handling**: Graceful degradation and recovery
4. **Caching Strategy**: Stats caching and token optimization
5. **Logging**: Comprehensive application logging

## API Endpoints

### File Upload (`POST /api/upload`)
**Request**: FormData with files
**Response**:
```json
{
    "success": true,
    "results": [
        {
            "filename": "meeting.docx",
            "success": true,
            "chunks": 15,
            "error": null
        }
    ],
    "processed": 1,
    "total": 1,
    "message": "Successfully processed 1 of 1 files"
}
```

### Chat (`POST /api/chat`)
**Request**:
```json
{
    "message": "What were the main topics in recent meetings?"
}
```
**Response**:
```json
{
    "success": true,
    "response": "Based on the meeting documents...",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Statistics (`GET /api/stats`)
**Response**:
```json
{
    "success": true,
    "stats": {
        "total_meetings": 25,
        "total_chunks": 450,
        "vector_index_size": 450,
        "average_chunk_length": 850,
        "date_range": {
            "earliest": "2024-01-01",
            "latest": "2024-01-15"
        },
        "openai_integration": {
            "llm_model": "gpt-4o",
            "embedding_model": "text-embedding-3-large"
        }
    }
}
```

## Usage Examples

### Sample Queries
1. **Topic Analysis**: "What are the main topics from recent meetings?"
2. **Action Items**: "List all action items from last week's meetings"
3. **Participant Analysis**: "Who are the key participants and their roles?"
4. **Decision Tracking**: "Summarize decisions made in project meetings"
5. **Timeline Analysis**: "Show me upcoming deadlines and milestones"
6. **Problem Identification**: "What challenges or blockers were identified?"

### Document Processing Workflow
1. Upload documents via web interface or API
2. Automatic content extraction and AI analysis
3. Chunk generation with embeddings
4. Storage in vector database with metadata
5. Ready for intelligent querying

## Performance Characteristics

- **Vector Search**: Sub-second similarity search on thousands of chunks
- **Concurrent Processing**: Multi-threaded file processing
- **Memory Efficiency**: Streaming processing for large documents
- **Cache Optimization**: 5-minute stats cache with auto-refresh
- **Responsive Design**: 60fps animations and smooth scrolling

## Deployment Considerations

### System Requirements
- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- OpenAI API access
- Modern web browser support

### Production Setup
1. Configure environment variables
2. Set up reverse proxy (nginx recommended)
3. Configure SSL/TLS certificates
4. Set up log rotation
5. Configure backup strategies for database files

## Error Handling

### File Processing Errors
- Invalid file formats → User-friendly error messages
- Large files → Size limit enforcement with clear feedback
- Corrupted files → Graceful error handling with logging
- API failures → Retry mechanisms with exponential backoff

### System Resilience
- Database connection failures → Automatic reconnection
- OpenAI API failures → Circuit breaker pattern
- Memory constraints → Streaming processing
- Network issues → Timeout handling and retry logic

## Future Enhancement Opportunities

### Technical Improvements
1. **Advanced RAG**: Implement graph-based RAG for better context
2. **Multi-modal Support**: Add image and audio processing
3. **Real-time Collaboration**: WebSocket-based live updates
4. **Advanced Analytics**: Meeting sentiment and engagement analysis
5. **Custom Models**: Fine-tuned models for specific domains

### Feature Extensions
1. **Meeting Scheduling**: Integration with calendar systems
2. **Automated Summaries**: Regular meeting digest generation
3. **Export Capabilities**: PDF/Word report generation
4. **Advanced Search**: Faceted search with filters
5. **Notification System**: Alert system for action items

---

*This documentation represents the current state of the UHG Meeting Document AI System. Update this file when making significant changes to maintain accuracy and usefulness for future development and maintenance.*