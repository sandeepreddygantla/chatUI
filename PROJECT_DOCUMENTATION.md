# UHG Meeting Document AI System

## Overview

A sophisticated **multi-user** AI-powered web application designed to process, analyze, and provide intelligent insights from meeting documents. Built with Flask backend and modern JavaScript frontend, the system leverages OpenAI's GPT-4o model and advanced vector search capabilities for comprehensive meeting document analysis. **Version 2.0** introduces complete user authentication, project-based organization, and user-isolated data processing.

## Project Structure

```
AI assistant old/
├── flask_app.py              # Mi want another change in the UI. where uhg user we see i want the current user details over there instead of showing on top of chat body.ain Flask application server with authentication
├── meeting_processor.py      # Core document processing and AI logic (multi-user)
├── requirements.txt          # Python dependencies (includes Flask-Login, bcrypt)
├── templates/
│   ├── chat.html             # Main web interface template
│   ├── login.html            # User login page
│   └── register.html         # User registration page
├── static/
│   ├── script.js             # Frontend JavaScript logic (with auth)
│   ├── styles.css            # Application styling
│   └── icons/
│       ├── Optum-logo.png    # UHG/Optum logo (main branding)
│       ├── favicon.png       # Application favicon
│       └── paperclip.svg     # File attachment icon
├── meeting_documents/        # User-specific document storage
│   ├── user_john/            # User-specific folders
│   ├── user_sarah/           # Organized by username
│   └── user_admin/           # Complete data isolation
├── uploads/                  # Temporary file upload directory
├── temp/                     # Temporary processing directory
├── logs/
│   └── flask_app.log         # Application logs
├── backups/                  # Backup storage
├── tiktoken_cache/           # Token caching for OpenAI
├── meeting_documents.db      # SQLite database (now with users, projects, meetings)
└── vector_index.faiss        # FAISS vector database index
```

## Core Components

### 1. Flask Application (`flask_app.py`)

**Main Features:**
- **Multi-user authentication system** with Flask-Login
- RESTful API endpoints for file upload, chat, and system management
- **User session management** with secure password hashing (bcrypt)
- Comprehensive error handling and logging
- File validation and security measures
- Integration with the meeting processor
- **User-scoped data access** - complete data isolation between users

**Authentication Endpoints:**
- `GET /login` - User login page
- `POST /login` - Process login credentials
- `GET /register` - User registration page  
- `POST /register` - Create new user account
- `POST /logout` - User logout
- `GET /api/auth/status` - Check authentication status

**Main Application Endpoints (All Require Authentication):**
- `GET /` - Main chat interface (redirects to login if not authenticated)
- `POST /api/upload` - File upload and processing (user-scoped)
- `POST /api/chat` - Chat message processing (user-scoped document filtering)
- `GET /api/documents` - Retrieve user's documents only
- `GET /api/stats` - System statistics for current user
- `POST /api/refresh` - System refresh
- `GET /api/test` - System health check

**Security Features:**
- **Password-based authentication** with bcrypt hashing
- **Session management** with 24-hour timeout
- **User data isolation** - users can only access their own data
- File type validation (.docx, .txt, .pdf)
- File size limits (50MB max)
- Secure filename handling
- Input sanitization
- **CSRF protection** and secure session cookies

### 2. Meeting Processor (`meeting_processor.py`)

**Core Architecture:**
- **OpenAI Integration**: Uses GPT-4o for text analysis and text-embedding-3-large for vector embeddings
- **Multi-User Vector Database**: FAISS + SQLite hybrid storage with user isolation
- **Document Processing**: Supports DOCX, PDF, and TXT files with user context
- **Hybrid Search**: Combines semantic similarity and keyword matching with user-scoped filtering
- **Project Organization**: Documents organized by users and projects

**Key Classes:**

#### `User`
```python
@dataclass
class User:
    user_id: str
    username: str
    email: str
    full_name: str
    password_hash: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    role: str = 'user'
```

#### `Project`
```python
@dataclass
class Project:
    project_id: str
    user_id: str
    project_name: str
    description: str
    created_at: datetime
    is_active: bool = True
```

#### `Meeting`
```python
@dataclass
class Meeting:
    meeting_id: str
    user_id: str
    project_id: str
    meeting_name: str
    meeting_date: datetime
    created_at: datetime
```

#### `MeetingDocument` (Updated)
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
    chunk_count: int = 0
    file_size: int = 0
    user_id: Optional[str] = None      # NEW: User who owns this document
    meeting_id: Optional[str] = None   # NEW: Associated meeting
    project_id: Optional[str] = None   # NEW: Associated project
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

#### `VectorDatabase` (Multi-User Enhanced)
- **FAISS index** for high-performance vector similarity search
- **SQLite for metadata storage** with complete multi-user schema
- **User isolation** - all queries automatically filtered by user context
- **Project/Meeting organization** - hierarchical document structure
- **Automatic migration** from single-user to multi-user database
- Automatic vector normalization for cosine similarity
- Batch processing for efficient insertions

**Database Schema:**
```sql
-- User Management
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL, 
    full_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    role TEXT DEFAULT 'user'
);

-- Project Organization  
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

-- Meeting Management
CREATE TABLE meetings (
    meeting_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    meeting_name TEXT NOT NULL,
    meeting_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (project_id) REFERENCES projects (project_id)
);

-- Document Storage (Updated)
CREATE TABLE documents (
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
    user_id TEXT,           -- NEW: User ownership
    meeting_id TEXT,        -- NEW: Meeting association
    project_id TEXT,        -- NEW: Project association
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id),
    FOREIGN KEY (project_id) REFERENCES projects (project_id)
);

-- Chunk Storage (Updated)
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    user_id TEXT,           -- NEW: User context
    meeting_id TEXT,        -- NEW: Meeting context
    project_id TEXT,        -- NEW: Project context
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (document_id),
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id),
    FOREIGN KEY (project_id) REFERENCES projects (project_id)
);

-- Session Management
CREATE TABLE user_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);
```

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
- `showConversationMenu()`: Dynamic dropdown positioning for conversation actions
- `confirmEdit()`: Conversation title editing with validation and persistence
- `confirmDelete()`: Safe conversation deletion with custom modal confirmation
- `closeConversationMenu()`: Click-outside-to-close behavior for dropdown menus

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

### User Management & Authentication
1. **Multi-User Support**: Complete user isolation with individual accounts
2. **Password Authentication**: Secure bcrypt hashing with session management
3. **User Registration**: Self-service account creation with validation
4. **Session Management**: 24-hour sessions with automatic logout
5. **Data Isolation**: Users can only access their own documents and conversations
6. **Future SSO Ready**: Architecture prepared for enterprise SSO integration

### Document Processing (User-Scoped)
1. **Multi-format Support**: DOCX, PDF, TXT files with user context
2. **Intelligent Parsing**: AI-powered content extraction
3. **Metadata Extraction**: Topics, participants, action items, dates
4. **Chunking Strategy**: Configurable size (1000 chars) with overlap (200 chars)
5. **Vector Embeddings**: High-dimensional semantic representations
6. **User-Specific Storage**: Documents stored in user folders (`meeting_documents/user_john/`)
7. **Project Organization**: Automatic assignment to user's default project
8. **Meeting Association**: Each document linked to a specific meeting

### Document Selection & Querying (Enhanced)
1. **@ Mention System**: Type "@" to select specific documents for targeted queries
2. **Automatic User Scope**: Queries without @ mentions automatically include ALL user documents
3. **Fuzzy Search**: Real-time document filtering by filename (user's documents only)
4. **Multi-Document Selection**: Select multiple documents using visual pills interface
5. **Smart Dropdown**: Automatically positions above/below input based on available space
6. **Keyboard Navigation**: Arrow keys, Enter, and Escape support for accessibility

### Query Processing Behavior
**Current Implementation:**
- **With @ mentions**: `@filename.docx what was discussed?` → Uses only specified documents
- **Without @ mentions**: `what were the decisions?` → Uses ALL documents for current user
- **User Isolation**: Users never see or access other users' documents
- **Project Context**: Documents organized by projects (default project auto-created)
- **Meeting Context**: Each document belongs to a specific meeting

### Search & Retrieval
1. **Hybrid Search**: Semantic + keyword matching
2. **Timeframe Filtering**: Date-based document filtering
3. **Context Ranking**: Intelligent chunk selection and scoring
4. **Multi-document Synthesis**: Cross-document information aggregation

### Web Interface
1. **Chat Interface**: Real-time conversational AI
2. **File Management**: Drag-and-drop upload with progress tracking
3. **Conversation History**: Persistent conversation management with advanced individual management
   - Three dots menu (⋯) for each conversation with rename and delete options
   - Custom confirmation dialogs (non-browser default) for delete operations
   - Inline rename functionality with Enter/Escape key support
   - Smart positioning dropdown menus with click-outside-to-close behavior
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
**Request (Standard)**:
```json
{
    "message": "What were the main topics in recent meetings?"
}
```

**Request (Document-Filtered)**:
```json
{
    "message": "Give me summary of the meeting",
    "document_ids": ["doc_123", "doc_456"]
}
```

**Response**:
```json
{
    "success": true,
    "response": "Based on the meeting documents...",
    "follow_up_questions": [
        "What were the key decisions made?",
        "Who were the main participants?",
        "What are the next steps?"
    ],
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Documents (`GET /api/documents`)
**Response**:
```json
{
    "success": true,
    "documents": [
        {
            "document_id": "doc_123",
            "filename": "Print Migration Meeting.docx", 
            "date": "2024-01-15T10:00:00Z",
            "file_size": 1024000
        }
    ],
    "count": 1
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

#### General Queries (All Documents)
1. **Topic Analysis**: "What are the main topics from recent meetings?"
2. **Action Items**: "List all action items from last week's meetings"
3. **Participant Analysis**: "Who are the key participants and their roles?"
4. **Decision Tracking**: "Summarize decisions made in project meetings"
5. **Timeline Analysis**: "Show me upcoming deadlines and milestones"

#### Document-Specific Queries (@ Mention System)
1. **Single Document**: "@Print Migration.docx give me summary of the meeting"
2. **Multiple Documents**: "@Print Migration.docx @Team Standup.docx what are common themes?"
3. **Filtered Analysis**: "@Budget Review.docx what decisions were made about spending?"
4. **Comparison**: "@Q1 Planning.docx @Q2 Planning.docx compare the strategic priorities"
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

## Technical Implementation Details

### @ Mention System Architecture

**Frontend Components (script.js):**
- `detectAtMention(input)`: Parses cursor position to detect @ symbols followed by search text
- `showDocumentDropdown(searchText)`: Displays filtered document list with smart positioning
- `filterDocuments(searchText)`: Fuzzy search filtering by filename
- `selectDocument(doc)`: Adds documents to selection with visual pills
- `setupAtMentionDetection()`: Event listeners for input detection and keyboard navigation

**CSS Styling (styles.css):**
- `.document-dropdown`: Positioned dropdown with smart above/below positioning
- `.document-item`: Individual document entries with hover states
- `.document-pill`: Selected document visual indicators
- Responsive design for mobile and desktop

**Backend Integration:**
- `/api/documents` endpoint returns document list with metadata
- `/api/chat` accepts `document_ids` parameter for filtered queries
- `answer_query()` method enhanced to filter search by specific documents

**Key Features:**
- Real-time dropdown positioning based on viewport space
- Keyboard accessibility (arrow keys, Enter, Escape)
- Multi-document selection with visual feedback
- Fuzzy search with filename matching
- Event handling with proper cleanup and error handling

## Version History

### Version 2.0 (July 2025) - Multi-User System
**Major Features Added:**
- **Complete user authentication system** with registration, login, logout
- **Multi-user architecture** with complete data isolation
- **Project-based organization** with automatic project creation
- **Meeting management** with document-meeting associations
- **User-scoped querying** - automatic filtering by user context
- **Enhanced database schema** with users, projects, meetings tables
- **User-specific file storage** in isolated folders
- **Session management** with secure password hashing
- **Automatic database migration** from single-user to multi-user
- **Professional authentication UI** with modern design

**Query Behavior Changes:**
- Queries without @ mentions now automatically include ALL user documents (not just summary keywords)
- Complete data isolation - users never see other users' data
- Enhanced @ mention system with user context

**Breaking Changes:**
- All API endpoints now require authentication
- Database schema updated (automatic migration included)
- File storage structure changed to user-specific folders

### Version 1.0 (July 2025) - Single User System  
**Original Features:**
- Single-user meeting document AI system
- @ mention document selection system for targeted querying
- OpenAI GPT-4o integration for analysis
- FAISS vector database for semantic search
- Support for DOCX, PDF, TXT files
- Hybrid search combining semantic and keyword matching
- Real-time conversation management
- Markdown response formatting

---

**Architecture Notes:**
- Built for enterprise deployment with SSO extensibility
- User data completely isolated at database and file system level
- Ready for horizontal scaling with user-specific data partitioning
- Designed for 100+ concurrent users with individual project management

*Last updated: July 2025 - Added complete multi-user authentication and project organization system*

*This documentation represents the current state of the UHG Meeting Document AI System v2.0. Update this file when making significant changes to maintain accuracy and usefulness for future development and maintenance.*