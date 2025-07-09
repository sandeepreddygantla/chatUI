# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set required environment variable
export OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Application
```bash
# Start the Flask development server
python flask_app.py

# Access the application
# Main interface: http://localhost:5000
# Health check: http://localhost:5000/api/test
```

### Database Operations
```bash
# The SQLite database (meeting_documents.db) is automatically created
# No manual migration commands needed - automatic schema migration on startup

# Manual backup
cp meeting_documents.db backups/meeting_documents_backup_$(date +%Y%m%d_%H%M%S).db
```

### Testing and Validation
```bash
# Test syntax and imports
python3 -m py_compile meeting_processor.py
python3 -m py_compile flask_app.py

# Test comprehensive summary logic
python3 test_comprehensive_summary.py
python3 test_flexible_summary.py

# Test OpenAI API connection
curl -X GET http://localhost:5000/api/test

# Test user authentication flow
curl -X POST http://localhost:5000/api/auth/status

# Test file upload (requires authentication)
curl -X POST -F "files=@test.docx" http://localhost:5000/api/upload
```

## Architecture Overview

### Core System Design
This is a **multi-user AI meeting document analysis system** built with Flask backend and modern JavaScript frontend. The system uses OpenAI's GPT-4o for intelligent document analysis and FAISS for vector similarity search.

**Key Architectural Principles:**
- **User Isolation**: Complete data separation between users at database and filesystem level
- **Project-Based Organization**: Documents organized in user-specific projects and meetings
- **Hybrid Search**: Combines semantic vector search with keyword matching
- **Real-time Chat Interface**: Conversational AI with persistent conversation history

### Multi-User Architecture
```
User Authentication → Project Selection → Document Upload → AI Analysis → Chat Interface
     (bcrypt)           (auto-created)      (user folder)    (GPT-4o)     (conversation)
```

**Data Isolation Pattern:**
- Database: All queries filtered by `user_id`
- Filesystem: Documents stored in `meeting_documents/user_{username}/`
- API: All endpoints require authentication and automatically scope to current user
- Vector Search: Embeddings tagged with user context

### Core Components

#### 1. Flask Application (`flask_app.py`)
- **Authentication System**: Flask-Login with bcrypt password hashing
- **File Upload Handler**: Processes DOCX, PDF, TXT files with user context
- **Chat API**: Handles conversational queries with document filtering
- **User Management**: Registration, login, session management

#### 2. Meeting Processor (`meeting_processor.py`)
- **VectorDatabase Class**: FAISS + SQLite hybrid with user isolation
- **EnhancedMeetingDocumentProcessor**: AI-powered document analysis with flexible comprehensive summaries
- **Intelligent Date Filtering**: Calendar-aware timeframe detection (current week, last month, etc.)
- **Hybrid Search Engine**: Semantic + keyword search with user scoping
- **Flexible Project Summary System**: User-centric responses that process ALL files without rigid templates

#### 3. Frontend Interface (`static/`)
- **Advanced Document Selection**: `@filename` mentions for targeted queries  
- **Folder Navigation**: `#folder` system for browsing project documents
- **Real-time Chat**: Conversation management with markdown rendering
- **User Profile System**: Enhanced user information display

### Document Processing Pipeline
```
Upload → Content Extraction → AI Analysis → Chunking → Vector Embedding → Database Storage
  ↓           (python-docx)      (GPT-4o)    (1000 chars)  (text-embedding-3-large)
User Folder → Meeting Creation → Project Assignment → FAISS Index → SQLite Metadata
```

### Database Schema Architecture
The system uses SQLite with the following key tables:
- **users**: Authentication and user profiles
- **projects**: User-specific project organization  
- **meetings**: Meeting-document associations
- **documents**: Document metadata with user/project context
- **chunks**: Text chunks with embeddings for vector search

**Critical User Isolation**: All tables include `user_id` foreign keys and all queries are automatically filtered by the authenticated user's context.

### Search and Query System

#### Document Selection Patterns
- **No Mention**: `"What were the decisions?"` → Uses ALL user documents
- **File Mention**: `"@meeting.docx What was discussed?"` → Uses only specified file(s)
- **Folder Mention**: `"#Project Alpha What's the status?"` → Uses all files in folder
- **Date Filtering**: `"Last week's summary"` → Automatically filters by calendar week
- **Project Summary**: `"Give me a project summary"` → Processes ALL files with user-centric approach

#### Comprehensive Project Summary System
The system detects comprehensive project queries and processes ALL files (no 5-file limitation):
- **Small Projects (≤15 files)**: Individual file analysis with maximum detail
- **Medium Projects (16-50 files)**: Smart sampling with relevance scoring
- **Large Projects (50+ files)**: Hierarchical processing with time-period grouping
- **User-Centric Responses**: Answers what users actually ask instead of forcing rigid templates

#### Intelligent Date Detection
The system automatically detects and processes temporal queries:
- `"current week"` → Monday-Sunday of current week
- `"last month"` → Previous calendar month boundaries
- `"last 3 months"` → Rolling 90-day window
- `"this quarter"` → Current fiscal quarter

### Frontend State Management

#### @ Mention System (`script.js`)
```javascript
// Key functions for document selection
detectAtMention(input)           // Detects @ symbols and text
showDocumentDropdown(searchText) // Displays filtered document list  
selectDocument(doc)              // Adds document to selection
parseMessageForDocuments(message) // Extracts document filters for API
```

#### Conversation Management
- **Local Storage**: Conversations persisted in browser
- **Real-time Updates**: Auto-save conversation state
- **Mobile Responsive**: Touch-optimized sidebar and responsive design

## Development Guidelines

### Working with User Authentication
All API endpoints (except `/login`, `/register`, `/api/auth/status`) require authentication. Use the `@login_required` decorator and access `current_user` for user context.

```python
@app.route('/api/example')
@login_required
def example():
    user_id = current_user.user_id  # Always use this for data scoping
    # Ensure all database queries include user_id filtering
```

### Adding New Document Processing Features
When extending document processing:
1. **Always include user context** in `meeting_processor.py` methods
2. **Use existing chunking strategy** (1000 chars, 200 overlap)
3. **Add vector embeddings** for new content types
4. **Update database schema** with proper foreign key relationships

### Working with Comprehensive Project Summaries
The system uses a simplified, flexible approach for project summaries:

#### Key Functions
- `detect_project_summary_query()`: Detects queries requesting comprehensive analysis
- `_generate_flexible_comprehensive_answer()`: Single function that adapts to user queries
- `_get_detailed_content_from_all_files()`: For small projects (≤15 files)
- `_get_smart_sampled_content()`: For medium projects (16-50 files)  
- `_get_summarized_content_with_excerpts()`: For large projects (50+ files)

#### Design Principles
- **User-Centric**: Responds to what users actually ask, not predefined templates
- **All Files Processed**: Every file in the project contributes to the analysis
- **Scalable**: Automatically adapts processing strategy based on project size
- **Natural Language**: AI responds naturally instead of forcing artificial structure

#### Adding New Query Types
When extending the comprehensive summary system:
1. **Extend keyword detection** in `detect_project_summary_query()`
2. **Keep responses natural** - avoid rigid templates
3. **Maintain file count transparency** - show how many files were processed
4. **Test with various project sizes** to ensure scalability

### Frontend Development Patterns
- **Document Selection**: Extend the `@` mention system in `script.js`
- **API Integration**: Use existing authentication headers and error handling
- **Mobile Responsive**: Follow existing responsive patterns in `styles.css`
- **Conversation UI**: Integrate with existing conversation management

### Date-Based Query Enhancement
The system includes intelligent date filtering. When adding temporal features:
- **Use `_detect_timeframe_from_query()`** for natural language date parsing
- **Extend `_calculate_date_range()`** for new date patterns
- **Follow calendar-aware logic** (Monday-Sunday weeks, proper month boundaries)
- **Add corresponding frontend date options** in `getDateOptions()`

### Vector Search Optimization
- **Embeddings**: Use OpenAI text-embedding-3-large (3072 dimensions)
- **FAISS Index**: Inner product similarity with normalized vectors
- **User Scoping**: Always filter by user_id before vector search
- **Hybrid Strategy**: 70% semantic, 30% keyword search weighting

## Critical Security Considerations

### User Data Isolation
**NEVER** bypass user isolation. All database queries must include user context:
```python
# CORRECT - Always filter by user
documents = self.get_user_documents(user_id)

# INCORRECT - Never query all documents
documents = self.get_all_documents()  # Security violation
```

### File Handling Security
- **Filename Sanitization**: Use `secure_filename()` for all uploads
- **File Type Validation**: Only allow DOCX, PDF, TXT files
- **Size Limits**: Enforce 50MB maximum file size
- **Path Traversal Prevention**: Store files in user-specific directories

### Authentication Security
- **Password Hashing**: Use bcrypt with proper salt
- **Session Management**: 24-hour session timeout
- **Input Validation**: Sanitize all user inputs
- **API Authentication**: Require login for all data endpoints

## Environment Variables

### Required Configuration
```bash
OPENAI_API_KEY=sk-...          # OpenAI API key for GPT-4o and embeddings
SECRET_KEY=...                 # Flask session encryption (auto-generated if not set)
```

### Optional Configuration  
```bash
FLASK_ENV=development          # Development mode
TIKTOKEN_CACHE_DIR=./tiktoken_cache  # Token cache directory (default)
```

## Project-Specific Context

### UHG/Optum Branding
- **Logo**: `static/icons/Optum-logo.png` used in header
- **Color Scheme**: UHG brand colors defined in `styles.css`
- **Terminology**: "Meeting Document AI" as the primary product name

### Document Organization Philosophy
- **Projects**: Top-level organization (auto-created for users)
- **Meetings**: Documents belong to specific meetings within projects
- **Folders**: Frontend abstraction over project structure
- **User Isolation**: Complete separation at all levels

### AI Model Configuration
- **LLM**: GPT-4o with 4000 max tokens for analysis
- **Embeddings**: text-embedding-3-large with 3072 dimensions  
- **Search Strategy**: Hybrid approach combining semantic and keyword search
- **Context Limit**: 10 document chunks for normal queries, unlimited for comprehensive summaries
- **Project Summary Strategy**: Flexible processing that adapts from 15 files to 100+ files

### Recent Architecture Improvements
The system recently underwent major simplification of the comprehensive project summary feature:

#### What Changed
- **Removed**: 6 rigid processing functions with predefined templates
- **Added**: 1 flexible function that adapts to user queries
- **Improved**: User experience now responds naturally to specific questions
- **Maintained**: All files processing capability and scalability

#### Impact
- **User queries like "What challenges?" now get focused answers about challenges only**
- **No more forced 7-section responses that ignore user intent**
- **90% reduction in code complexity while maintaining all core benefits**
- **Natural conversation flow instead of rigid formats**

This system is designed for enterprise deployment with SSO extensibility and horizontal scaling capabilities for 100+ concurrent users.