from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import json
import logging
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import shutil
import bcrypt
import secrets

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/flask_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import your existing classes
try:
    from meeting_processor import EnhancedMeetingDocumentProcessor
    logger.info("Successfully imported meeting_processor")
except ImportError as e:
    logger.error(f"Failed to import meeting_processor: {e}")
    logger.error("Make sure meeting_processor.py is in the same directory")
    exit(1)

# Create Flask app with proper static folder configuration
app = Flask(__name__, 
           static_folder='static',  # Explicitly set static folder
           static_url_path='/static',  # Set static URL path
           template_folder='templates')  # Set template folder

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Enhanced session configuration for persistence
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)  # 30 day session
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent XSS attacks
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)  # 30 day remember me
app.config['REMEMBER_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['REMEMBER_COOKIE_HTTPONLY'] = True

# Configure Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Global processor instance
processor = None

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username, email, full_name):
        self.id = user_id
        self.user_id = user_id
        self.username = username
        self.email = email
        self.full_name = full_name
    
    def get_id(self):
        return self.user_id

@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login"""
    if processor:
        user = processor.vector_db.get_user_by_id(user_id)
        if user:
            return User(user.user_id, user.username, user.email, user.full_name)
    return None

def initialize_processor():
    """Initialize the document processor"""
    global processor
    try:
        logger.info("Initializing Enhanced Meeting Document Processor...")
        processor = EnhancedMeetingDocumentProcessor(chunk_size=1000, chunk_overlap=200)
        
        # Clean up expired sessions on startup
        if processor and processor.vector_db:
            cleaned_count = processor.vector_db.cleanup_expired_sessions()
            logger.info(f"Cleaned up {cleaned_count} expired sessions on startup")
        
        logger.info("Processor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        return False

# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'GET':
        return render_template('register.html')
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        full_name = data.get('full_name', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        
        # Validation
        if not all([username, email, full_name, password]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        if password != confirm_password:
            return jsonify({'success': False, 'error': 'Passwords do not match'}), 400
        
        if len(password) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400
        
        if not processor:
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Create user
        user_id = processor.vector_db.create_user(username, email, full_name, password_hash)
        
        # Create default project
        project_id = processor.vector_db.create_project(user_id, "Default Project", "Default project for meetings")
        
        logger.info(f"New user registered: {username} ({user_id})")
        return jsonify({
            'success': True, 
            'message': 'Registration successful! Please log in.',
            'user_id': user_id
        })
        
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'success': False, 'error': 'Registration failed'}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'GET':
        return render_template('login.html')
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password are required'}), 400
        
        if not processor:
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        # Get user
        user = processor.vector_db.get_user_by_username(username)
        if not user:
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
        
        # Check password
        if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
        
        # Login user with permanent session
        flask_user = User(user.user_id, user.username, user.email, user.full_name)
        login_user(flask_user, remember=True)
        session.permanent = True  # Make session permanent
        
        # Update last login
        processor.vector_db.update_user_last_login(user.user_id)
        
        logger.info(f"User logged in: {username}")
        return jsonify({
            'success': True, 
            'message': 'Login successful',
            'user': {
                'user_id': user.user_id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'error': 'Login failed'}), 500

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    """User logout"""
    username = current_user.username
    logout_user()
    logger.info(f"User logged out: {username}")
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/auth/status')
def auth_status():
    """Check authentication status and validate session"""
    try:
        if current_user.is_authenticated:
            # Validate that the user still exists in the database
            if processor:
                user = processor.vector_db.get_user_by_id(current_user.user_id)
                if user and user.is_active:
                    # Extend session on successful validation
                    session.permanent = True
                    
                    return jsonify({
                        'authenticated': True,
                        'user': {
                            'user_id': current_user.user_id,
                            'username': current_user.username,
                            'email': current_user.email,
                            'full_name': current_user.full_name
                        }
                    })
                else:
                    # User no longer exists or is inactive - logout
                    logout_user()
                    return jsonify({'authenticated': False, 'reason': 'user_inactive'}), 401
            else:
                # Processor not available - return unauthenticated
                return jsonify({'authenticated': False, 'reason': 'system_unavailable'}), 401
        else:
            return jsonify({'authenticated': False, 'reason': 'not_logged_in'}), 401
            
    except Exception as e:
        logger.error(f"Auth status check error: {e}")
        return jsonify({'authenticated': False, 'reason': 'validation_error'}), 401

@app.route('/')
def index():
    """Main chat interface - authentication handled by frontend"""
    # Let the frontend handle authentication check to support persistent sessions
    # This prevents immediate redirect on page refresh, allowing JS to validate session
    return render_template('chat.html')

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_files():
    """Handle file uploads with detailed result tracking"""
    try:
        logger.info("Upload request received")
        
        if 'files' not in request.files:
            logger.error("No files in request")
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            logger.error("No files selected")
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        
        # Get project selection from form data
        project_id = request.form.get('project_id', '').strip()
        logger.info(f"Project selection: {project_id}")
        
        if not processor:
            logger.error("Processor not initialized")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        # Validate project belongs to user
        if project_id:
            user_projects = processor.vector_db.get_user_projects(current_user.user_id)
            project_exists = any(p.project_id == project_id for p in user_projects)
            if not project_exists:
                return jsonify({'success': False, 'error': 'Invalid project selection'}), 400
        
        logger.info(f"Processing {len(files)} files for project {project_id or 'default'}")
        
        results = []
        successful_uploads = 0
        
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                logger.info(f"Processing file: {filename}")
                
                file_result = {
                    'filename': filename,
                    'success': False,
                    'error': None,
                    'chunks': 0
                }
                
                try:
                    # Validate file extension
                    if not filename.lower().endswith(('.docx', '.txt', '.pdf')):
                        file_result['error'] = 'Unsupported file format'
                        results.append(file_result)
                        logger.warning(f"Unsupported file format: {filename}")
                        continue
                    
                    # Save file to temp directory
                    temp_path = os.path.join('temp', filename)
                    os.makedirs('temp', exist_ok=True)
                    file.save(temp_path)
                    logger.info(f"File saved to: {temp_path}")
                    
                    # Check file size
                    file_size = os.path.getsize(temp_path)
                    if file_size == 0:
                        file_result['error'] = 'File is empty'
                        results.append(file_result)
                        os.remove(temp_path)
                        continue
                    
                    if file_size > 50 * 1024 * 1024:  # 50MB limit
                        file_result['error'] = 'File too large (max 50MB)'
                        results.append(file_result)
                        os.remove(temp_path)
                        continue
                    
                    # Process document
                    content = processor.read_document_content(temp_path)
                    if not content or not content.strip():
                        file_result['error'] = 'No readable content found'
                        results.append(file_result)
                        os.remove(temp_path)
                        logger.warning(f"No content extracted from {filename}")
                        continue
                    
                    logger.info(f"Content extracted from {filename}, length: {len(content)}")
                    
                    # Parse and process document with user context
                    meeting_doc = processor.parse_document_content(content, filename)
                    
                    # Add user context to document
                    meeting_doc.user_id = current_user.user_id
                    
                    # Use selected project or default project
                    user_projects = processor.vector_db.get_user_projects(current_user.user_id)
                    if user_projects:
                        if project_id:
                            # Use selected project
                            selected_project = next((p for p in user_projects if p.project_id == project_id), None)
                            if selected_project:
                                meeting_doc.project_id = selected_project.project_id
                                logger.info(f"Assigned document to selected project: {selected_project.project_name}")
                        else:
                            # Use default project (first one)
                            default_project = user_projects[0]
                            meeting_doc.project_id = default_project.project_id
                            logger.info(f"Assigned document to default project: {default_project.project_name}")
                        
                        # Create a basic meeting for the document
                        if meeting_doc.project_id:
                            meeting_id = processor.vector_db.create_meeting(
                                current_user.user_id,
                                meeting_doc.project_id,
                                f"Meeting - {filename}",
                                meeting_doc.date
                            )
                            meeting_doc.meeting_id = meeting_id
                    
                    chunks = processor.chunk_document(meeting_doc)
                    
                    # Create project-based folder structure
                    project_folder_name = "Default Project"  # Default fallback
                    if meeting_doc.project_id:
                        # Get the project name for folder creation
                        selected_project = next((p for p in user_projects if p.project_id == meeting_doc.project_id), None)
                        if selected_project:
                            # Sanitize project name for folder creation
                            project_folder_name = selected_project.project_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                            project_folder_name = "".join(c for c in project_folder_name if c.isalnum() or c in ("_", "-"))
                    
                    # Create project-specific folder structure
                    user_folder = f"meeting_documents/user_{current_user.username}"
                    project_folder = os.path.join(user_folder, f"project_{project_folder_name}")
                    os.makedirs(project_folder, exist_ok=True)
                    
                    # Set the folder path for the document
                    folder_path = f"user_{current_user.username}/project_{project_folder_name}"
                    meeting_doc.folder_path = folder_path
                    
                    permanent_path = os.path.join(project_folder, filename)
                    
                    # Handle duplicate filenames
                    counter = 1
                    original_permanent_path = permanent_path
                    while os.path.exists(permanent_path):
                        name, ext = os.path.splitext(original_permanent_path)
                        permanent_path = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    shutil.move(temp_path, permanent_path)
                    
                    # Add document to database with folder path
                    processor.vector_db.add_document(meeting_doc, chunks)
                    
                    # Success!
                    file_result['success'] = True
                    file_result['chunks'] = len(chunks)
                    successful_uploads += 1
                    
                    logger.info(f"Successfully processed {filename} with {len(chunks)} chunks")
                    
                except Exception as e:
                    file_result['error'] = str(e)
                    logger.error(f"Error processing {filename}: {e}")
                    
                    # Clean up temp file if it exists
                    try:
                        temp_path = os.path.join('temp', filename)
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up temp file: {cleanup_error}")
                
                results.append(file_result)
        
        # Save vector index if any files were processed successfully
        if successful_uploads > 0:
            try:
                processor.vector_db.save_index()
                logger.info(f"Vector index saved after processing {successful_uploads} files")
            except Exception as e:
                logger.error(f"Error saving vector index: {e}")
        
        # Prepare response
        response_data = {
            'success': True,
            'results': results,
            'processed': successful_uploads,
            'total': len(results),
            'message': f'Successfully processed {successful_uploads} of {len(results)} files'
        }
        
        logger.info(f"Upload completed: {successful_uploads}/{len(results)} files processed successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Critical upload error: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'processed': 0,
            'total': 0
        }), 500

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        document_ids = data.get('document_ids', None)  # Document filtering
        project_id = data.get('project_id', None)  # Single project filtering (legacy)
        project_ids = data.get('project_ids', None)  # Multiple project filtering (enhanced)
        meeting_ids = data.get('meeting_ids', None)  # Meeting filtering
        date_filters = data.get('date_filters', None)  # Date filtering
        folder_path = data.get('folder_path', None)  # Folder-based filtering
        
        logger.info(f"Chat request received: {message[:100]}...")
        if document_ids:
            logger.info(f"Document filter: {document_ids}")
        if project_id:
            logger.info(f"Project filter: {project_id}")
        if project_ids:
            logger.info(f"Enhanced project filters: {project_ids}")
        if meeting_ids:
            logger.info(f"Meeting filters: {meeting_ids}")
        if date_filters:
            logger.info(f"Date filters: {date_filters}")
        if folder_path:
            logger.info(f"Folder filter: {folder_path}")
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        if not processor:
            logger.error("Processor not initialized for chat")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        # Check if documents are available
        try:
            vector_size = getattr(processor.vector_db.index, 'ntotal', 0) if processor.vector_db.index else 0
            logger.info(f"Vector database size: {vector_size}")
        except Exception as e:
            logger.error(f"Error checking vector database: {e}")
            vector_size = 0
        
        if vector_size == 0:
            response = "I don't have any documents to analyze yet. Please upload some meeting documents first! 📁"
            follow_up_questions = []
            logger.info("No documents available, sending default response")
        else:
            try:
                logger.info("Generating response using processor")
                user_id = current_user.user_id
                
                # Combine project filters (legacy and enhanced)
                combined_project_ids = []
                if project_id:
                    combined_project_ids.append(project_id)
                if project_ids:
                    combined_project_ids.extend(project_ids)
                final_project_id = combined_project_ids[0] if combined_project_ids else None
                
                response, context = processor.answer_query(
                    message, 
                    user_id=user_id, 
                    document_ids=document_ids, 
                    project_id=final_project_id,
                    meeting_ids=meeting_ids,
                    date_filters=date_filters,
                    folder_path=folder_path,
                    context_limit=10, 
                    include_context=True
                )
                logger.info(f"Response generated, length: {len(response)}")
                
                # Generate follow-up questions
                try:
                    follow_up_questions = processor.generate_follow_up_questions(message, response, context)
                    logger.info(f"Generated {len(follow_up_questions)} follow-up questions")
                except Exception as follow_up_error:
                    logger.error(f"Error generating follow-up questions: {follow_up_error}")
                    follow_up_questions = []
                    
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                response = f"I encountered an error while processing your question: {str(e)}"
                follow_up_questions = []
        
        return jsonify({
            'success': True,
            'response': response,
            'follow_up_questions': follow_up_questions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/documents')
@login_required
def get_documents():
    """Get list of all documents for file selection"""
    try:
        logger.info("Documents request received")
        
        if not processor:
            logger.error("Processor not initialized for documents")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        user_id = current_user.user_id
        documents = processor.vector_db.get_all_documents(user_id)
        
        return jsonify({
            'success': True,
            'documents': documents,
            'count': len(documents)
        })
        
    except Exception as e:
        logger.error(f"Documents error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Project Management Endpoints
@app.route('/api/projects')
@login_required
def get_projects():
    """Get all projects for the current user"""
    try:
        logger.info("Projects request received")
        
        if not processor:
            logger.error("Processor not initialized for projects")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        user_id = current_user.user_id
        projects = processor.vector_db.get_user_projects(user_id)
        
        # Convert projects to dictionaries
        project_list = []
        for project in projects:
            project_list.append({
                'project_id': project.project_id,
                'project_name': project.project_name,
                'description': project.description,
                'created_at': project.created_at.isoformat(),
                'is_active': project.is_active
            })
        
        return jsonify({
            'success': True,
            'projects': project_list,
            'count': len(project_list)
        })
        
    except Exception as e:
        logger.error(f"Projects error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/projects', methods=['POST'])
@login_required
def create_project():
    """Create a new project"""
    try:
        data = request.get_json()
        project_name = data.get('project_name', '').strip()
        description = data.get('description', '').strip()
        
        if not project_name:
            return jsonify({'success': False, 'error': 'Project name is required'}), 400
        
        if not processor:
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        user_id = current_user.user_id
        project_id = processor.vector_db.create_project(user_id, project_name, description)
        
        logger.info(f"New project created: {project_name} ({project_id}) for user {current_user.username}")
        return jsonify({
            'success': True,
            'message': 'Project created successfully',
            'project_id': project_id
        })
        
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Create project error: {e}")
        return jsonify({'success': False, 'error': 'Failed to create project'}), 500

@app.route('/api/meetings')
@login_required
def get_meetings():
    """Get all meetings for the current user"""
    try:
        logger.info("Meetings request received")
        
        if not processor:
            logger.error("Processor not initialized for meetings")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        user_id = current_user.user_id
        meetings = processor.vector_db.get_user_meetings(user_id)
        
        # Convert meetings to dictionaries
        meeting_list = []
        for meeting in meetings:
            meeting_list.append({
                'meeting_id': meeting.meeting_id,
                'title': meeting.meeting_name,  # Use meeting_name from the dataclass
                'date': meeting.meeting_date.isoformat() if meeting.meeting_date else None,
                'participants': '',  # This will be populated from documents later
                'project_id': meeting.project_id,
                'created_at': meeting.created_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'meetings': meeting_list,
            'count': len(meeting_list)
        })
        
    except Exception as e:
        logger.error(f"Meetings error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
@login_required
def get_stats():
    """Get system statistics"""
    try:
        logger.info("Stats request received")
        
        if not processor:
            logger.error("Processor not initialized for stats")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        stats = processor.get_meeting_statistics()
        
        if "error" in stats:
            logger.error(f"Error in stats: {stats['error']}")
            return jsonify({'success': False, 'error': stats['error']}), 500
        
        logger.info("Stats generated successfully")
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/refresh', methods=['POST'])
@login_required
def refresh_system():
    """Refresh the system"""
    try:
        logger.info("System refresh requested")
        
        if processor:
            processor.refresh_clients()
            logger.info("System refreshed successfully")
            return jsonify({'success': True, 'message': 'System refreshed successfully'})
        else:
            logger.error("Processor not initialized for refresh")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/test')
def test_system():
    """Test endpoint to check if system is working"""
    try:
        status = {
            'processor_initialized': processor is not None,
            'vector_db_available': False,
            'vector_size': 0
        }
        
        if processor:
            try:
                status['vector_db_available'] = processor.vector_db is not None
                if processor.vector_db and processor.vector_db.index:
                    status['vector_size'] = getattr(processor.vector_db.index, 'ntotal', 0)
            except Exception as e:
                logger.error(f"Error checking vector DB: {e}")
        
        logger.info(f"System test status: {status}")
        return jsonify({'success': True, 'status': status})
        
    except Exception as e:
        logger.error(f"Test error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure required directories exist
    for directory in ['uploads', 'temp', 'meeting_documents', 'logs', 'backups', 'templates', 'static']:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Check if required files exist
    required_files = {
        'templates/chat.html': 'HTML template',
        'static/styles.css': 'CSS stylesheet', 
        'static/script.js': 'JavaScript file'
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{file_path} ({description})")
    
    if missing_files:
        print("Missing required files:")
        for missing in missing_files:
            print(f"   - {missing}")
        exit(1)
    
    # Initialize processor
    if initialize_processor():
        pass
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        pass