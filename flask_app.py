from flask import Flask, render_template, request, jsonify
import os
import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import shutil

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

app.config['SECRET_KEY'] = 'uhg-meeting-ai-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Global processor instance
processor = None

def initialize_processor():
    """Initialize the document processor"""
    global processor
    try:
        logger.info("Initializing Enhanced Meeting Document Processor...")
        processor = EnhancedMeetingDocumentProcessor(chunk_size=1000, chunk_overlap=200)
        logger.info("Processor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        return False

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html')

@app.route('/api/upload', methods=['POST'])
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
        
        if not processor:
            logger.error("Processor not initialized")
            return jsonify({'success': False, 'error': 'System not initialized'}), 500
        
        logger.info(f"Processing {len(files)} files")
        
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
                    
                    # Parse and process document
                    meeting_doc = processor.parse_document_content(content, filename)
                    chunks = processor.chunk_document(meeting_doc)
                    processor.vector_db.add_document(meeting_doc, chunks)
                    
                    # Move to permanent storage
                    os.makedirs('meeting_documents', exist_ok=True)
                    permanent_path = os.path.join('meeting_documents', filename)
                    
                    # Handle duplicate filenames
                    counter = 1
                    original_permanent_path = permanent_path
                    while os.path.exists(permanent_path):
                        name, ext = os.path.splitext(original_permanent_path)
                        permanent_path = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    shutil.move(temp_path, permanent_path)
                    
                    # Success!
                    file_result['success'] = True
                    file_result['chunks'] = len(chunks)
                    successful_uploads += 1
                    
                    logger.info(f"✅ Successfully processed {filename} with {len(chunks)} chunks")
                    
                except Exception as e:
                    file_result['error'] = str(e)
                    logger.error(f"❌ Error processing {filename}: {e}")
                    
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
                logger.info(f"✅ Vector index saved after processing {successful_uploads} files")
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
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        logger.info(f"Chat request received: {message[:100]}...")
        
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
            logger.info("No documents available, sending default response")
        else:
            try:
                logger.info("Generating response using processor")
                response = processor.answer_query(message, context_limit=10)
                logger.info(f"Response generated, length: {len(response)}")
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                response = f"I encountered an error while processing your question: {str(e)}"
        
        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
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
        print(f"✅ Created directory: {directory}")
    
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
        print("❌ Missing required files:")
        for missing in missing_files:
            print(f"   - {missing}")
        exit(1)
    
    # Initialize processor
    if initialize_processor():
        print("🚀 Starting UHG Meeting Document AI Flask Server...")
        print("📍 Access the application at: http://localhost:5000")
        print("✅ All static files found and configured")
        print("🧪 Test the system at: http://localhost:5000/api/test")
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to initialize processor. Please check your configuration.")
        print("Make sure meeting_processor.py is available and working.")