#!/usr/bin/env python3
"""
Migration script to move existing files to project-folder structure
and update database with folder_path information.
"""

import os
import shutil
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sanitize_project_name(project_name):
    """Sanitize project name for folder creation"""
    if not project_name:
        return "Default_Project"
    
    # Replace spaces and special characters
    sanitized = project_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in ("_", "-"))
    
    return sanitized if sanitized else "Default_Project"

def get_project_name_by_id(cursor, project_id):
    """Get project name by project ID"""
    cursor.execute("SELECT project_name FROM projects WHERE project_id = ?", (project_id,))
    result = cursor.fetchone()
    return result[0] if result else "Default Project"

def migrate_existing_files():
    """Migrate existing files to project-folder structure"""
    db_path = "meeting_documents.db"
    
    if not os.path.exists(db_path):
        logger.error(f"Database file {db_path} not found")
        return False
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get all documents that need migration (no folder_path set)
        cursor.execute("""
            SELECT document_id, filename, user_id, project_id
            FROM documents 
            WHERE folder_path IS NULL OR folder_path = ''
        """)
        
        documents = cursor.fetchall()
        logger.info(f"Found {len(documents)} documents to migrate")
        
        if not documents:
            logger.info("No documents need migration")
            return True
        
        # Get all projects for reference
        cursor.execute("SELECT project_id, project_name FROM projects")
        projects = dict(cursor.fetchall())
        
        # Get all users for reference
        cursor.execute("SELECT user_id, username FROM users")
        users = dict(cursor.fetchall())
        
        migrated_count = 0
        
        for document_id, filename, user_id, project_id in documents:
            try:
                # Get user info
                username = users.get(user_id, f"user_{user_id}")
                
                # Get project info
                project_name = projects.get(project_id, "Default Project")
                sanitized_project_name = sanitize_project_name(project_name)
                
                # Determine old and new paths
                old_user_folder = f"meeting_documents/user_{username}"
                new_user_folder = f"meeting_documents/user_{username}"
                new_project_folder = f"{new_user_folder}/project_{sanitized_project_name}"
                
                old_file_path = os.path.join(old_user_folder, filename)
                new_file_path = os.path.join(new_project_folder, filename)
                
                # Check if old file exists
                if os.path.exists(old_file_path):
                    # Create new project folder if it doesn't exist
                    os.makedirs(new_project_folder, exist_ok=True)
                    
                    # Move file to new location
                    if not os.path.exists(new_file_path):
                        shutil.move(old_file_path, new_file_path)
                        logger.info(f"Moved {filename} to {new_project_folder}")
                    else:
                        logger.warning(f"File {filename} already exists in {new_project_folder}")
                    
                    # Update database with folder_path
                    folder_path = f"user_{username}/project_{sanitized_project_name}"
                    cursor.execute("""
                        UPDATE documents 
                        SET folder_path = ? 
                        WHERE document_id = ?
                    """, (folder_path, document_id))
                    
                    migrated_count += 1
                    logger.info(f"Updated database for {filename} with folder_path: {folder_path}")
                    
                else:
                    logger.warning(f"File {old_file_path} not found, skipping")
                    
                    # Still update database with expected folder_path
                    folder_path = f"user_{username}/project_{sanitized_project_name}"
                    cursor.execute("""
                        UPDATE documents 
                        SET folder_path = ? 
                        WHERE document_id = ?
                    """, (folder_path, document_id))
                    
            except Exception as e:
                logger.error(f"Error migrating {filename}: {e}")
                continue
        
        # Commit all changes
        conn.commit()
        logger.info(f"Successfully migrated {migrated_count} documents")
        
        # Clean up empty user folders
        cleanup_empty_folders()
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()

def cleanup_empty_folders():
    """Clean up empty user folders after migration"""
    meeting_docs_path = Path("meeting_documents")
    
    if not meeting_docs_path.exists():
        return
    
    # Find all user folders
    for user_folder in meeting_docs_path.glob("user_*"):
        if user_folder.is_dir():
            try:
                # Check if folder is empty (no files, only project folders)
                contents = list(user_folder.iterdir())
                files_in_root = [item for item in contents if item.is_file()]
                
                if files_in_root:
                    logger.info(f"User folder {user_folder} still has files in root, leaving them")
                else:
                    logger.info(f"User folder {user_folder} has been fully migrated to project folders")
                    
            except Exception as e:
                logger.error(f"Error checking {user_folder}: {e}")

def verify_migration():
    """Verify that migration was successful"""
    db_path = "meeting_documents.db"
    
    if not os.path.exists(db_path):
        logger.error(f"Database file {db_path} not found")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check for documents without folder_path
        cursor.execute("""
            SELECT COUNT(*) FROM documents 
            WHERE folder_path IS NULL OR folder_path = ''
        """)
        
        unmigrated_count = cursor.fetchone()[0]
        
        if unmigrated_count > 0:
            logger.warning(f"{unmigrated_count} documents still need migration")
            return False
        
        # Check that all documents have valid folder_path
        cursor.execute("SELECT COUNT(*) FROM documents WHERE folder_path IS NOT NULL")
        migrated_count = cursor.fetchone()[0]
        
        logger.info(f"Migration verification: {migrated_count} documents have folder_path set")
        
        # Show folder structure
        cursor.execute("SELECT DISTINCT folder_path FROM documents WHERE folder_path IS NOT NULL")
        folder_paths = [row[0] for row in cursor.fetchall()]
        
        logger.info("Current folder structure:")
        for folder_path in sorted(folder_paths):
            logger.info(f"  - {folder_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False
        
    finally:
        conn.close()

def main():
    """Main migration function"""
    logger.info("Starting migration to project-folder structure")
    logger.info("=" * 60)
    
    # Backup database before migration
    backup_db()
    
    # Run migration
    if migrate_existing_files():
        logger.info("Migration completed successfully")
        
        # Verify migration
        if verify_migration():
            logger.info("Migration verification passed")
        else:
            logger.error("Migration verification failed")
    else:
        logger.error("Migration failed")
    
    logger.info("=" * 60)
    logger.info("Migration process completed")

def backup_db():
    """Create a backup of the database before migration"""
    db_path = "meeting_documents.db"
    
    if os.path.exists(db_path):
        backup_path = f"meeting_documents_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")

if __name__ == "__main__":
    main()