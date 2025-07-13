// Global variables
let uploadedFiles = [];
let isProcessing = false;
let conversationHistory = [];
let savedConversations = [];
let currentConversationId = null;
let conversationCounter = 1;

// Document selection variables
let availableDocuments = [];
let selectedDocuments = [];
let isDocumentDropdownOpen = false;

// Enhanced @ mention variables
let availableProjects = [];
let availableMeetings = [];
let availableFolders = [];
let selectedMentions = [];
let currentMentionType = 'document';

// Storage keys
const STORAGE_KEYS = {
    CONVERSATIONS: 'uhg_conversations',
    CURRENT_ID: 'uhg_current_conversation_id',
    COUNTER: 'uhg_conversation_counter'
};

// Configure marked.js for robust Markdown parsing
function initializeMarkdownParser() {
    if (typeof marked !== 'undefined') {
        // Configure marked with custom options
        marked.setOptions({
            breaks: true,        // Convert line breaks to <br>
            gfm: true,          // GitHub Flavored Markdown
            headerIds: false,   // Don't add IDs to headers
            mangle: false,      // Don't mangle email addresses
            sanitize: false,    // We'll use DOMPurify for sanitization
            smartLists: true,   // Use smarter list behavior
            smartypants: false, // Don't use smart quotes
            xhtml: false       // Don't output XHTML
        });

        // Custom renderer for better formatting
        const renderer = new marked.Renderer();
        
        // Custom heading renderer
        renderer.heading = function(text, level) {
            const escapedText = text.toLowerCase().replace(/[^\w]+/g, '-');
            return `<h${level} class="markdown-heading">${text}</h${level}>`;
        };

        // Custom code block renderer
        renderer.code = function(code, language) {
            const validLang = language && /^[a-zA-Z0-9_+-]*$/.test(language) ? language : '';
            const escapedCode = code.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return `<pre class="markdown-code-block"><code class="language-${validLang}">${escapedCode}</code></pre>`;
        };

        // Custom inline code renderer
        renderer.codespan = function(text) {
            return `<code class="markdown-inline-code">${text}</code>`;
        };

        // Custom blockquote renderer
        renderer.blockquote = function(quote) {
            return `<blockquote class="markdown-blockquote">${quote}</blockquote>`;
        };

        // Custom table renderer
        renderer.table = function(header, body) {
            return `<table class="markdown-table">
                <thead>${header}</thead>
                <tbody>${body}</tbody>
            </table>`;
        };

        // Custom list renderer
        renderer.list = function(body, ordered, start) {
            const type = ordered ? 'ol' : 'ul';
            const startAttr = (ordered && start !== 1) ? ` start="${start}"` : '';
            return `<${type}${startAttr} class="markdown-list">${body}</${type}>`;
        };

        // Custom link renderer (for security)
        renderer.link = function(href, title, text) {
            // Only allow safe URLs
            const safeProtocols = ['http:', 'https:', 'mailto:'];
            let isValidUrl = false;
            
            try {
                const url = new URL(href);
                isValidUrl = safeProtocols.includes(url.protocol);
            } catch (e) {
                isValidUrl = false;
            }
            
            if (!isValidUrl) {
                return text; // Return just the text if URL is unsafe
            }
            
            const titleAttr = title ? ` title="${title}"` : '';
            return `<a href="${href}"${titleAttr} target="_blank" rel="noopener noreferrer" class="markdown-link">${text}</a>`;
        };

        marked.use({ renderer });
        
        return true;
    } else {
        return false;
    }
}

// Enhanced Markdown to HTML conversion with security
function formatMarkdownToHTML(text) {
    if (!text) return '';
    
    try {
        // Check if marked.js is available
        if (typeof marked === 'undefined') {
            return formatMarkdownToHTMLBasic(text);
        }

        // Parse Markdown to HTML using marked.js
        let html = marked.parse(text);
        
        // Sanitize the HTML using DOMPurify if available
        if (typeof DOMPurify !== 'undefined') {
            html = DOMPurify.sanitize(html, {
                ALLOWED_TAGS: [
                    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                    'p', 'br', 'strong', 'b', 'em', 'i', 'u',
                    'code', 'pre', 'blockquote',
                    'ul', 'ol', 'li',
                    'table', 'thead', 'tbody', 'tr', 'th', 'td',
                    'a', 'img',
                    'div', 'span'
                ],
                ALLOWED_ATTR: [
                    'href', 'title', 'target', 'rel',
                    'src', 'alt', 'width', 'height',
                    'class', 'start'
                ],
                ALLOWED_URI_REGEXP: /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|sms|cid|xmpp):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i
            });
        }
        
        return html;
        
    } catch (error) {
        return formatMarkdownToHTMLBasic(text);
    }
}

// Fallback basic Markdown parser (improved version of original)
function formatMarkdownToHTMLBasic(text) {
    if (!text) return '';
    
    let html = text;
    
    // Convert headers (#### becomes h4, etc.)
    html = html.replace(/^#### (.*$)/gim, '<h4 class="markdown-heading">$1</h4>');
    html = html.replace(/^### (.*$)/gim, '<h3 class="markdown-heading">$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2 class="markdown-heading">$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1 class="markdown-heading">$1</h1>');
    
    // Convert bold text
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
    
    // Convert italic text
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/_(.*?)_/g, '<em>$1</em>');
    
    // Convert code blocks
    html = html.replace(/```([\s\S]*?)```/g, '<pre class="markdown-code-block"><code>$1</code></pre>');
    
    // Convert inline code
    html = html.replace(/`(.*?)`/g, '<code class="markdown-inline-code">$1</code>');
    
    // Convert blockquotes
    html = html.replace(/^> (.*$)/gim, '<blockquote class="markdown-blockquote">$1</blockquote>');
    
    // Convert unordered lists
    html = html.replace(/^\* (.*$)/gim, '<li>$1</li>');
    html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
    
    // Convert numbered lists
    html = html.replace(/^\d+\. (.*$)/gim, '<li>$1</li>');
    
    // Wrap consecutive list items in ul/ol tags
    html = html.replace(/(<li>.*<\/li>)/gs, function(match) {
        if (match.includes('</li>\n<li>') || match.includes('</li><li>')) {
            return '<ul class="markdown-list">' + match + '</ul>';
        }
        return '<ul class="markdown-list">' + match + '</ul>';
    });
    
    // Convert line breaks to proper HTML
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    
    // Wrap in paragraphs if not already wrapped
    if (!html.startsWith('<') || (!html.includes('<p>') && !html.includes('<h') && !html.includes('<ul>') && !html.includes('<blockquote>'))) {
        html = '<p>' + html + '</p>';
    }
    
    // Clean up empty paragraphs
    html = html.replace(/<p><\/p>/g, '');
    html = html.replace(/<p><br><\/p>/g, '');
    
    return html;
}


// Initialize the app
// Find this line in the DOMContentLoaded event listener:
document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize markdown parser
    initializeMarkdownParser();
    
    // Check authentication status first
    checkAuthenticationStatus();
    
    // Setup event listeners
    setupEventListeners();
    
    // Load system stats
    loadSystemStats();
    
    // Setup textarea auto-resize
    autoResize();
    
    // Load persisted data and initialize UI
    loadPersistedDataAndInitializeUI();
    
    // Initialize the app
    initializeApp();
    
    // Setup auto-save
    setupAutoSave();
    
    // Initialize mobile fixes
    initializeMobileFixes();
    
    // Load documents for @ mentions
    loadDocuments();
    
    // Load projects and meetings for enhanced @ mentions
    loadProjects();
    loadMeetings();
    loadFolders();
    
    // Setup @ mention detection
    setupAtMentionDetection();
    
});

// Enhanced event listeners setup
function setupEventListeners() {
    
    const messageInput = document.getElementById('message-input');
    if (messageInput) {
        messageInput.addEventListener('input', autoResize);
        messageInput.addEventListener('keydown', handleKeyPress);
    }

    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    const uploadArea = document.getElementById('upload-area');
    if (uploadArea) {
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    }

    // Enhanced page unload handling
    window.addEventListener('beforeunload', function(e) {
        if (conversationHistory.length > 0) {
            saveCurrentConversationToPersistentStorage();
        }
        persistAllData();
    });

    // Enhanced visibility change handling (for tab switching)
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            // Page is hidden, save current state
            if (conversationHistory.length > 0) {
                saveCurrentConversationToPersistentStorage();
            }
        }
    });
}

// Persistent Storage Functions
function saveToLocalStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
    } catch (error) {
    }
}

function loadFromLocalStorage(key, defaultValue = null) {
    try {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : defaultValue;
    } catch (error) {
        return defaultValue;
    }
}

// Enhanced data loading with proper UI initialization
function loadPersistedDataAndInitializeUI() {
    
    // Load saved conversations
    const savedData = loadFromLocalStorage(STORAGE_KEYS.CONVERSATIONS, []);
    savedConversations = savedData;
    // Load current conversation ID
    currentConversationId = loadFromLocalStorage(STORAGE_KEYS.CURRENT_ID, null);
    // Load conversation counter
    conversationCounter = loadFromLocalStorage(STORAGE_KEYS.COUNTER, 1);

    // Initialize UI based on current conversation state
    if (currentConversationId) {
        const currentConv = savedConversations.find(c => c.id === currentConversationId);
        if (currentConv) {
            conversationHistory = [...currentConv.history];
            loadConversationUI();
            updateChatTitle(currentConv.title);
        } else {
            currentConversationId = null;
            conversationHistory = [];
            showWelcomeScreen();
            saveToLocalStorage(STORAGE_KEYS.CURRENT_ID, null);
        }
    } else {
        conversationHistory = [];
        showWelcomeScreen();
    }

    // Always update conversation list
    updateConversationList();
}

function persistAllData() {
    saveToLocalStorage(STORAGE_KEYS.CONVERSATIONS, savedConversations);
    saveToLocalStorage(STORAGE_KEYS.CURRENT_ID, currentConversationId);
    saveToLocalStorage(STORAGE_KEYS.COUNTER, conversationCounter);
}

function saveCurrentConversationToPersistentStorage() {
    if (conversationHistory.length > 0) {
        saveCurrentConversation();
        persistAllData();
    }
}

// New helper function to clear messages area properly
function clearMessagesArea() {
    const messagesArea = document.getElementById('messages-area');
    messagesArea.innerHTML = '';
    
    // Hide welcome screen since we're loading a conversation
    const welcomeScreen = document.getElementById('welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }
}

// Updated loadConversationUI function with better error handling
function loadConversationUI() {
    
    // Ensure welcome screen is hidden
    const welcomeScreen = document.getElementById('welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }

    const messagesArea = document.getElementById('messages-area');
    
    // Clear existing content
    messagesArea.innerHTML = '';
    
    // Add conversations from history
    conversationHistory.forEach((msg, index) => {
        addMessageToUI(msg.role, msg.content, false); // false = don't update history
    });
    
    // Scroll to bottom after loading all messages
    setTimeout(() => {
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }, 100);
    
}

// Enhanced showWelcomeScreen function
function showWelcomeScreen() {
    
    const messagesArea = document.getElementById('messages-area');
    messagesArea.innerHTML = `
        <div class="welcome-screen" id="welcome-screen">
            <div class="welcome-icon">🤖</div>
            <div class="welcome-title">Welcome to Document Fulfillment</div>
            <div class="welcome-text">
                Upload your meeting documents and start asking questions. I can help you analyze meeting content, extract key insights, find action items, and track discussions across multiple documents using advanced AI.
            </div>
            
            <div class="sample-prompts">
                <div class="sample-prompts-title">Try these sample prompts:</div>
                <div class="sample-prompt-grid">
                    <button class="sample-prompt" onclick="insertSampleQuery('Summarize all meetings')">
                        📄 Summarize all meetings
                    </button>
                    <button class="sample-prompt" onclick="insertSampleQuery('What are the main topics discussed in recent meetings?')">
                        What are the main topics discussed in recent meetings?
                    </button>
                    <button class="sample-prompt" onclick="insertSampleQuery('Give me an overview of all meeting highlights')">
                        📋 Give me an overview of all meeting highlights
                    </button>
                    <button class="sample-prompt" onclick="insertSampleQuery('List all action items from last week\\'s meetings')">
                        List all action items from last week's meetings
                    </button>
                    <button class="sample-prompt" onclick="insertSampleQuery('Provide a comprehensive summary of all meetings')">
                        🗂️ Provide a comprehensive summary of all meetings
                    </button>
                    <button class="sample-prompt" onclick="insertSampleQuery('Summarize decisions made in project meetings')">
                        Summarize decisions made in project meetings
                    </button>
                </div>
            </div>
        </div>
    `;
    
    // Reset chat title
    updateChatTitle('UHG Meeting Document AI');
}

function autoResize() {
    const textarea = document.getElementById('message-input');
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Function to detect summary queries in frontend
function detectSummaryQuery(query) {
    const summaryKeywords = [
        'summarize', 'summary', 'summaries', 'overview', 'brief', 
        'recap', 'highlights', 'key points', 'main points',
        'all meetings', 'all documents', 'overall', 'across all',
        'consolidate', 'aggregate', 'compile', 'comprehensive',
        'meetings summary', 'meeting summaries', 'summarize meetings',
        'summarize the meetings', 'summary of meetings', 'summary of all'
    ];
    
    const queryLower = query.toLowerCase();
    return summaryKeywords.some(keyword => queryLower.includes(keyword));
}

// Updated sendMessage function to handle conversation context properly
async function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (!message || isProcessing) return;

    isProcessing = true;
    
    // Hide welcome screen
    const welcomeScreen = document.getElementById('welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }
    
    // Clear any existing follow-up questions
    clearFollowUpQuestions();
    
    // Hide document dropdown if open
    hideDocumentDropdown();

    // Parse message for document selection and enhanced @ mentions
    const { cleanMessage, documentIds, projectIds, meetingIds, dateFilters, folderPath } = parseMessageForDocuments(message);
    
    // Debug logging
    console.log('Parsed message data:', { cleanMessage, documentIds, projectIds, meetingIds, dateFilters, folderPath });
    if (folderPath) {
        console.log('✅ Folder path detected:', folderPath);
    }
    
    // Check if this is a summary query and show notification
    const isSummaryQuery = detectSummaryQuery(message);
    if (isSummaryQuery && !documentIds) {
        showNotification('📊 Processing summary across all available documents...');
    }

    // Add user message to UI and history
    addMessageToUI('user', message, true);
    conversationHistory.push({
        role: 'user', 
        content: message, 
        timestamp: new Date().toISOString()
    });
    
    input.value = '';
    selectedDocuments = []; // Clear selected documents after sending
    selectedMentions = []; // Clear selected mentions after sending
    updateSelectedDocuments();
    autoResize();

    // Show typing indicator
    showTypingIndicator();

    try {
        const requestBody = { message: cleanMessage };
        if (documentIds) {
            requestBody.document_ids = documentIds;
        }
        if (projectIds) {
            requestBody.project_ids = projectIds;
        }
        if (meetingIds) {
            requestBody.meeting_ids = meetingIds;
        }
        if (dateFilters) {
            requestBody.date_filters = dateFilters;
        }
        if (folderPath) {
            requestBody.folder_path = folderPath;
        }
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        hideTypingIndicator();

        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                // Add assistant response to UI and history
                addMessageToUI('assistant', data.response, true);
                conversationHistory.push({
                    role: 'assistant', 
                    content: data.response, 
                    timestamp: new Date().toISOString()
                });
                
                // Add follow-up questions if available
                if (data.follow_up_questions && data.follow_up_questions.length > 0) {
                    addFollowUpQuestions(data.follow_up_questions);
                }
                
                // Auto-save conversation after each exchange
                saveCurrentConversationToPersistentStorage();
                
            } else {
                addMessageToUI('assistant', 'Sorry, I encountered an error: ' + (data.error || 'Unknown error'), true);
            }
        } else {
            addMessageToUI('assistant', 'Sorry, I\'m having trouble connecting to the server. Please try again.', true);
        }
    } catch (error) {
        hideTypingIndicator();
        addMessageToUI('assistant', 'Sorry, I\'m having trouble processing your request. Please check your connection and try again.', true);
    } finally {
        isProcessing = false;
    }
}

// Updated addMessageToUI function with better history management
function addMessageToUI(sender, content, updateHistory = true) {
    const messagesArea = document.getElementById('messages-area');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = sender === 'user' ? 'U' : 'AI';

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Format content based on sender
    if (sender === 'assistant') {
        // Use enhanced Markdown parser for AI responses
        messageContent.innerHTML = formatMarkdownToHTML(content);
    } else {
        // User messages as plain text (no Markdown parsing)
        messageContent.textContent = content;
    }

    if (sender === 'assistant') {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        actionsDiv.innerHTML = `
            <button class="message-action-btn" title="Like">👍</button>
            <button class="message-action-btn" title="Dislike">👎</button>
            <button class="message-action-btn" title="Copy" onclick="copyToClipboard(\`${content.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`)">📋</button>
            <button class="message-action-btn" title="Regenerate" onclick="regenerateResponse()">🔄</button>
        `;
        messageContent.appendChild(actionsDiv);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    messagesArea.appendChild(messageDiv);

    // Scroll to bottom
    messagesArea.scrollTop = messagesArea.scrollHeight;
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showNotification('Copied to clipboard!');
    }).catch(function() {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        showNotification('Copied to clipboard!');
    });
}

function showNotification(message, type = 'success') {
    
    // Remove any existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(n => n.remove());
    
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    
    // Different colors for different types
    const colors = {
        success: '#28a745',
        error: '#dc3545',
        info: '#17a2b8',
        warning: '#ffc107'
    };
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${colors[type] || colors.success};
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        z-index: 10000;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        animation: slideIn 0.3s ease-out;
        max-width: 400px;
        word-wrap: break-word;
        border-left: 4px solid rgba(255, 255, 255, 0.3);
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

function addFollowUpQuestions(questions) {
    const messagesArea = document.getElementById('messages-area');
    
    // Remove any existing follow-up questions
    const existingFollowUps = messagesArea.querySelectorAll('.follow-up-container');
    existingFollowUps.forEach(container => container.remove());
    
    if (!questions || questions.length === 0) return;
    
    const followUpContainer = document.createElement('div');
    followUpContainer.className = 'follow-up-container';
    
    const headerDiv = document.createElement('div');
    headerDiv.className = 'follow-up-header';
    headerDiv.innerHTML = '<span>💡</span> <span>Suggested follow-up questions:</span>';
    followUpContainer.appendChild(headerDiv);
    
    const questionsDiv = document.createElement('div');
    questionsDiv.className = 'follow-up-questions';
    
    questions.forEach((question, index) => {
        const questionButton = document.createElement('button');
        questionButton.className = 'follow-up-question';
        questionButton.textContent = question;
        questionButton.onclick = () => {
            // Set the question in the input and send it
            const input = document.getElementById('message-input');
            input.value = question;
            autoResize();
            
            // Remove follow-up questions after selection
            followUpContainer.remove();
            
            // Send the message
            sendMessage();
        };
        questionsDiv.appendChild(questionButton);
    });
    
    followUpContainer.appendChild(questionsDiv);
    messagesArea.appendChild(followUpContainer);
    
    // Scroll to show follow-up questions
    messagesArea.scrollTop = messagesArea.scrollHeight;
}

function clearFollowUpQuestions() {
    const messagesArea = document.getElementById('messages-area');
    const existingFollowUps = messagesArea.querySelectorAll('.follow-up-container');
    existingFollowUps.forEach(container => container.remove());
}

// Document Selection Functions
async function loadDocuments() {
    try {
        const response = await fetch('/api/documents');
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                availableDocuments = data.documents;
                console.log('Loaded documents:', availableDocuments);
                console.log('Sample document:', availableDocuments[0]);
                console.log('Sample document folder_path value:', availableDocuments[0]?.folder_path);
                console.log('Sample document folder_path type:', typeof availableDocuments[0]?.folder_path);
                // Also reload folders when documents are loaded to keep them in sync
                await loadFolders();
            }
        }
    } catch (error) {
    }
}

function detectAtMention(input) {
    const text = input.value;
    const cursorPos = input.selectionStart;
    
    // Find @ or # symbols and check if cursor is after one
    let atPos = -1;
    let symbol = '';
    for (let i = cursorPos - 1; i >= 0; i--) {
        if (text[i] === '@' || text[i] === '#') {
            // Check if symbol is at start or after whitespace
            if (i === 0 || /\s/.test(text[i - 1])) {
                atPos = i;
                symbol = text[i];
                break;
            }
        }
        if (/\s/.test(text[i])) {
            break; // Stop at whitespace
        }
    }
    
    if (atPos !== -1) {
        const fullMention = text.substring(atPos + 1, cursorPos);
        
        // Parse mention syntax (@ for projects/meetings/dates, # for folders/files)
        const mentionData = symbol === '#' ? parseFolderMention(fullMention) : parseEnhancedMention(fullMention);
        
        
        return { 
            isActive: true, 
            searchText: fullMention,
            atPos,
            symbol: symbol,
            ...mentionData
        };
    }
    
    return { isActive: false, searchText: '', atPos: -1, type: 'document', searchTerm: '' };
}

// Parse folder mentions with # symbol
function parseFolderMention(mentionText) {
    
    if (mentionText === '') {
        // Just # typed, show folders
        return {
            type: 'folder_list',
            searchTerm: '',
            prefix: '',
            value: ''
        };
    }
    
    // Check if there's a > at the end to show files within folder
    const hasArrow = mentionText.endsWith('>');
    const folderName = hasArrow ? mentionText.slice(0, -1) : mentionText;
    
    
    const result = {
        type: hasArrow ? 'folder_files' : 'folder_selected',
        searchTerm: folderName,
        prefix: '',
        value: folderName,
        showFiles: hasArrow
    };
    
    return result;
}

function parseEnhancedMention(mentionText) {
    // Enhanced @ mention parsing for @project:name, @meeting:name, @date:today syntax
    const colonIndex = mentionText.indexOf(':');
    
    if (colonIndex === -1) {
        // Simple @ mention (legacy document search)
        return {
            type: 'document',
            searchTerm: mentionText,
            prefix: '',
            value: mentionText
        };
    }
    
    const prefix = mentionText.substring(0, colonIndex).toLowerCase();
    const value = mentionText.substring(colonIndex + 1);
    
    // Determine mention type
    switch (prefix) {
        case 'project':
            return {
                type: 'project',
                searchTerm: value,
                prefix: 'project',
                value: value
            };
        case 'meeting':
            return {
                type: 'meeting',
                searchTerm: value,
                prefix: 'meeting',
                value: value
            };
        case 'date':
            return {
                type: 'date',
                searchTerm: value,
                prefix: 'date',
                value: value
            };
        case 'folder':
            // Check if there's a / at the end to show files within folder
            const hasSlash = value.endsWith('/');
            const folderName = hasSlash ? value.slice(0, -1) : value;
            return {
                type: hasSlash ? 'folder_files' : 'folder',
                searchTerm: folderName,
                prefix: 'folder',
                value: folderName,
                showFiles: hasSlash
            };
        case 'file':
            return {
                type: 'file',
                searchTerm: value,
                prefix: 'file',
                value: value
            };
        default:
            // Unknown prefix, treat as document search
            return {
                type: 'document',
                searchTerm: mentionText,
                prefix: '',
                value: mentionText
            };
    }
}

function filterDocuments(searchText) {
    if (!searchText.trim()) {
        return availableDocuments;
    }
    
    const search = searchText.toLowerCase();
    return availableDocuments.filter(doc => 
        doc.filename.toLowerCase().includes(search) ||
        (doc.title && doc.title.toLowerCase().includes(search))
    );
}

function showDocumentDropdown(searchText = '') {
    const dropdown = document.getElementById('document-dropdown');
    const documentList = document.getElementById('document-list');
    
    if (!dropdown || !documentList) {
        return;
    }
    
    // Update header to show both options
    const header = dropdown.querySelector('.document-dropdown-header');
    if (header) {
        header.textContent = '📁 Select Folder or 📄 File';
    }
    
    
    // Show folders first, then individual files
    const filteredFolders = filterFolders(searchText);
    const filteredDocs = availableDocuments.filter(doc => 
        searchText === '' || doc.filename.toLowerCase().includes(searchText.toLowerCase())
    );
    
    
    let html = '';
    
    // Add folders section
    if (filteredFolders.length > 0) {
        html += '<div style="padding: 8px 12px; background: #F3F4F6; font-weight: 600; font-size: 12px; color: #4B5563; border-bottom: 1px solid #E5E7EB;">📁 FOLDERS (use # symbol)</div>';
        html += filteredFolders.map(folder => `
            <div class="document-item folder-item" data-folder-path="${folder.folder_path}" data-folder-name="${folder.display_name}">
                <div class="document-icon">📁</div>
                <div class="document-info">
                    <div class="document-filename" title="${folder.display_name}">${folder.display_name}</div>
                    <div class="document-meta">
                        <div class="document-date">📂 ${folder.folder_path}</div>
                        <div class="document-size">📄 ${getDocumentCountForFolder(folder.folder_path)} files • Use #${folder.display_name}> to browse files</div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    // Add files section
    if (filteredDocs.length > 0) {
        html += '<div style="padding: 8px 12px; background: #F0F9FF; font-weight: 600; font-size: 12px; color: #1E40AF; border-bottom: 1px solid #DBEAFE; margin-top: 4px;">📄 INDIVIDUAL FILES</div>';
        html += filteredDocs.map(doc => {
            const isSelected = selectedDocuments.some(selected => selected.document_id === doc.document_id);
            const date = new Date(doc.date).toLocaleDateString();
            const size = formatFileSize(doc.file_size);
            
            // Add folder info to help identify files
            const folderInfo = doc.folder_path ? doc.folder_path.split('/').pop() : 'Default';
            
            return `
                <div class="document-item file-item ${isSelected ? 'selected' : ''}" data-doc-id="${doc.document_id}">
                    <div class="document-icon">📄</div>
                    <div class="document-info">
                        <div class="document-filename" title="${doc.filename}">${doc.filename}</div>
                        <div class="document-meta">
                            <div class="document-date">📅 ${date}</div>
                            <div class="document-size">📁 ${folderInfo} • 📊 ${size}</div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }
    
    if (html === '') {
        documentList.innerHTML = '<div style="padding: 16px; text-align: center; color: #6B7280;">No folders or files found</div>';
    } else {
        documentList.innerHTML = html;
        
        // Add click listeners for folders
        documentList.querySelectorAll('.folder-item').forEach(item => {
            item.addEventListener('click', () => {
                const folderName = item.dataset.folderName;
                const folderPath = item.dataset.folderPath;
                const folder = filteredFolders.find(f => f.folder_path === folderPath);
                if (folder) {
                    selectMention('folder', folder.display_name, { folder_path: folderPath });
                }
            });
        });
        
        // Add click listeners for files
        documentList.querySelectorAll('.file-item').forEach(item => {
            item.addEventListener('click', () => {
                const docId = item.dataset.docId;
                const doc = filteredDocs.find(d => d.document_id === docId);
                if (doc) {
                    selectSingleFile(doc);
                }
            });
        });
    }
    
    dropdown.classList.add('active');
    isDocumentDropdownOpen = true;
    
    // Smart positioning: show above input if no space below
    setTimeout(() => {
        const rect = dropdown.getBoundingClientRect();
        const viewportHeight = window.innerHeight;
        const spaceBelow = viewportHeight - rect.top;
        const dropdownHeight = 300; // max-height from CSS
        
        if (spaceBelow < dropdownHeight && rect.top > dropdownHeight) {
            // Not enough space below, show above
            dropdown.style.top = 'auto';
            dropdown.style.bottom = '100%';
            dropdown.style.marginTop = '0';
            dropdown.style.marginBottom = '4px';
        } else {
            // Enough space below or not enough space above, show below
            dropdown.style.top = '100%';
            dropdown.style.bottom = 'auto';
            dropdown.style.marginTop = '4px';
            dropdown.style.marginBottom = '0';
        }
    }, 0);
}

function showFolderFilesDropdown(folderName, searchText = '') {
    
    const dropdown = document.getElementById('document-dropdown');
    const documentList = document.getElementById('document-list');
    
    if (!dropdown || !documentList) {
        return;
    }
    
    // Update header
    const header = dropdown.querySelector('.document-dropdown-header');
    if (header) {
        header.textContent = `📄 Files in ${folderName}`;
    }
    
    // Find the folder
    const folder = availableFolders.find(f => f.display_name === folderName);
    
    if (folder) {
        // Get files from this folder - use multiple strategies
        let folderDocs = [];
        
        // Strategy 1: If folder has pre-stored documents (from folder creation)
        if (folder.documents && folder.documents.length > 0) {
            folderDocs = folder.documents.filter(doc => 
                searchText === '' || doc.filename.toLowerCase().includes(searchText.toLowerCase())
            );
        } else {
            
            // Strategy 2: Match by project_id or show all documents if default folder
            if (folder.project_id === 'default' || folderName === 'Default Folder') {
                // For default folder, show all documents
                folderDocs = availableDocuments.filter(doc => 
                    searchText === '' || doc.filename.toLowerCase().includes(searchText.toLowerCase())
                );
            } else {
                // Match by project_id
                folderDocs = availableDocuments.filter(doc => {
                    const matchesByProjectId = doc.project_id === folder.project_id;
                    const matchesByFolderPath = doc.folder_path === folder.folder_path;
                    const matchesSearch = searchText === '' || 
                                        doc.filename.toLowerCase().includes(searchText.toLowerCase());
                    
                    return (matchesByProjectId || matchesByFolderPath) && matchesSearch;
                });
            }
        }
        
        if (folderDocs.length === 0) {
            documentList.innerHTML = '<div style="padding: 16px; text-align: center; color: #6B7280;">No files found in this folder</div>';
        } else {
            documentList.innerHTML = folderDocs.map(doc => {
                const isSelected = selectedDocuments.some(selected => selected.document_id === doc.document_id);
                const date = new Date(doc.date).toLocaleDateString();
                const size = formatFileSize(doc.file_size);
                
                return `
                    <div class="document-item ${isSelected ? 'selected' : ''}" data-doc-id="${doc.document_id}">
                        <div class="document-icon">📄</div>
                        <div class="document-info">
                            <div class="document-filename" title="${doc.filename}">${doc.filename}</div>
                            <div class="document-meta">
                                <div class="document-date">📅 ${date}</div>
                                <div class="document-size">📊 ${size}</div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            // Add click listeners for files
            documentList.querySelectorAll('.document-item').forEach(item => {
                item.addEventListener('click', () => {
                    const docId = item.dataset.docId;
                    const doc = folderDocs.find(d => d.document_id === docId);
                    if (doc) {
                        selectDocument(doc);
                    }
                });
            });
        }
    } else {
        documentList.innerHTML = '<div style="padding: 16px; text-align: center; color: #6B7280;">Folder not found</div>';
    }
    
    dropdown.classList.add('active');
    isDocumentDropdownOpen = true;
}

function hideDocumentDropdown() {
    const dropdown = document.getElementById('document-dropdown');
    dropdown.classList.remove('active');
    isDocumentDropdownOpen = false;
}

// Enhanced @ mention dropdown functions
function showEnhancedDropdown(mention) {
    currentMentionType = mention.type;
    
    switch (mention.type) {
        case 'document':
            showDocumentDropdown(mention.searchTerm);
            break;
        case 'project':
            showProjectDropdown(mention.searchTerm);
            break;
        case 'meeting':
            showMeetingDropdown(mention.searchTerm);
            break;
        case 'date':
            showDateDropdown(mention.searchTerm);
            break;
        case 'folder':
            showFolderDropdown(mention.searchTerm);
            break;
        case 'folder_list':
            showFolderListDropdown(mention.searchTerm);
            break;
        case 'folder_selected':
            showSelectedFolderDropdown(mention.searchTerm);
            break;
        case 'folder_files':
            showFolderFilesDropdown(mention.searchTerm);
            break;
        default:
            showDocumentDropdown(mention.searchTerm);
    }
}

function showFolderListDropdown(searchText = '') {
    const dropdown = document.getElementById('document-dropdown');
    const documentList = document.getElementById('document-list');
    
    if (!dropdown || !documentList) {
        return;
    }
    
    // Update header
    const header = dropdown.querySelector('.document-dropdown-header');
    if (header) {
        header.textContent = '📁 Select Folder (# symbol)';
    }
    
    const filteredFolders = filterFolders(searchText);
    
    if (filteredFolders.length === 0) {
        documentList.innerHTML = '<div style="padding: 16px; text-align: center; color: #6B7280;">No folders found</div>';
    } else {
        documentList.innerHTML = filteredFolders.map(folder => `
            <div class="document-item folder-item" data-folder-name="${folder.display_name}">
                <div class="document-icon">📁</div>
                <div class="document-info">
                    <div class="document-filename" title="${folder.display_name}">${folder.display_name}</div>
                    <div class="document-meta">
                        <div class="document-date">📂 Type "#${folder.display_name}>" to browse files</div>
                        <div class="document-size">📄 ${getDocumentCountForFolder(folder.folder_path)} files</div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    // Add click listeners for folders
    documentList.querySelectorAll('.folder-item').forEach(item => {
        item.addEventListener('click', () => {
            const folderName = item.dataset.folderName;
            selectFolderMention(folderName);
        });
    });
    
    dropdown.classList.add('active');
    isDocumentDropdownOpen = true;
}

function showSelectedFolderDropdown(folderName) {
    const dropdown = document.getElementById('document-dropdown');
    const documentList = document.getElementById('document-list');
    
    if (!dropdown || !documentList) {
        return;
    }
    
    // Update header
    const header = dropdown.querySelector('.document-dropdown-header');
    if (header) {
        header.textContent = `📁 Folder Selected: ${folderName}`;
    }
    
    documentList.innerHTML = `
        <div style="padding: 16px; text-align: center;">
            <div style="margin-bottom: 12px;">
                <strong>📁 ${folderName}</strong>
            </div>
            <div style="color: #6B7280; font-size: 14px; margin-bottom: 8px;">
                This will search all files in this folder
            </div>
            <div style="color: #9CA3AF; font-size: 12px;">
                Add ">" to browse individual files: #${folderName}>
            </div>
        </div>
    `;
    
    dropdown.classList.add('active');
    isDocumentDropdownOpen = true;
}

function selectFolderMention(folderName) {
    const input = document.getElementById('message-input');
    const mention = detectAtMention(input);
    
    if (mention.isActive) {
        const text = input.value;
        const beforeSymbol = text.substring(0, mention.atPos);
        const afterMention = text.substring(input.selectionStart);
        
        // Replace the # mention with the selected folder
        const replacement = `#${folderName}`;
        
        input.value = beforeSymbol + replacement + ' ' + afterMention;
        input.focus();
        
        // Store the mention data for message processing
        selectedMentions.push({
            type: 'folder_selected',
            displayName: folderName,
            data: { folder_name: folderName }
        });
    }
    
    hideEnhancedDropdown();
}

function hideEnhancedDropdown() {
    hideDocumentDropdown();
}

function showProjectDropdown(searchText = '') {
    const dropdown = document.getElementById('document-dropdown');
    const documentList = document.getElementById('document-list');
    
    if (!dropdown || !documentList) {
        return;
    }
    
    // Update header to indicate project selection
    const header = dropdown.querySelector('.document-dropdown-header');
    if (header) {
        header.textContent = '📁 Select Project';
    }
    
    const filteredProjects = filterProjects(searchText);
    
    if (filteredProjects.length === 0) {
        documentList.innerHTML = '<div style="padding: 16px; text-align: center; color: #6B7280;">No matching projects found</div>';
    } else {
        documentList.innerHTML = filteredProjects.map(project => `
            <div class="document-item" data-project-id="${project.project_id}">
                <div class="document-icon">📁</div>
                <div class="document-info">
                    <div class="document-filename" title="${project.project_name}">${project.project_name}</div>
                    <div class="document-meta">
                        <div class="document-date">📅 ${new Date(project.created_at).toLocaleDateString()}</div>
                        <div class="document-size">📝 ${project.description || 'No description'}</div>
                    </div>
                </div>
            </div>
        `).join('');
        
        // Add click listeners
        documentList.querySelectorAll('.document-item').forEach(item => {
            item.addEventListener('click', () => {
                const projectId = item.dataset.projectId;
                const project = filteredProjects.find(p => p.project_id === projectId);
                if (project) {
                    selectMention('project', project.project_name, { project_id: projectId });
                }
            });
        });
    }
    
    dropdown.classList.add('active');
    isDocumentDropdownOpen = true;
}

function showMeetingDropdown(searchText = '') {
    const dropdown = document.getElementById('document-dropdown');
    const documentList = document.getElementById('document-list');
    
    if (!dropdown || !documentList) {
        return;
    }
    
    // Update header to indicate meeting selection
    const header = dropdown.querySelector('.document-dropdown-header');
    if (header) {
        header.textContent = '📋 Select Meeting';
    }
    
    const filteredMeetings = filterMeetings(searchText);
    
    if (filteredMeetings.length === 0) {
        documentList.innerHTML = '<div style="padding: 16px; text-align: center; color: #6B7280;">No matching meetings found</div>';
    } else {
        documentList.innerHTML = filteredMeetings.map(meeting => `
            <div class="document-item" data-meeting-id="${meeting.meeting_id}">
                <div class="document-icon">📋</div>
                <div class="document-info">
                    <div class="document-filename" title="${meeting.title}">${meeting.title}</div>
                    <div class="document-meta">
                        <div class="document-date">📅 ${new Date(meeting.date).toLocaleDateString()}</div>
                        <div class="document-size">👥 ${meeting.participants || 'No participants listed'}</div>
                    </div>
                </div>
            </div>
        `).join('');
        
        // Add click listeners
        documentList.querySelectorAll('.document-item').forEach(item => {
            item.addEventListener('click', () => {
                const meetingId = item.dataset.meetingId;
                const meeting = filteredMeetings.find(m => m.meeting_id === meetingId);
                if (meeting) {
                    selectMention('meeting', meeting.title, { meeting_id: meetingId });
                }
            });
        });
    }
    
    dropdown.classList.add('active');
    isDocumentDropdownOpen = true;
}

function showDateDropdown(searchText = '') {
    const dropdown = document.getElementById('document-dropdown');
    const documentList = document.getElementById('document-list');
    
    if (!dropdown || !documentList) {
        return;
    }
    
    // Update header to indicate date selection
    const header = dropdown.querySelector('.document-dropdown-header');
    if (header) {
        header.textContent = '📅 Select Date';
    }
    
    const dateOptions = getDateOptions(searchText);
    
    documentList.innerHTML = dateOptions.map(dateOption => `
        <div class="document-item" data-date-value="${dateOption.value}">
            <div class="document-icon">📅</div>
            <div class="document-info">
                <div class="document-filename">${dateOption.label}</div>
                <div class="document-meta">
                    <div class="document-date">${dateOption.description}</div>
                </div>
            </div>
        </div>
    `).join('');
    
    // Add click listeners
    documentList.querySelectorAll('.document-item').forEach(item => {
        item.addEventListener('click', () => {
            const dateValue = item.dataset.dateValue;
            const dateOption = dateOptions.find(d => d.value === dateValue);
            if (dateOption) {
                selectMention('date', dateOption.value, { date_filter: dateValue });
            }
        });
    });
    
    dropdown.classList.add('active');
    isDocumentDropdownOpen = true;
}

function showFolderDropdown(searchText = '') {
    const dropdown = document.getElementById('document-dropdown');
    const documentList = document.getElementById('document-list');
    
    if (!dropdown || !documentList) {
        return;
    }
    
    // Update header to indicate folder selection
    const header = dropdown.querySelector('.document-dropdown-header');
    if (header) {
        header.textContent = '📁 Select Folder';
    }
    
    const filteredFolders = filterFolders(searchText);
    
    if (filteredFolders.length === 0) {
        documentList.innerHTML = '<div style="padding: 16px; text-align: center; color: #6B7280;">No matching folders found</div>';
    } else {
        documentList.innerHTML = filteredFolders.map(folder => `
            <div class="document-item" data-folder-path="${folder.folder_path}">
                <div class="document-icon">📁</div>
                <div class="document-info">
                    <div class="document-filename" title="${folder.display_name}">${folder.display_name}</div>
                    <div class="document-meta">
                        <div class="document-date">📂 ${folder.folder_path}</div>
                        <div class="document-size">📄 ${getDocumentCountForFolder(folder.folder_path)} documents</div>
                    </div>
                </div>
            </div>
        `).join('');
        
        // Add click listeners
        documentList.querySelectorAll('.document-item').forEach(item => {
            item.addEventListener('click', () => {
                const folderPath = item.dataset.folderPath;
                const folder = filteredFolders.find(f => f.folder_path === folderPath);
                if (folder) {
                    selectMention('folder', folder.display_name, { folder_path: folderPath });
                }
            });
        });
    }
    
    dropdown.classList.add('active');
    isDocumentDropdownOpen = true;
}

function selectDocument(doc) {
    // Check if already selected
    if (selectedDocuments.some(selected => selected.document_id === doc.document_id)) {
        return;
    }
    
    selectedDocuments.push(doc);
    updateSelectedDocuments();
    hideDocumentDropdown();
    
    // Clear the @ mention from input
    const input = document.getElementById('message-input');
    const mention = detectAtMention(input);
    if (mention.isActive) {
        const text = input.value;
        const beforeAt = text.substring(0, mention.atPos);
        const afterMention = text.substring(input.selectionStart);
        input.value = beforeAt + afterMention;
        input.focus();
    }
}

function removeDocument(docId) {
    selectedDocuments = selectedDocuments.filter(doc => doc.document_id !== docId);
    updateSelectedDocuments();
}

function updateSelectedDocuments() {
    const container = document.getElementById('selected-documents');
    
    if (selectedDocuments.length === 0) {
        container.innerHTML = '';
        return;
    }
    
    container.innerHTML = selectedDocuments.map(doc => `
        <div class="document-pill">
            <span class="document-name" title="${doc.filename}">${doc.filename}</span>
            <button class="remove-btn" onclick="removeDocument('${doc.document_id}')" title="Remove">×</button>
        </div>
    `).join('');
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function parseMessageForDocuments(message) {
    // Extract selected document IDs
    const documentIds = selectedDocuments.map(doc => doc.document_id);
    
    // Parse enhanced @ mentions from the message
    const enhancedMentions = parseEnhancedMentionsFromMessage(message);
    
    // Clean the message by removing processed @ and # mentions
    let cleanMessage = message.trim();
    enhancedMentions.forEach(mention => {
        let mentionText;
        if (mention.type === 'folder_files' || mention.type === 'folder_selected') {
            // Handle # syntax for folders
            mentionText = mention.type === 'folder_files' ? `#${mention.value}>` : `#${mention.value}`;
        } else {
            // Handle @ syntax for other mentions
            mentionText = `@${mention.prefix}:${mention.value}`;
        }
        cleanMessage = cleanMessage.replace(mentionText, '').trim();
    });
    
    // Combine all filtering data
    const filterData = {
        documentIds: documentIds.length > 0 ? documentIds : null,
        projectIds: null,
        meetingIds: null,
        dateFilters: null,
        folderPath: null
    };
    
    // Extract filter data from enhanced mentions
    enhancedMentions.forEach(mention => {
        switch (mention.type) {
            case 'project':
                if (!filterData.projectIds) filterData.projectIds = [];
                const project = availableProjects.find(p => p.project_name === mention.value);
                if (project) {
                    filterData.projectIds.push(project.project_id);
                }
                break;
            case 'meeting':
                if (!filterData.meetingIds) filterData.meetingIds = [];
                const meeting = availableMeetings.find(m => m.title === mention.value);
                if (meeting) {
                    filterData.meetingIds.push(meeting.meeting_id);
                }
                break;
            case 'date':
                if (!filterData.dateFilters) filterData.dateFilters = [];
                filterData.dateFilters.push(mention.value);
                break;
            case 'folder':
            case 'folder_files':
            case 'folder_selected':
                // Case-insensitive folder lookup
                console.log(`Looking for folder: "${mention.value}"`);
                console.log('Available folders:', availableFolders.map(f => ({ name: f.display_name, path: f.folder_path })));
                
                const folder = availableFolders.find(f => 
                    f.display_name.toLowerCase() === mention.value.toLowerCase()
                );
                if (folder) {
                    filterData.folderPath = folder.folder_path;
                    console.log('✅ Found folder:', folder.display_name, 'Path:', folder.folder_path);
                } else {
                    console.log('❌ Folder not found:', mention.value);
                    console.log('Available folder names:', availableFolders.map(f => f.display_name));
                }
                break;
            case 'file':
                if (!filterData.documentIds) filterData.documentIds = [];
                const document = availableDocuments.find(d => d.filename === mention.value);
                if (document) {
                    filterData.documentIds.push(document.document_id);
                } else {
                }
                break;
        }
    });
    
    // Apply folder-based filtering to selected documents
    // If folder is selected and documents are selected, filter documents to only those in the folder
    if (filterData.folderPath && filterData.documentIds) {
        const folderDocuments = availableDocuments.filter(doc => doc.folder_path === filterData.folderPath);
        const folderDocumentIds = folderDocuments.map(doc => doc.document_id);
        filterData.documentIds = filterData.documentIds.filter(id => folderDocumentIds.includes(id));
        
        // If no documents remain after filtering, clear the document IDs to use folder filtering
        if (filterData.documentIds.length === 0) {
            filterData.documentIds = null;
        }
    }
    
    return {
        cleanMessage,
        ...filterData
    };
}

function parseEnhancedMentionsFromMessage(message) {
    const mentions = [];
    console.log('Parsing message for mentions:', message);
    
    // Parse @ mentions (projects, meetings, dates, files)
    const atMentionRegex = /@(project|meeting|date|file):([^@\s]+)/g;
    let match;
    
    while ((match = atMentionRegex.exec(message)) !== null) {
        mentions.push({
            type: match[1],
            prefix: match[1],
            value: match[2],
            fullMatch: match[0]
        });
    }
    
    // Parse # mentions (folders) with > syntax
    const folderMentionRegex = /#([^#>]+)>/g;
    while ((match = folderMentionRegex.exec(message)) !== null) {
        mentions.push({
            type: 'folder_files',
            prefix: '',
            value: match[1].trim(),
            fullMatch: match[0]
        });
    }
    
    // Parse folder selections without > (support multi-word folder names)
    // Capture everything after # until we hit common action words
    const folderSelectionRegex = /#([^#>]+?)(?=\s+(?:give|show|tell|what|how|when|where|summarize|summary|list|find|search|get|provide|explain|analyze|report|create|add|update|delete|help|do)|$)/gi;
    while ((match = folderSelectionRegex.exec(message)) !== null) {
        console.log('Folder regex matched:', match[1].trim(), 'from:', match[0]);
        mentions.push({
            type: 'folder_selected',
            prefix: '',
            value: match[1].trim(),
            fullMatch: match[0]
        });
    }
    
    return mentions;
}

// Enhanced @ mention helper functions
function filterProjects(searchText) {
    if (!searchText.trim()) {
        return availableProjects;
    }
    
    const search = searchText.toLowerCase();
    return availableProjects.filter(project => 
        project.project_name.toLowerCase().includes(search) ||
        (project.description && project.description.toLowerCase().includes(search))
    );
}

function filterMeetings(searchText) {
    if (!searchText.trim()) {
        return availableMeetings;
    }
    
    const search = searchText.toLowerCase();
    return availableMeetings.filter(meeting => 
        meeting.title.toLowerCase().includes(search) ||
        (meeting.participants && meeting.participants.toLowerCase().includes(search))
    );
}

function filterFolders(searchText) {
    if (!searchText.trim()) {
        return availableFolders;
    }
    
    const search = searchText.toLowerCase();
    return availableFolders.filter(folder => 
        folder.display_name.toLowerCase().includes(search) ||
        folder.folder_path.toLowerCase().includes(search)
    );
}

function getDocumentCountForFolder(folderPath) {
    if (!availableDocuments || availableDocuments.length === 0) {
        return 0;
    }
    
    // If looking for documents by folder_path
    const countByFolderPath = availableDocuments.filter(doc => doc.folder_path === folderPath).length;
    
    // If no documents found by folder_path, try to match by project pattern
    if (countByFolderPath === 0 && folderPath.includes('project_')) {
        const projectId = folderPath.split('project_')[1];
        const countByProject = availableDocuments.filter(doc => doc.project_id === projectId).length;
        return countByProject;
    }
    
    return countByFolderPath;
}

function getDateOptions(searchText) {
    const today = new Date();
    
    // Calculate week boundaries (Monday to Sunday)
    const currentWeekStart = new Date(today);
    currentWeekStart.setDate(today.getDate() - today.getDay() + 1);
    const currentWeekEnd = new Date(currentWeekStart);
    currentWeekEnd.setDate(currentWeekStart.getDate() + 6);
    
    const lastWeekStart = new Date(currentWeekStart);
    lastWeekStart.setDate(currentWeekStart.getDate() - 7);
    const lastWeekEnd = new Date(currentWeekStart);
    lastWeekEnd.setDate(currentWeekStart.getDate() - 1);
    
    // Calculate month boundaries
    const currentMonthStart = new Date(today.getFullYear(), today.getMonth(), 1);
    const currentMonthEnd = new Date(today.getFullYear(), today.getMonth() + 1, 0);
    
    const lastMonthStart = new Date(today.getFullYear(), today.getMonth() - 1, 1);
    const lastMonthEnd = new Date(today.getFullYear(), today.getMonth(), 0);
    
    const options = [
        // Current periods
        {
            label: 'Current Week',
            value: 'current_week',
            description: `${currentWeekStart.toLocaleDateString()} - ${currentWeekEnd.toLocaleDateString()}`
        },
        {
            label: 'Current Month',
            value: 'current_month',
            description: `${currentMonthStart.toLocaleDateString()} - ${currentMonthEnd.toLocaleDateString()}`
        },
        {
            label: 'Current Year',
            value: 'current_year',
            description: today.getFullYear().toString()
        },
        
        // Last periods
        {
            label: 'Last Week',
            value: 'last_week',
            description: `${lastWeekStart.toLocaleDateString()} - ${lastWeekEnd.toLocaleDateString()}`
        },
        {
            label: 'Last Month',
            value: 'last_month',
            description: `${lastMonthStart.toLocaleDateString()} - ${lastMonthEnd.toLocaleDateString()}`
        },
        {
            label: 'Last Quarter',
            value: 'last_quarter',
            description: 'Previous 3 months'
        },
        {
            label: 'Last Year',
            value: 'last_year',
            description: (today.getFullYear() - 1).toString()
        },
        
        // Rolling periods
        {
            label: 'Last 7 Days',
            value: 'last_7_days',
            description: 'Rolling 7 days'
        },
        {
            label: 'Last 14 Days',
            value: 'last_14_days',
            description: 'Rolling 2 weeks'
        },
        {
            label: 'Last 30 Days',
            value: 'last_30_days',
            description: 'Rolling 30 days'
        },
        {
            label: 'Last 3 Months',
            value: 'last_3_months',
            description: 'Rolling 90 days'
        },
        {
            label: 'Last 6 Months',
            value: 'last_6_months',
            description: 'Rolling 180 days'
        },
        
        // Recent
        {
            label: 'Recent',
            value: 'recent',
            description: 'Last 30 days'
        }
    ];
    
    if (!searchText.trim()) {
        return options;
    }
    
    const search = searchText.toLowerCase();
    return options.filter(option => 
        option.label.toLowerCase().includes(search) ||
        option.value.toLowerCase().includes(search)
    );
}

function selectMention(type, displayName, data) {
    const input = document.getElementById('message-input');
    const mention = detectAtMention(input);
    
    if (mention.isActive) {
        const text = input.value;
        const beforeAt = text.substring(0, mention.atPos);
        const afterMention = text.substring(input.selectionStart);
        
        // Replace the @ mention with the selected item
        let replacement = '';
        switch (type) {
            case 'project':
                replacement = `@project:${displayName}`;
                break;
            case 'meeting':
                replacement = `@meeting:${displayName}`;
                break;
            case 'date':
                replacement = `@date:${displayName}`;
                break;
            case 'folder':
                replacement = `#${displayName}`;
                break;
            default:
                replacement = displayName;
        }
        
        input.value = beforeAt + replacement + ' ' + afterMention;
        input.focus();
        
        // Store the mention data for message processing
        selectedMentions.push({
            type,
            displayName,
            data
        });
    }
    
    hideEnhancedDropdown();
}

function selectSingleFile(doc) {
    
    const input = document.getElementById('message-input');
    const mention = detectAtMention(input);
    
    if (mention.isActive) {
        const text = input.value;
        const beforeAt = text.substring(0, mention.atPos);
        const afterMention = text.substring(input.selectionStart);
        
        // Replace the @ mention with the selected file
        const replacement = `@file:${doc.filename}`;
        
        input.value = beforeAt + replacement + ' ' + afterMention;
        input.focus();
        
        // Store the mention data for message processing
        selectedMentions.push({
            type: 'file',
            displayName: doc.filename,
            data: { 
                document_id: doc.document_id,
                filename: doc.filename,
                folder_path: doc.folder_path
            }
        });
        
    }
    
    hideEnhancedDropdown();
}

async function loadProjects() {
    try {
        const response = await fetch('/api/projects');
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.projects) {
                availableProjects = data.projects;
            }
        }
    } catch (error) {
    }
}

async function loadMeetings() {
    try {
        const response = await fetch('/api/meetings');
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.meetings) {
                availableMeetings = data.meetings;
            }
        }
    } catch (error) {
    }
}

// Load available folders from documents
async function loadFolders() {
    try {
        // Ensure projects are loaded first
        if (!availableProjects || availableProjects.length === 0) {
            await loadProjects();
        }
        
        const response = await fetch('/api/documents');
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.documents) {
                // Extract unique folder paths from documents
                console.log('All documents in loadFolders:', data.documents);
                data.documents.forEach((doc, i) => {
                    console.log(`Doc ${i}: ${doc.filename}, folder_path: "${doc.folder_path}" (type: ${typeof doc.folder_path}), project: ${doc.project_name}`);
                });
                
                const docsWithFolders = data.documents.filter(doc => doc.folder_path);
                console.log('Documents with folder_path:', docsWithFolders);
                const folderPaths = [...new Set(data.documents
                    .filter(doc => doc.folder_path && doc.folder_path.trim())
                    .map(doc => doc.folder_path))];
                
                console.log('Unique folder paths found:', folderPaths);
                console.log('Folder paths length:', folderPaths.length);
                
                if (folderPaths.length === 0) {
                    // If no folder_path, try to infer from project_id or create default folders
                    
                    // Group documents by project_id if available
                    const projectGroups = {};
                    data.documents.forEach(doc => {
                        const projectId = doc.project_id || 'default';
                        if (!projectGroups[projectId]) {
                            projectGroups[projectId] = [];
                        }
                        projectGroups[projectId].push(doc);
                    });
                    
                    
                    const projectFolders = Object.keys(projectGroups).map((projectId, index) => {
                        let folderName;
                        let actualFolderPath = `user_folder/project_${projectId}`; // fallback
                        
                        if (projectId === 'default') {
                            folderName = 'Default Folder';
                        } else {
                            // Find the actual project name from availableProjects
                            const project = availableProjects.find(p => p.project_id === projectId);
                            folderName = project ? project.project_name : `Project ${index + 1}`;
                            
                            // Use actual folder_path from documents if available
                            const docWithFolderPath = projectGroups[projectId].find(doc => doc.folder_path);
                            if (docWithFolderPath) {
                                actualFolderPath = docWithFolderPath.folder_path;
                                console.log(`Project "${folderName}" (${projectId}) - Using actual folder path: ${actualFolderPath}`);
                            } else {
                                console.log(`Project "${folderName}" (${projectId}) - No folder_path found, using fallback: ${actualFolderPath}`);
                            }
                        }
                        return {
                            folder_path: actualFolderPath,
                            folder_name: folderName,
                            display_name: folderName,
                            project_id: projectId, // Store the actual project_id for matching
                            documents: projectGroups[projectId] // Store the documents in this folder
                        };
                    });
                    
                    // Always add a "Default Folder" that contains all documents if it doesn't exist
                    const hasDefaultFolder = projectFolders.some(f => f.display_name === 'Default Folder');
                    if (!hasDefaultFolder) {
                        projectFolders.unshift({
                            folder_path: 'user_folder/project_default',
                            folder_name: 'Default Folder',
                            display_name: 'Default Folder',
                            project_id: 'default',
                            documents: data.documents // All documents
                        });
                    }
                    
                    availableFolders = projectFolders;
                } else {
                    availableFolders = folderPaths.map(path => {
                        const parts = path.split('/');
                        const folderName = parts[parts.length - 1]; // Get the last part as folder name
                        const folderDocs = data.documents.filter(doc => doc.folder_path === path);
                        
                        // Try to get project_name from documents in this folder
                        let displayName = folderName.replace(/^project_/, '').replace(/_/g, ' '); // Fallback: clean up folder name
                        const docWithProjectName = folderDocs.find(doc => doc.project_name);
                        if (docWithProjectName && docWithProjectName.project_name) {
                            displayName = docWithProjectName.project_name; // Use actual project name
                        }
                        
                        return {
                            folder_path: path,
                            folder_name: folderName,
                            display_name: displayName,
                            project_id: docWithProjectName ? docWithProjectName.project_id : null,
                            documents: folderDocs // Store associated documents
                        };
                    });
                }
                
            } else {
                // Create default folders even if no documents
                availableFolders = [
                    {
                        folder_path: 'user_folder/project_default',
                        folder_name: 'Default Folder',
                        display_name: 'Default Folder'
                    }
                ];
            }
        } else {
            // Create default folders even if API fails
            availableFolders = [
                {
                    folder_path: 'user_folder/project_default',
                    folder_name: 'Default Folder',
                    display_name: 'Default Folder'
                }
            ];
        }
    } catch (error) {
        // Create default folders even if there's an error
        availableFolders = [
            {
                folder_path: 'user_folder/project_default',
                folder_name: 'Default Folder',
                display_name: 'Default Folder'
            }
        ];
    }
    
    // Ensure we always have at least one folder
    if (availableFolders.length === 0) {
        availableFolders = [
            {
                folder_path: 'user_folder/project_default',
                folder_name: 'Default Folder',
                display_name: 'Default Folder'
            }
        ];
    }
}

function setupAtMentionDetection() {
    const input = document.getElementById('message-input');
    
    // Handle @ mention detection on input
    input.addEventListener('input', function(e) {
        const mention = detectAtMention(input);
        
        if (mention.isActive) {
            showEnhancedDropdown(mention);
        } else {
            hideEnhancedDropdown();
        }
    });
    
    // Handle keyboard navigation
    input.addEventListener('keydown', function(e) {
        if (isDocumentDropdownOpen) {
            if (e.key === 'Escape') {
                e.preventDefault();
                hideEnhancedDropdown();
            } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                navigateEnhancedDropdown(e.key === 'ArrowDown' ? 1 : -1);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                selectHighlightedItem();
            }
        }
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (isDocumentDropdownOpen && !e.target.closest('.input-area')) {
            hideEnhancedDropdown();
        }
    });
}

let highlightedDocumentIndex = -1;

function navigateDocumentDropdown(direction) {
    const items = document.querySelectorAll('.document-item:not(.selected)');
    if (items.length === 0) return;
    
    // Remove previous highlight
    items.forEach(item => item.classList.remove('highlighted'));
    
    // Update index
    highlightedDocumentIndex += direction;
    if (highlightedDocumentIndex < 0) highlightedDocumentIndex = items.length - 1;
    if (highlightedDocumentIndex >= items.length) highlightedDocumentIndex = 0;
    
    // Add new highlight
    items[highlightedDocumentIndex].classList.add('highlighted');
    items[highlightedDocumentIndex].scrollIntoView({ block: 'nearest' });
}

function navigateEnhancedDropdown(direction) {
    const items = document.querySelectorAll('.document-item');
    if (items.length === 0) return;
    
    // Remove previous highlight
    items.forEach(item => item.classList.remove('highlighted'));
    
    // Update index
    highlightedDocumentIndex += direction;
    if (highlightedDocumentIndex < 0) highlightedDocumentIndex = items.length - 1;
    if (highlightedDocumentIndex >= items.length) highlightedDocumentIndex = 0;
    
    // Add new highlight
    items[highlightedDocumentIndex].classList.add('highlighted');
    items[highlightedDocumentIndex].scrollIntoView({ block: 'nearest' });
}

function selectHighlightedDocument() {
    const highlighted = document.querySelector('.document-item.highlighted');
    if (highlighted) {
        highlighted.click();
    }
}

function selectHighlightedItem() {
    const highlighted = document.querySelector('.document-item.highlighted');
    if (highlighted) {
        highlighted.click();
    }
}

function showTypingIndicator() {
    document.getElementById('typing-indicator').classList.add('active');
    const messagesArea = document.getElementById('messages-area');
    messagesArea.scrollTop = messagesArea.scrollHeight;
}

function hideTypingIndicator() {
    document.getElementById('typing-indicator').classList.remove('active');
}

// Updated updateConversationList function with better active state management
function updateConversationList() {
    const listContainer = document.getElementById('conversation-list');
    
    // Clear existing conversations except "Getting Started"
    listContainer.innerHTML = '';
    
    // Add saved conversations (newest first)
    const sortedConversations = [...savedConversations].sort((a, b) => 
        new Date(b.lastUpdated) - new Date(a.lastUpdated)
    );
    
    sortedConversations.forEach(conv => {
        const conversationItem = document.createElement('button');
        
        // Set active class based on current conversation ID
        const isActive = conv.id === currentConversationId;
        conversationItem.className = `conversation-item ${isActive ? 'active' : ''}`;
        
        // Add click handler
        conversationItem.onclick = () => {
            loadConversation(conv.id);
        };
        
        // Truncate title if too long
        const title = conv.title.length > 30 ? conv.title.substring(0, 30) + '...' : conv.title;
        
        conversationItem.innerHTML = `
            <span class="conversation-icon">💬</span>
            <span class="conversation-title" title="${conv.title}">${title}</span>
            <button class="conversation-menu-btn" onclick="event.stopPropagation(); showConversationMenu(event, '${conv.id}')" title="More options">
                <span>⋯</span>
            </button>
        `;
        listContainer.appendChild(conversationItem);
    });
    
    // Update "Getting Started" visibility and active state
    const gettingStartedBtn = document.querySelector('.conversation-item');
    if (gettingStartedBtn) {
        // Remove active class first
        gettingStartedBtn.classList.remove('active');
        
        if (savedConversations.length > 0) {
            // Hide getting started if we have conversations and one is active
            if (currentConversationId !== null) {
                gettingStartedBtn.style.display = 'none';
            } else {
                gettingStartedBtn.style.display = 'flex';
                gettingStartedBtn.classList.add('active');
            }
        } else {
            // Show getting started if no conversations exist
            gettingStartedBtn.style.display = 'flex';
            if (currentConversationId === null) {
                gettingStartedBtn.classList.add('active');
            }
        }
    }
}

// Updated saveCurrentConversation function with better error handling
function saveCurrentConversation() {
    if (conversationHistory.length === 0) {
        return;
    }
    
    
    // Create title from first user message
    const firstUserMessage = conversationHistory.find(msg => msg.role === 'user');
    const title = firstUserMessage ? 
        (firstUserMessage.content.length > 35 ? 
            firstUserMessage.content.substring(0, 35) + '...' : 
            firstUserMessage.content) : 
        `Chat ${conversationCounter}`;
    
    try {
        // Save or update conversation
        if (currentConversationId) {
            // Update existing conversation
            const existingConv = savedConversations.find(c => c.id === currentConversationId);
            if (existingConv) {
                existingConv.history = [...conversationHistory];
                existingConv.title = title;
                existingConv.lastUpdated = new Date().toISOString();
            } else {
            }
        } else {
            // Create new conversation
            const newConversation = {
                id: Date.now().toString(),
                title: title,
                history: [...conversationHistory],
                createdAt: new Date().toISOString(),
                lastUpdated: new Date().toISOString()
            };
            savedConversations.unshift(newConversation);
            currentConversationId = newConversation.id;
            conversationCounter++;
        }
        
        updateConversationList();
        updateChatTitle(title);
        
    } catch (error) {
    }
}

// Updated loadConversation function with proper state management
function loadConversation(conversationId) {
    
    const conversation = savedConversations.find(c => c.id === conversationId);
    if (!conversation) {
        return;
    }
    
    // Save current conversation before switching (if there's content and it's different)
    if (currentConversationId && 
        currentConversationId !== conversationId && 
        conversationHistory.length > 0) {
        saveCurrentConversationToPersistentStorage();
    }
    
    // Set new conversation as current
    currentConversationId = conversationId;
    conversationHistory = [...conversation.history]; // Deep copy to prevent reference issues
    
    
    // Clear and rebuild the UI
    clearMessagesArea();
    loadConversationUI();
    
    // Update conversation list to show active state
    updateConversationList();
    
    // Update chat title
    updateChatTitle(conversation.title);
    
    // Save current conversation ID to localStorage
    saveToLocalStorage(STORAGE_KEYS.CURRENT_ID, currentConversationId);
    
}

// Updated startNewChat function with proper cleanup
function startNewChat() {
    
    // Save current conversation if it exists and has content
    if (currentConversationId && conversationHistory.length > 0) {
        saveCurrentConversationToPersistentStorage();
    }
    
    // Reset conversation state
    conversationHistory = [];
    currentConversationId = null;
    
    // Clear the messages area and show welcome screen
    showWelcomeScreen();
    
    // Clear any follow-up questions
    clearFollowUpQuestions();
    
    // Update UI elements
    updateConversationList();
    updateChatTitle('UHG Meeting Document AI');
    
    // Save current state
    saveToLocalStorage(STORAGE_KEYS.CURRENT_ID, null);
    
}

// Updated clearChat function (for "Getting Started" button)
function clearChat() {
    
    // Same as starting new chat
    startNewChat();
}

// New helper function to update chat title
function updateChatTitle(title) {
    const chatTitleElement = document.getElementById('chat-title');
    if (chatTitleElement) {
        chatTitleElement.textContent = title || 'UHG Meeting Document AI';
    }
}

function deleteConversation(conversationId) {
    
    // Find the conversation to get its title for confirmation
    const conversation = savedConversations.find(c => c.id === conversationId);
    if (!conversation) {
        return;
    }
    
    // Show confirmation dialog
    const confirmMessage = `Are you sure you want to delete the conversation "${conversation.title}"?\n\nThis action cannot be undone.`;
    if (!confirm(confirmMessage)) {
        return;
    }
    
    try {
        // Remove from saved conversations array
        const conversationIndex = savedConversations.findIndex(c => c.id === conversationId);
        if (conversationIndex !== -1) {
            savedConversations.splice(conversationIndex, 1);
        }
        
        // Handle if we're deleting the currently active conversation
        if (currentConversationId === conversationId) {
            currentConversationId = null;
            conversationHistory = [];
            
            // Show welcome screen
            showWelcomeScreen();
            updateChatTitle('UHG Meeting Document AI');
        }
        
        // Update UI and save to localStorage
        updateConversationList();
        persistAllData();
        
        // Show success notification
        showNotification(`Conversation "${conversation.title}" deleted successfully`);
        
        
    } catch (error) {
        showNotification('Error deleting conversation. Please try again.');
    }
}

function clearAllConversations() {
    if (confirm('Are you sure you want to clear all conversation history? This action cannot be undone.')) {
        // Clear all data
        savedConversations = [];
        conversationHistory = [];
        currentConversationId = null;
        conversationCounter = 1;
        
        // Clear localStorage
        localStorage.removeItem(STORAGE_KEYS.CONVERSATIONS);
        localStorage.removeItem(STORAGE_KEYS.CURRENT_ID);
        localStorage.removeItem(STORAGE_KEYS.COUNTER);
        
        // Reset UI
        showWelcomeScreen();
        updateConversationList();
        
        showNotification('All conversations have been cleared.');
        
        // Close settings modal
        const modal = document.querySelector('.modal');
        if (modal) modal.remove();
    }
}

function exportConversations() {
    if (savedConversations.length === 0) {
        showNotification('No conversations to export.');
        return;
    }
    
    const exportData = {
        exportDate: new Date().toISOString(),
        conversationCount: savedConversations.length,
        conversations: savedConversations
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `uhg-meeting-ai-conversations-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    showNotification('Conversations exported successfully!');
}

async function initializeApp() {
    try {
        const response = await fetch('/api/test');
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                // Load stats only once during initialization
                await loadSystemStats(false); // Don't force refresh on startup
                
                // Only show welcome screen if no conversation is loaded
                if (conversationHistory.length === 0 && data.status.vector_size > 0) {
                    // Don't hide welcome screen automatically if user has documents but no conversation
                }
            }
        }
    } catch (error) {
    }
    
    // Initialize conversation list
    updateConversationList();
}

// Enhanced regenerate function that maintains conversation context
function regenerateResponse() {
    if (conversationHistory.length >= 2) {
        const lastUserMessage = conversationHistory[conversationHistory.length - 2];
        
        
        // Remove last assistant response from history
        conversationHistory.pop();
        
        // Remove last message from UI
        const messages = document.querySelectorAll('.message.assistant');
        if (messages.length > 0) {
            messages[messages.length - 1].remove();
        }
        
        // Resend the message
        showTypingIndicator();
        
        setTimeout(async () => {
            try {
                // Parse the message again to get filtering parameters for retry
                const { cleanMessage, documentIds, projectIds, meetingIds, dateFilters, folderPath } = parseMessageForDocuments(lastUserMessage.content);
                
                // Build the same request body as the original send
                const retryRequestBody = { message: cleanMessage };
                if (documentIds) {
                    retryRequestBody.document_ids = documentIds;
                }
                if (projectIds) {
                    retryRequestBody.project_ids = projectIds;
                }
                if (meetingIds) {
                    retryRequestBody.meeting_ids = meetingIds;
                }
                if (dateFilters) {
                    retryRequestBody.date_filters = dateFilters;
                }
                if (folderPath) {
                    retryRequestBody.folder_path = folderPath;
                }
                
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(retryRequestBody)
                });

                hideTypingIndicator();

                if (response.ok) {
                    const data = await response.json();
                    if (data.success) {
                        addMessageToUI('assistant', data.response, true);
                        conversationHistory.push({
                            role: 'assistant', 
                            content: data.response, 
                            timestamp: new Date().toISOString()
                        });
                        saveCurrentConversationToPersistentStorage();
                    }
                }
            } catch (error) {
                hideTypingIndicator();
                addMessageToUI('assistant', 'Sorry, I encountered an error while regenerating the response.', true);
            }
        }, 1000);
    }
}

function showUploadModal() {
    const modal = document.getElementById('upload-modal');
    if (modal) {
        loadProjects(); // Load projects when modal is shown
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        // Don't clear files automatically - let user decide
        updateUploadedFilesList();
    }
}

function removeUploadedFile(index) {
    if (index >= 0 && index < uploadedFiles.length) {
        const fileName = uploadedFiles[index].name;
        uploadedFiles.splice(index, 1);
        updateUploadedFilesList();
        showNotification(`Removed ${fileName}`);
    }
}

function hideUploadModal() {
    document.getElementById('upload-modal').classList.remove('active');
    document.body.style.overflow = 'auto';
    uploadedFiles = [];
    updateUploadedFilesList();
}



function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    addFilesToUpload(files);
}

function handleDragOver(event) {
    event.preventDefault();
    document.getElementById('upload-area').classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    document.getElementById('upload-area').classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    document.getElementById('upload-area').classList.remove('dragover');
    const files = Array.from(event.dataTransfer.files);
    addFilesToUpload(files);
}

function addFilesToUpload(files) {
    const validExtensions = ['.docx', '.txt', '.pdf'];
    const maxSize = 50 * 1024 * 1024; // 50MB
    let addedCount = 0;
    let errorCount = 0;

    files.forEach(file => {
        const extension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validExtensions.includes(extension)) {
            showNotification(`❌ ${file.name}: Unsupported format. Supported: .docx, .txt, .pdf`);
            errorCount++;
            return;
        }

        if (file.size > maxSize) {
            showNotification(`❌ ${file.name}: File too large (max 50MB)`);
            errorCount++;
            return;
        }

        if (file.size === 0) {
            showNotification(`❌ ${file.name}: File is empty`);
            errorCount++;
            return;
        }

        // Check for duplicates
        if (uploadedFiles.find(f => f.name === file.name)) {
            showNotification(`${file.name}: File already added`, 'warning');
            return;
        }

        // Add valid file
        uploadedFiles.push({
            file: file,
            name: file.name,
            size: formatFileSize(file.size),
            status: 'ready'
        });
        addedCount++;
    });

    updateUploadedFilesList();

    // Show summary notification
    if (addedCount > 0) {
        showNotification(`✅ Added ${addedCount} file${addedCount > 1 ? 's' : ''} for processing`);
    }
    
    if (errorCount > 0) {
    }
}

function clearUploadedFiles() {
    uploadedFiles = [];
    updateUploadedFilesList();
    showNotification('File list cleared');
}

function updateUploadedFilesList() {
    const container = document.getElementById('uploaded-files');
    const processBtn = document.getElementById('process-btn');
    
    if (uploadedFiles.length === 0) {
        container.innerHTML = '';
        processBtn.disabled = true;
        return;
    }

    // Enable process button only if there are files ready to process
    const readyFiles = uploadedFiles.filter(f => f.status === 'ready');
    processBtn.disabled = readyFiles.length === 0;
    
    container.innerHTML = uploadedFiles.map(fileObj => `
        <div class="file-item">
            <div class="file-info">
                <div class="file-icon">${getFileIcon(fileObj.name)}</div>
                <div class="file-details">
                    <div class="file-name">${fileObj.name}</div>
                    <div class="file-size">${fileObj.size}</div>
                    ${fileObj.error ? `<div class="file-error" style="color: #FF612B; font-size: 11px; margin-top: 2px;">${fileObj.error}</div>` : ''}
                </div>
            </div>
            <div class="file-status ${fileObj.status}">${getStatusText(fileObj.status)}</div>
        </div>
    `).join('');
}

function getFileIcon(filename) {
    const extension = filename.split('.').pop().toLowerCase();
    const icons = {
        'docx': '📄',
        'txt': '📝',
        'pdf': '📋'
    };
    return icons[extension] || '📄';
}

function getStatusText(status) {
    const texts = {
        'ready': '📋 Ready',
        'processing': '⏳ Processing...',
        'success': '✅ Processed',
        'error': '❌ Error'
    };
    return texts[status] || status;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function processFiles() {
    if (uploadedFiles.length === 0) return;

    const processBtn = document.getElementById('process-btn');
    processBtn.disabled = true;
    processBtn.innerHTML = 'Processing...';

    try {
        // Set all files to processing status
        uploadedFiles.forEach(fileObj => fileObj.status = 'processing');
        updateUploadedFilesList();

        // Create FormData for upload
        const formData = new FormData();
        uploadedFiles.forEach(fileObj => {
            formData.append('files', fileObj.file);
        });
        
        // Add selected project if any
        const projectSelect = document.getElementById('project-select');
        if (projectSelect && projectSelect.value) {
            formData.append('project_id', projectSelect.value);
        }

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            
            let successCount = 0;

            if (result.success && result.results && Array.isArray(result.results)) {
                // Process individual file results
                result.results.forEach(fileResult => {
                    const fileObj = uploadedFiles.find(f => f.name === fileResult.filename);
                    
                    if (fileObj) {
                        if (fileResult.success) {
                            fileObj.status = 'success';
                            successCount++;
                        } else {
                            fileObj.status = 'error';
                            fileObj.error = fileResult.error || 'Processing failed';
                        }
                    } else {
                        // Try to find by partial match
                        const partialMatch = uploadedFiles.find(f => 
                            f.name.includes(fileResult.filename) || fileResult.filename.includes(f.name)
                        );
                        if (partialMatch && fileResult.success) {
                            partialMatch.status = 'success';
                            successCount++;
                        }
                    }
                });
            } else if (result.success) {
                // Fallback: if no individual results but overall success
                uploadedFiles.forEach(fileObj => {
                    fileObj.status = 'success';
                    successCount++;
                });
            }
            
            // Additional fallback: Use processed count from backend
            if (successCount === 0 && result.success && result.processed > 0) {
                uploadedFiles.slice(0, result.processed).forEach(fileObj => {
                    fileObj.status = 'success';
                    successCount++;
                });
            }
            
            if (!result.success) {
                // Overall failure
                uploadedFiles.forEach(fileObj => {
                    fileObj.status = 'error';
                    fileObj.error = result.error || 'Upload failed';
                });
            }

            // Update UI with final status
            updateUploadedFilesList();

            // Refresh stats if any files were processed successfully
            if (successCount > 0) {
                await loadSystemStats(true); // Force refresh after upload
            }

            // Show completion message
            setTimeout(() => {
                if (successCount > 0) {
                    const message = `Successfully processed ${successCount} of ${uploadedFiles.length} documents! Documents are now available for querying.`;
                    showNotification(message);
                } else {
                    const message = `Failed to process any documents. Please check the files and try again.`;
                    showNotification(message, 'error');
                }
                
                // Only close modal if at least one file was successful
                if (successCount > 0) {
                    hideUploadModal();
                }
            }, 500);

        } else {
            // HTTP error response
            const errorText = await response.text();
            throw new Error(`Upload failed with status ${response.status}: ${errorText}`);
        }

    } catch (error) {
        
        // Set all files to error status
        uploadedFiles.forEach(fileObj => {
            fileObj.status = 'error';
            fileObj.error = error.message || 'Network error';
        });
        updateUploadedFilesList();
        
        showNotification('Upload failed. Please check your connection and try again.', 'error');
        
    } finally {
        // Reset button state
        processBtn.disabled = false;
        processBtn.innerHTML = 'Process Files';
    }
}

let statsCache = null;
let lastStatsUpdate = null;
const STATS_CACHE_DURATION = 5 * 60 * 1000; // 5 minutes cache

async function loadSystemStats(forceRefresh = false) {
    try {
        // Check if we have cached stats and they're still valid
        if (!forceRefresh && statsCache && lastStatsUpdate) {
            const timeSinceLastUpdate = Date.now() - lastStatsUpdate;
            if (timeSinceLastUpdate < STATS_CACHE_DURATION) {
                return statsCache;
            }
        }

        const response = await fetch('/api/stats');
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.stats) {
                // Cache the stats
                statsCache = data.stats;
                lastStatsUpdate = Date.now();
                return data.stats;
            }
        }
        return null;
    } catch (error) {
        return null;
    }
}

async function refreshSystem() {
    try {
        const response = await fetch('/api/refresh', { method: 'POST' });
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                // Clear stats cache when system is refreshed
                statsCache = null;
                lastStatsUpdate = null;
                
                showNotification('System refreshed successfully!');
                
                // Load fresh stats after refresh
                await loadSystemStats(true);
            } else {
                showNotification('Refresh failed: ' + (data.error || 'Unknown error'));
            }
        } else {
            showNotification('Refresh failed. Please try again.');
        }
    } catch (error) {
        showNotification('Refresh failed. Please check your connection.');
    }
}


function showStats() {
    const modal = document.createElement('div');
    modal.className = 'modal active';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">System Statistics</div>
                <button class="close-btn" onclick="this.closest('.modal').remove()">×</button>
            </div>
            <div id="stats-content">
                <div style="text-align: center; padding: 40px;">
                    <div style="font-size: 48px; margin-bottom: 16px; color: #FF612B;">📊</div>
                    <div style="color: #4B4D4F;">Loading statistics...</div>
                </div>
            </div>
            <div style="margin-top: 16px; text-align: center;">
                <button class="btn btn-secondary" onclick="refreshStatsModal()" style="margin-right: 8px;">
                    🔄 Refresh Stats
                </button>
                <button class="btn btn-secondary" onclick="clearStatsCache()">
                    🗑️ Clear Cache
                </button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    loadDetailedStats();
}

async function refreshStatsModal() {
    const container = document.getElementById('stats-content');
    if (container) {
        container.innerHTML = `
            <div style="text-align: center; padding: 40px;">
                <div style="font-size: 48px; margin-bottom: 16px; color: #FF612B;">🔄</div>
                <div style="color: #4B4D4F;">Refreshing statistics...</div>
            </div>
        `;
    }
    await loadDetailedStats(true); // Force refresh
}


function clearStatsCache() {
    statsCache = null;
    lastStatsUpdate = null;
    showNotification('Stats cache cleared');
}

async function loadDetailedStats(forceRefresh = false) {
    try {
        const stats = await loadSystemStats(forceRefresh);
        
        if (stats) {
            displayDetailedStats(stats);
        } else {
            displayStatsError('No data available');
        }
    } catch (error) {
        displayStatsError('Connection error');
    }
}


function displayDetailedStats(stats) {
    const container = document.getElementById('stats-content');
    if (!container) return;

    container.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 24px;">
            <div class="stats-card">
                <div class="stats-title">Documents</div>
                <div style="font-size: 32px; font-weight: 700; color: #002677; margin: 8px 0;">
                    ${stats.total_meetings || 0}
                </div>
                <div style="font-size: 12px; color: #4B4D4F;">Total meetings processed</div>
            </div>
            <div class="stats-card">
                <div class="stats-title">Chunks</div>
                <div style="font-size: 32px; font-weight: 700; color: #FF612B; margin: 8px 0;">
                    ${stats.total_chunks || 0}
                </div>
                <div style="font-size: 12px; color: #4B4D4F;">Text chunks for search</div>
            </div>
            <div class="stats-card">
                <div class="stats-title">Vector Index</div>
                <div style="font-size: 32px; font-weight: 700; color: #002677; margin: 8px 0;">
                    ${stats.vector_index_size || 0}
                </div>
                <div style="font-size: 12px; color: #4B4D4F;">Embedded vectors</div>
            </div>
            <div class="stats-card">
                <div class="stats-title">Average Chunk Size</div>
                <div style="font-size: 32px; font-weight: 700; color: #FF612B; margin: 8px 0;">
                    ${stats.average_chunk_length || 0}
                </div>
                <div style="font-size: 12px; color: #4B4D4F;">Characters per chunk</div>
            </div>
        </div>

        ${stats.date_range ? `
        <div class="stats-card" style="margin-bottom: 20px;">
            <div class="stats-title">Date Range</div>
            <div style="display: flex; justify-content: space-between; margin-top: 12px;">
                <div>
                    <div style="font-size: 12px; color: #4B4D4F;">Earliest</div>
                    <div style="font-weight: 600; color: #002677;">${stats.date_range.earliest || 'N/A'}</div>
                </div>
                <div>
                    <div style="font-size: 12px; color: #4B4D4F;">Latest</div>
                    <div style="font-weight: 600; color: #002677;">${stats.date_range.latest || 'N/A'}</div>
                </div>
            </div>
        </div>
        ` : ''}
    `;
}

function displayStatsError(error) {
    const container = document.getElementById('stats-content');
    if (!container) return;

    container.innerHTML = `
        <div style="text-align: center; padding: 40px;">
            <div>
                <div style="font-size: 48px; margin-bottom: 16px; color: #FF612B;">❌</div>
                <div style="color: #FF612B; font-weight: 600; margin-bottom: 8px;">Error Loading Statistics</div>
                <div style="color: #4B4D4F; font-size: 14px;">${error}</div>
            </div>
        </div>
    `;
}

function showHelp() {
    const modal = document.createElement('div');
    modal.className = 'modal active';
    modal.innerHTML = `
        <div>
            <div class="modal-content">
                <div class="modal-header">
                    <div class="modal-title">Help & Examples</div>
                    <button class="close-btn" onclick="this.closest('.modal').remove()">×</button>
                </div>
                <div style="line-height: 1.6;">
                    <h3 style="color: #002677; margin-bottom: 16px;">Getting Started</h3>
                    <ol style="margin-bottom: 24px; padding-left: 20px; color: #4B4D4F;">
                        <li style="margin-bottom: 8px;">Upload your meeting documents (.docx, .txt, .pdf)</li>
                        <li style="margin-bottom: 8px;">Wait for processing to complete</li>
                        <li style="margin-bottom: 8px;">Start asking questions about your meetings</li>
                    </ol>

                    <h3 style="color: #002677; margin-bottom: 16px;">Example Questions</h3>
                    <div style="display: grid; gap: 12px; margin-bottom: 24px;">
                        <div style="padding: 16px; background: #FAF8F2; border: 1px solid #D9F6FA; border-radius: 8px; cursor: pointer;" onclick="insertSampleQuery('What are the main topics from recent meetings?'); this.closest('.modal').remove();">
                            <strong style="color: #002677;">📋 Topic Analysis</strong><br />
                            <em style="color: #4B4D4F;">"What are the main topics from recent meetings?"</em>
                        </div>
                        <div style="padding: 16px; background: #FAF8F2; border: 1px solid #D9F6FA; border-radius: 8px; cursor: pointer;" onclick="insertSampleQuery('What action items were discussed last week?'); this.closest('.modal').remove();">
                            <strong style="color: #002677;">✅ Action Items</strong><br />
                            <em style="color: #4B4D4F;">"What action items were discussed last week?"</em>
                        </div>
                        <div style="padding: 16px; background: #FAF8F2; border: 1px solid #D9F6FA; border-radius: 8px; cursor: pointer;" onclick="insertSampleQuery('Who are the key participants in our meetings?'); this.closest('.modal').remove();">
                            <strong style="color: #002677;">👥 Participants</strong><br />
                            <em style="color: #4B4D4F;">"Who are the key participants in our meetings?"</em>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

function showStats() {
    const modal = document.createElement('div');
    modal.className = 'modal active';
    modal.innerHTML = `
        <div>
            <div class="modal-content">
                <div class="modal-header">
                    <div class="modal-title">System Statistics</div>
                    <button class="close-btn" onclick="this.closest('.modal').remove()">×</button>
                </div>
                <div id="stats-content" style="line-height: 1.6; text-align: center; padding: 20px;">
                    <div style="color: #4B4D4F;">Loading statistics...</div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    
    // Load stats
    loadSystemStats(true).then(stats => {
        const content = document.getElementById('stats-content');
        if (stats) {
            content.innerHTML = `
                <div style="display: grid; gap: 16px; text-align: left;">
                    <div style="padding: 16px; background: #F8F9FF; border-radius: 8px;">
                        <h3 style="color: #002677; margin: 0 0 12px 0;">📊 Document Statistics</h3>
                        <div style="color: #4B4D4F; display: grid; gap: 8px;">
                            <div><strong>Total Documents:</strong> ${stats.total_meetings || 0}</div>
                            <div><strong>Total Chunks:</strong> ${stats.total_chunks || 0}</div>
                            <div><strong>Vector Index Size:</strong> ${stats.vector_index_size || 0}</div>
                        </div>
                    </div>
                    <div style="padding: 16px; background: #F8F9FF; border-radius: 8px;">
                        <h3 style="color: #002677; margin: 0 0 12px 0;">📅 Date Range</h3>
                        <div style="color: #4B4D4F; display: grid; gap: 8px;">
                            <div><strong>Earliest:</strong> ${stats.date_range?.earliest || 'N/A'}</div>
                            <div><strong>Latest:</strong> ${stats.date_range?.latest || 'N/A'}</div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            content.innerHTML = `
                <div style="color: #FF612B;">
                    <div style="font-weight: 600; margin-bottom: 8px;">Unable to load statistics</div>
                    <div style="font-size: 14px;">Please try again later</div>
                </div>
            `;
        }
    });
}

function insertSampleQuery(query) {
    document.getElementById('message-input').value = query;
    autoResize();
}

function showSettings() {
    const cacheInfo = statsCache && lastStatsUpdate ? 
        `Last updated: ${new Date(lastStatsUpdate).toLocaleString()}` : 
        'No cached stats';

    const modal = document.createElement('div');
    modal.className = 'modal active';
    modal.innerHTML = `
        <div>
            <div class="modal-content">
                <div class="modal-header">
                    <div class="modal-title">Settings</div>
                    <button class="close-btn" onclick="this.closest('.modal').remove()">×</button>
                </div>
                <div style="line-height: 1.6;">
                    <h3 style="color: #002677; margin-bottom: 16px;">Chat History</h3>
                    <div style="margin-bottom: 24px;">
                        <button class="btn btn-secondary" onclick="clearAllConversations()" style="margin-right: 12px;">
                            🗑️ Clear All Conversations
                        </button>
                        <button class="btn btn-secondary" onclick="exportConversations()">
                            📥 Export Conversations
                        </button>
                    </div>

                    <h3 style="color: #002677; margin-bottom: 16px;">System Performance</h3>
                    <div style="margin-bottom: 24px;">
                        <button class="btn btn-secondary" onclick="clearStatsCache(); this.closest('.modal').remove();" style="margin-right: 12px;">
                            🗑️ Clear Stats Cache
                        </button>
                        <button class="btn btn-secondary" onclick="loadSystemStats(true).then(() => showNotification('Stats refreshed'))">
                            🔄 Refresh Stats Now
                        </button>
                    </div>

                    <h3 style="color: #002677; margin-bottom: 16px;">Storage Information</h3>
                    <div style="background: #FAF8F2; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                        <div style="font-size: 14px; color: #4B4D4F;">
                            <strong>Conversations saved:</strong> ${savedConversations.length}<br>
                            <strong>Current conversation:</strong> ${currentConversationId ? 'Active' : 'None'}<br>
                            <strong>Stats cache:</strong> ${cacheInfo}<br>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

function getStatsCacheInfo() {
    if (!statsCache || !lastStatsUpdate) {
        return 'No cached stats available';
    }
    
    const age = Date.now() - lastStatsUpdate;
    const minutes = Math.floor(age / 60000);
    const seconds = Math.floor((age % 60000) / 1000);
    
    return `Cache age: ${minutes}m ${seconds}s (expires in ${Math.max(0, Math.floor((STATS_CACHE_DURATION - age) / 60000))}m)`;
}

// Enhanced auto-save functionality
function setupAutoSave() {
    // Remove this line: setInterval(loadSystemStats, 30000);
    
    // Only keep auto-save conversations every 2 minutes
    setInterval(() => {
        if (conversationHistory.length > 0 && currentConversationId) {
            saveCurrentConversationToPersistentStorage();
        }
    }, 120000); // 2 minutes
}

// Sidebar Toggle Functionality
let sidebarOpen = false;

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const container = document.querySelector('.container');
    const backdrop = document.getElementById('mobile-backdrop');
    const toggleIcon = document.getElementById('sidebar-toggle-icon');
    const toggleBtn = document.getElementById('sidebar-toggle');
    
    if (window.innerWidth <= 768) {
        // Mobile behavior
        if (sidebarOpen) {
            closeMobileSidebar();
        } else {
            openMobileSidebar();
        }
    } else {
        // Desktop behavior
        if (container.classList.contains('sidebar-collapsed')) {
            // Show sidebar
            sidebar.classList.remove('collapsed');
            container.classList.remove('sidebar-collapsed');
            toggleIcon.textContent = '☰';
            toggleBtn.classList.remove('active');
        } else {
            // Hide sidebar
            sidebar.classList.add('collapsed');
            container.classList.add('sidebar-collapsed');
            toggleIcon.textContent = '≫';
            toggleBtn.classList.add('active');
        }
    }
}

function openMobileSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const backdrop = document.getElementById('mobile-backdrop');
    const toggleIcon = document.getElementById('sidebar-toggle-icon');
    const toggleBtn = document.getElementById('sidebar-toggle');
    
    sidebar.classList.add('mobile-open');
    backdrop.classList.add('active');
    toggleIcon.textContent = '✕';
    toggleBtn.classList.add('active');
    sidebarOpen = true;
    
    // Prevent body scroll when sidebar is open
    document.body.style.overflow = 'hidden';
}

function closeMobileSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const backdrop = document.getElementById('mobile-backdrop');
    const toggleIcon = document.getElementById('sidebar-toggle-icon');
    const toggleBtn = document.getElementById('sidebar-toggle');
    
    sidebar.classList.remove('mobile-open');
    backdrop.classList.remove('active');
    toggleIcon.textContent = '☰';
    toggleBtn.classList.remove('active');
    sidebarOpen = false;
    
    // Restore body scroll
    document.body.style.overflow = '';
}

// Handle window resize
function handleWindowResize() {
    const container = document.querySelector('.container');
    const sidebar = document.querySelector('.sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle');
    
    if (window.innerWidth > 768) {
        // Desktop: close mobile sidebar if open
        closeMobileSidebar();
        
        // Reset container class on desktop resize
        if (sidebar.classList.contains('collapsed')) {
            container.classList.add('sidebar-collapsed');
        } else {
            container.classList.remove('sidebar-collapsed');
        }
        
        // Show sidebar toggle button on desktop too for collapsible sidebar
        if (toggleBtn) {
            toggleBtn.style.display = 'flex';
        }
    } else {
        // Mobile: remove desktop collapsed classes
        container.classList.remove('sidebar-collapsed');
        
        // Mobile: ensure toggle button is visible
        if (toggleBtn) {
            toggleBtn.style.display = 'flex';
        }
    }
}

// Mobile viewport height fix
function setMobileViewportHeight() {
    // Fix mobile viewport height issues
    const vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
}

// Initialize mobile fixes
function initializeMobileFixes() {
    // Set initial viewport height
    setMobileViewportHeight();
    
    // Handle window resize
    window.addEventListener('resize', () => {
        handleWindowResize();
        setMobileViewportHeight();
    });
    
    // Handle orientation change on mobile
    window.addEventListener('orientationchange', () => {
        setTimeout(() => {
            setMobileViewportHeight();
            // Scroll to bottom if chat is active
            const messagesArea = document.getElementById('messages-area');
            if (messagesArea && conversationHistory.length > 0) {
                messagesArea.scrollTop = messagesArea.scrollHeight;
            }
        }, 100);
    });
    
    // Close sidebar when clicking on a conversation item on mobile
    const originalLoadConversation = window.loadConversation;
    window.loadConversation = function(conversationId) {
        originalLoadConversation(conversationId);
        if (window.innerWidth <= 768) {
            closeMobileSidebar();
        }
    };
    
    // Close sidebar when starting new chat on mobile
    const originalStartNewChat = window.startNewChat;
    window.startNewChat = function() {
        originalStartNewChat();
        if (window.innerWidth <= 768) {
            closeMobileSidebar();
        }
    };
    
    // Initial setup
    handleWindowResize();
}

// Debugging functions (can be removed in production)
function debugConversationState() {
}

// Make debug function globally available for testing
window.debugConversationState = debugConversationState;

window.statsManagement = {
    loadStats: loadSystemStats,
    clearCache: clearStatsCache,
    getCacheInfo: getStatsCacheInfo,
    forceRefresh: () => loadSystemStats(true)
};

// Conversation Menu Functions
let currentMenuConversationId = null;

function showConversationMenu(event, conversationId) {
    event.stopPropagation();
    
    const dropdown = document.getElementById('conversation-dropdown');
    const button = event.target.closest('.conversation-menu-btn');
    
    if (!dropdown || !button) return;
    
    // Store current conversation ID for menu actions
    currentMenuConversationId = conversationId;
    
    // Position dropdown relative to button
    const buttonRect = button.getBoundingClientRect();
    dropdown.style.position = 'fixed';
    dropdown.style.top = `${buttonRect.bottom + 5}px`;
    dropdown.style.left = `${buttonRect.right - 140}px`; // Align right edge of dropdown with right edge of button
    dropdown.style.display = 'block';
    dropdown.style.zIndex = '1000';
    
    // Add click outside to close functionality
    setTimeout(() => {
        document.addEventListener('click', closeConversationMenu);
    }, 10);
}

function closeConversationMenu() {
    const dropdown = document.getElementById('conversation-dropdown');
    if (dropdown) {
        dropdown.style.display = 'none';
    }
    currentMenuConversationId = null;
    document.removeEventListener('click', closeConversationMenu);
}

function showEditModal() {
    
    // Store the ID before closing menu (which clears currentMenuConversationId)
    const conversationId = currentMenuConversationId;
    closeConversationMenu();
    
    if (!conversationId) {
        showNotification('No conversation selected');
        return;
    }
    
    const conversation = savedConversations.find(c => c.id === conversationId);
    if (!conversation) {
        showNotification('Conversation not found');
        return;
    }
    
    // Set the ID back for the edit operation
    currentMenuConversationId = conversationId;
    
    const modal = document.getElementById('edit-modal');
    const input = document.getElementById('edit-input');
    
    if (!modal || !input) {
        return;
    }
    
    // Set current title in input
    input.value = conversation.title;
    input.focus();
    input.select();
    
    // Show modal
    modal.classList.add('active');
    
    // Handle Enter key in input
    input.onkeydown = function(e) {
        if (e.key === 'Enter') {
            confirmEdit();
        } else if (e.key === 'Escape') {
            closeEditModal();
        }
    };
}

function closeEditModal() {
    const modal = document.getElementById('edit-modal');
    const input = document.getElementById('edit-input');
    
    if (modal) modal.classList.remove('active');
    if (input) {
        input.value = '';
        input.onkeydown = null;
    }
}

function confirmEdit() {
    const input = document.getElementById('edit-input');
    const newTitle = input.value.trim();
    
    if (!newTitle) {
        showNotification('Please enter a conversation name');
        input.focus();
        return;
    }
    
    if (!currentMenuConversationId) {
        showNotification('No conversation selected');
        closeEditModal();
        return;
    }
    
    // Find and update conversation
    const conversation = savedConversations.find(c => c.id === currentMenuConversationId);
    if (!conversation) {
        showNotification('Conversation not found');
        closeEditModal();
        return;
    }
    
    const oldTitle = conversation.title;
    conversation.title = newTitle;
    conversation.updatedAt = new Date().toISOString();
    
    // Update UI and save
    updateConversationList();
    persistAllData();
    
    // Update chat title if this is the current conversation
    if (currentConversationId === currentMenuConversationId) {
        updateChatTitle(newTitle);
    }
    
    closeEditModal();
    showNotification(`Conversation renamed from "${oldTitle}" to "${newTitle}"`);
}

function showDeleteModal() {
    
    // Store the ID before closing menu (which clears currentMenuConversationId)
    const conversationId = currentMenuConversationId;
    closeConversationMenu();
    
    if (!conversationId) {
        showNotification('No conversation selected');
        return;
    }
    
    const conversation = savedConversations.find(c => c.id === conversationId);
    if (!conversation) {
        showNotification('Conversation not found');
        return;
    }
    
    // Set the ID back for the delete operation
    currentMenuConversationId = conversationId;
    
    const modal = document.getElementById('delete-modal');
    const message = document.getElementById('delete-message');
    
    if (!modal || !message) {
        return;
    }
    
    // Update message with conversation title
    message.innerHTML = `Are you sure you want to delete the conversation "<strong>${conversation.title}</strong>"?<br><br>This action cannot be undone.`;
    
    // Show modal
    modal.classList.add('active');
}

function closeDeleteModal() {
    const modal = document.getElementById('delete-modal');
    if (modal) modal.classList.remove('active');
}

function confirmDelete() {
    if (!currentMenuConversationId) {
        showNotification('No conversation selected');
        closeDeleteModal();
        return;
    }
    
    const conversation = savedConversations.find(c => c.id === currentMenuConversationId);
    if (!conversation) {
        showNotification('Conversation not found');
        closeDeleteModal();
        return;
    }
    
    try {
        // Remove from saved conversations array
        const conversationIndex = savedConversations.findIndex(c => c.id === currentMenuConversationId);
        if (conversationIndex !== -1) {
            savedConversations.splice(conversationIndex, 1);
        }
        
        // Handle if we're deleting the currently active conversation
        if (currentConversationId === currentMenuConversationId) {
            currentConversationId = null;
            conversationHistory = [];
            
            // Show welcome screen
            showWelcomeScreen();
            updateChatTitle('UHG Meeting Document AI');
        }
        
        // Update UI and save to localStorage
        updateConversationList();
        persistAllData();
        
        closeDeleteModal();
        showNotification(`Conversation "${conversation.title}" deleted successfully`);
        
    } catch (error) {
        closeDeleteModal();
        showNotification('Error deleting conversation. Please try again.');
    }
}

// Authentication Functions
async function checkAuthenticationStatus() {
    try {
        const response = await fetch('/api/auth/status');
        if (response.ok) {
            const data = await response.json();
            if (data.authenticated) {
                displayUserInfo(data.user);
                return true;
            }
        }
        // If not authenticated, redirect to login
        window.location.href = '/login';
        return false;
    } catch (error) {
        window.location.href = '/login';
        return false;
    }
}

function displayUserInfo(user) {
    const userInfo = document.getElementById('user-info');
    const userName = document.getElementById('user-name');
    const userEmail = document.getElementById('user-email');
    const logoutBtn = document.getElementById('logout-btn');
    
    if (userInfo && userName && userEmail) {
        userName.textContent = user.full_name || user.username;
        userEmail.textContent = user.email;
        userInfo.style.display = 'flex';
        
        if (logoutBtn) {
            logoutBtn.style.display = 'block';
        }
    }
}

async function logout() {
    try {
        const response = await fetch('/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (response.ok) {
            // Clear local storage
            localStorage.clear();
            
            // Redirect to login
            window.location.href = '/login';
        } else {
            showNotification('Logout failed. Please try again.');
        }
    } catch (error) {
        showNotification('Logout failed. Please try again.');
    }
}

// Global project management variables  
let selectedProjectId = null;

// Project Management Functions
async function loadProjects() {
    try {
        const response = await fetch('/api/projects');
        const data = await response.json();
        
        if (data.success) {
            availableProjects = data.projects;
            updateProjectSelect();
        } else {
        }
    } catch (error) {
    }
}

function updateProjectSelect() {
    const projectSelect = document.getElementById('project-select');
    if (!projectSelect) return;
    
    projectSelect.innerHTML = '';
    
    if (availableProjects.length === 0) {
        projectSelect.innerHTML = '<option value="">No projects available</option>';
        return;
    }
    
    // Add default option
    projectSelect.innerHTML = '<option value="">Select a project...</option>';
    
    // Add project options
    availableProjects.forEach(project => {
        const option = document.createElement('option');
        option.value = project.project_id;
        option.textContent = project.project_name;
        if (project.description) {
            option.title = project.description;
        }
        projectSelect.appendChild(option);
    });
    
    // Set selected project if any
    if (selectedProjectId) {
        projectSelect.value = selectedProjectId;
    }
}

function showCreateProjectModal() {
    const modal = document.getElementById('create-project-modal');
    const nameInput = document.getElementById('project-name-input');
    const descInput = document.getElementById('project-description-input');
    
    if (modal) {
        // Clear inputs
        nameInput.value = '';
        descInput.value = '';
        
        modal.classList.add('active');
        nameInput.focus();
    }
}

function closeCreateProjectModal() {
    const modal = document.getElementById('create-project-modal');
    if (modal) {
        modal.classList.remove('active');
    }
}

async function confirmCreateProject() {
    const nameInput = document.getElementById('project-name-input');
    const descInput = document.getElementById('project-description-input');
    
    const projectName = nameInput.value.trim();
    const description = descInput.value.trim();
    
    if (!projectName) {
        showNotification('Project name is required');
        nameInput.focus();
        return;
    }
    
    try {
        const response = await fetch('/api/projects', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                project_name: projectName,
                description: description
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('Project created successfully');
            closeCreateProjectModal();
            
            // Reload projects and select the new one
            await loadProjects();
            selectedProjectId = data.project_id;
            updateProjectSelect();
        } else {
            showNotification('Failed to create project: ' + data.error);
        }
    } catch (error) {
        showNotification('Failed to create project');
    }
}

// Upload modal function is defined above with project loading

// Close modals when clicking outside
document.addEventListener('click', function(event) {
    // Close edit modal when clicking outside
    const editModal = document.getElementById('edit-modal');
    if (editModal && editModal.classList.contains('active')) {
        if (event.target === editModal) {
            closeEditModal();
        }
    }
    
    // Close delete modal when clicking outside
    const deleteModal = document.getElementById('delete-modal');
    if (deleteModal && deleteModal.classList.contains('active')) {
        if (event.target === deleteModal) {
            closeDeleteModal();
        }
    }
    
    // Close create project modal when clicking outside
    const createProjectModal = document.getElementById('create-project-modal');
    if (createProjectModal && createProjectModal.classList.contains('active')) {
        if (event.target === createProjectModal) {
            closeCreateProjectModal();
        }
    }
    
    // Close user profile modal when clicking outside
    const userProfileModal = document.getElementById('user-profile-modal');
    if (userProfileModal && userProfileModal.classList.contains('active')) {
        if (event.target === userProfileModal) {
            closeUserProfile();
        }
    }
});

// User Profile Functions
function showUserProfile() {
    const modal = document.getElementById('user-profile-modal');
    if (modal) {
        // Fetch and display current user data
        updateUserProfileDisplay();
        modal.classList.add('active');
    }
}

function closeUserProfile() {
    const modal = document.getElementById('user-profile-modal');
    if (modal) {
        modal.classList.remove('active');
    }
}

function updateUserProfileDisplay() {
    // Check if user is authenticated and get user data
    fetch('/api/auth/status', {
        method: 'GET',
        credentials: 'include'  // Include cookies for session
    })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                // If auth check fails, redirect to login
                throw new Error('Authentication check failed');
            }
        })
        .then(data => {
            if (data.authenticated && data.user) {
                const user = data.user;
                
                // Update sidebar user info
                updateElement('sidebar-user-name', user.full_name || 'UHG User');
                updateElement('sidebar-user-email', user.email || 'user@uhg.com');
                
                // Update profile modal
                updateElement('profile-full-name', user.full_name || 'UHG User');
                updateElement('profile-username', user.username || 'uhg_user');
                updateElement('profile-email', user.email || 'user@uhg.com');
                
                // Update avatars with initials
                const initials = getInitials(user.full_name || 'UHG User');
                updateElement('user-avatar', initials);
                updateElement('profile-avatar', initials);
                updateElement('avatar-initials', initials);
                
                // Update timestamps (placeholder - would need backend support for real data)
                updateElement('profile-created-at', 'January 15, 2025');
                
                // Load user statistics
                loadUserStats();
                
            } else {
                // Redirect to login if not authenticated
                window.location.href = '/login';
            }
        })
        .catch(error => {
            // On any error, redirect to login (could be session expired, server restart, etc.)
            window.location.href = '/login';
        });
}

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function getInitials(fullName) {
    if (!fullName) return 'UU';
    const names = fullName.trim().split(' ');
    if (names.length === 1) {
        return names[0].substring(0, 2).toUpperCase();
    }
    return (names[0][0] + names[names.length - 1][0]).toUpperCase();
}

function formatDateTime(date) {
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    if (hours < 24) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    if (days < 7) return `${days} day${days > 1 ? 's' : ''} ago`;
    
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function loadUserStats() {
    // Load documents count
    if (availableDocuments) {
        updateElement('stat-documents', availableDocuments.length.toString());
    }
    
    // Load conversations count
    if (savedConversations) {
        updateElement('stat-conversations', savedConversations.length.toString());
    }
    
    // Load projects count
    if (availableProjects) {
        updateElement('stat-projects', availableProjects.length.toString());
    }
}

function showDefaultUserInfo() {
    // Show default user information
    updateElement('sidebar-user-name', 'UHG User');
    updateElement('sidebar-user-email', 'user@uhg.com');
    updateElement('profile-full-name', 'UHG User');
    updateElement('profile-username', 'uhg_user');
    updateElement('profile-email', 'user@uhg.com');
    updateElement('profile-user-id', 'USER001');
    
    const initials = getInitials('UHG User');
    updateElement('user-avatar', initials);
    updateElement('profile-avatar', initials);
    updateElement('avatar-initials', initials);
}

function showAccountSettings() {
    // Placeholder for account settings functionality
    showNotification('Account settings coming soon!');
    closeUserProfile();
}

// Date-based query intelligence
function detectDateQuery(message) {
    const datePatterns = [
        // Current periods
        /\b(current|this)\s+(week|month|quarter|year)\b/i,
        // Last periods  
        /\b(last|past|previous)\s+(week|month|quarter|year)\b/i,
        // Specific day counts
        /\b(last|past)\s+(\d+)\s+(days?|weeks?|months?)\b/i,
        // Recent
        /\b(recent|recently|lately)\b/i,
        // Summary with timeframe
        /\b(summary|summarize|overview).*\b(week|month|quarter|year|recent)\b/i,
        // Timeframe with summary
        /\b(week|month|quarter|year|recent).*\b(summary|summarize|overview)\b/i
    ];
    
    return datePatterns.some(pattern => pattern.test(message));
}

function addDateSuggestions(message) {
    if (detectDateQuery(message)) {
        
        // You could add UI hints here, such as:
        // - Highlighting suggested date options
        // - Showing a date picker
        // - Displaying timeline of available documents
        
        // For now, we'll just log it for debugging
        showNotification('💡 Tip: You can use specific date ranges like "last week", "current month", or "last 3 months" for more precise results!', 'info');
    }
}

// Enhance the sendMessage function to detect date queries
const originalSendMessage = sendMessage;
sendMessage = function() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    // Add date query intelligence
    if (message) {
        addDateSuggestions(message);
    }
    
    // Call original function
    return originalSendMessage.apply(this, arguments);
};

// Initialize user profile display when the page loads
document.addEventListener('DOMContentLoaded', function() {
    updateUserProfileDisplay();
});