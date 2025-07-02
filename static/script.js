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
        
        console.log('Markdown parser initialized successfully');
        return true;
    } else {
        console.error('marked.js not loaded');
        return false;
    }
}

// Enhanced Markdown to HTML conversion with security
function formatMarkdownToHTML(text) {
    if (!text) return '';
    
    try {
        // Check if marked.js is available
        if (typeof marked === 'undefined') {
            console.warn('marked.js not available, falling back to basic parsing');
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
        console.error('Error parsing markdown:', error);
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
    console.log('Initializing UHG Meeting AI...');
    
    // Initialize markdown parser
    initializeMarkdownParser();
    
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
    
    // Setup @ mention detection
    setupAtMentionDetection();
    
});

// Enhanced event listeners setup
function setupEventListeners() {
    console.log('Setting up event listeners...');
    
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
        console.log('Page unloading, saving data...');
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
        console.error('Error saving to localStorage:', error);
    }
}

function loadFromLocalStorage(key, defaultValue = null) {
    try {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : defaultValue;
    } catch (error) {
        console.error('Error loading from localStorage:', error);
        return defaultValue;
    }
}

// Enhanced data loading with proper UI initialization
function loadPersistedDataAndInitializeUI() {
    console.log('Loading persisted data and initializing UI...');
    
    // Load saved conversations
    const savedData = loadFromLocalStorage(STORAGE_KEYS.CONVERSATIONS, []);
    savedConversations = savedData;
    console.log(`Loaded ${savedConversations.length} saved conversations`);

    // Load current conversation ID
    currentConversationId = loadFromLocalStorage(STORAGE_KEYS.CURRENT_ID, null);
    console.log(`Current conversation ID: ${currentConversationId}`);

    // Load conversation counter
    conversationCounter = loadFromLocalStorage(STORAGE_KEYS.COUNTER, 1);

    // Initialize UI based on current conversation state
    if (currentConversationId) {
        const currentConv = savedConversations.find(c => c.id === currentConversationId);
        if (currentConv) {
            console.log(`Loading existing conversation: ${currentConv.title}`);
            conversationHistory = [...currentConv.history];
            loadConversationUI();
            updateChatTitle(currentConv.title);
        } else {
            console.warn(`Current conversation ${currentConversationId} not found, resetting`);
            currentConversationId = null;
            conversationHistory = [];
            showWelcomeScreen();
            saveToLocalStorage(STORAGE_KEYS.CURRENT_ID, null);
        }
    } else {
        console.log('No current conversation, showing welcome screen');
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
    console.log('Loading conversation UI...');
    
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
        console.log(`Adding message ${index + 1}: ${msg.role}`);
        addMessageToUI(msg.role, msg.content, false); // false = don't update history
    });
    
    // Scroll to bottom after loading all messages
    setTimeout(() => {
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }, 100);
    
    console.log('Conversation UI loaded successfully');
}

// Enhanced showWelcomeScreen function
function showWelcomeScreen() {
    console.log('Showing welcome screen...');
    
    const messagesArea = document.getElementById('messages-area');
    messagesArea.innerHTML = `
        <div class="welcome-screen" id="welcome-screen">
            <div class="welcome-icon">🤖</div>
            <div class="welcome-title">Welcome to UHG Meeting AI</div>
            <div class="welcome-text">
                Upload your meeting documents and start asking questions. I can help you analyze meeting content, extract key insights, find action items, and track discussions across multiple documents using advanced AI.
            </div>
            
            <div class="sample-prompts">
                <div class="sample-prompts-title">Try these sample prompts:</div>
                <div class="sample-prompt-grid">
                    <button class="sample-prompt" onclick="insertSampleQuery('What are the main topics discussed in recent meetings?')">
                        What are the main topics discussed in recent meetings?
                    </button>
                    <button class="sample-prompt" onclick="insertSampleQuery('List all action items from last week\\'s meetings')">
                        List all action items from last week's meetings
                    </button>
                    <button class="sample-prompt" onclick="insertSampleQuery('Who are the key participants and their roles?')">
                        Who are the key participants and their roles?
                    </button>
                    <button class="sample-prompt" onclick="insertSampleQuery('Summarize decisions made in project meetings')">
                        Summarize decisions made in project meetings
                    </button>
                    <button class="sample-prompt" onclick="insertSampleQuery('What challenges or blockers were identified?')">
                        What challenges or blockers were identified?
                    </button>
                    <button class="sample-prompt" onclick="insertSampleQuery('Show me upcoming deadlines and milestones')">
                        Show me upcoming deadlines and milestones
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

// Updated sendMessage function to handle conversation context properly
async function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (!message || isProcessing) return;

    console.log('Sending message in conversation:', currentConversationId || 'new conversation');

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

    // Parse message for document selection
    const { cleanMessage, documentIds } = parseMessageForDocuments(message);

    // Add user message to UI and history
    addMessageToUI('user', message, true);
    conversationHistory.push({
        role: 'user', 
        content: message, 
        timestamp: new Date().toISOString()
    });
    
    input.value = '';
    selectedDocuments = []; // Clear selected documents after sending
    updateSelectedDocuments();
    autoResize();

    // Show typing indicator
    showTypingIndicator();

    try {
        const requestBody = { message: cleanMessage };
        if (documentIds) {
            requestBody.document_ids = documentIds;
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
                
                console.log('Message sent and conversation saved');
            } else {
                addMessageToUI('assistant', 'Sorry, I encountered an error: ' + (data.error || 'Unknown error'), true);
            }
        } else {
            addMessageToUI('assistant', 'Sorry, I\'m having trouble connecting to the server. Please try again.', true);
        }
    } catch (error) {
        hideTypingIndicator();
        console.error('Chat error:', error);
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

function showNotification(message) {
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 3000);
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
                console.log(`Loaded ${availableDocuments.length} documents for @ mention selection`);
            }
        }
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

function detectAtMention(input) {
    const text = input.value;
    const cursorPos = input.selectionStart;
    
    // Find @ symbols and check if cursor is after one
    let atPos = -1;
    for (let i = cursorPos - 1; i >= 0; i--) {
        if (text[i] === '@') {
            // Check if @ is at start or after whitespace
            if (i === 0 || /\s/.test(text[i - 1])) {
                atPos = i;
                break;
            }
        }
        if (/\s/.test(text[i])) {
            break; // Stop at whitespace
        }
    }
    
    if (atPos !== -1) {
        const searchText = text.substring(atPos + 1, cursorPos);
        return { isActive: true, searchText, atPos };
    }
    
    return { isActive: false, searchText: '', atPos: -1 };
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
        console.error('Document dropdown elements not found');
        return;
    }
    
    const filteredDocs = filterDocuments(searchText);
    
    if (filteredDocs.length === 0) {
        documentList.innerHTML = '<div style="padding: 16px; text-align: center; color: #6B7280;">No matching documents found</div>';
    } else {
        documentList.innerHTML = filteredDocs.map(doc => {
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
        
        // Add click listeners
        documentList.querySelectorAll('.document-item').forEach(item => {
            item.addEventListener('click', () => {
                const docId = item.dataset.docId;
                const doc = filteredDocs.find(d => d.document_id === docId);
                if (doc) {
                    selectDocument(doc);
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

function hideDocumentDropdown() {
    const dropdown = document.getElementById('document-dropdown');
    dropdown.classList.remove('active');
    isDocumentDropdownOpen = false;
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
    
    // Return clean message and document IDs
    return {
        cleanMessage: message.trim(),
        documentIds: documentIds.length > 0 ? documentIds : null
    };
}

function setupAtMentionDetection() {
    const input = document.getElementById('message-input');
    
    // Handle @ mention detection on input
    input.addEventListener('input', function(e) {
        const mention = detectAtMention(input);
        
        if (mention.isActive) {
            showDocumentDropdown(mention.searchText);
        } else {
            hideDocumentDropdown();
        }
    });
    
    // Handle keyboard navigation
    input.addEventListener('keydown', function(e) {
        if (isDocumentDropdownOpen) {
            if (e.key === 'Escape') {
                e.preventDefault();
                hideDocumentDropdown();
            } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                navigateDocumentDropdown(e.key === 'ArrowDown' ? 1 : -1);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                selectHighlightedDocument();
            }
        }
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (isDocumentDropdownOpen && !e.target.closest('.input-area')) {
            hideDocumentDropdown();
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

function selectHighlightedDocument() {
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
            console.log(`Clicking on conversation: ${conv.id}`);
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
        console.log('No conversation history to save');
        return;
    }
    
    console.log('Saving current conversation...');
    
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
                console.log(`Updated existing conversation: ${title}`);
            } else {
                console.error(`Conversation ${currentConversationId} not found for update`);
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
            console.log(`Created new conversation: ${title}`);
        }
        
        updateConversationList();
        updateChatTitle(title);
        
    } catch (error) {
        console.error('Error saving conversation:', error);
    }
}

// Updated loadConversation function with proper state management
function loadConversation(conversationId) {
    console.log(`Loading conversation: ${conversationId}`);
    
    const conversation = savedConversations.find(c => c.id === conversationId);
    if (!conversation) {
        console.error(`Conversation ${conversationId} not found`);
        return;
    }
    
    // Save current conversation before switching (if there's content and it's different)
    if (currentConversationId && 
        currentConversationId !== conversationId && 
        conversationHistory.length > 0) {
        console.log('Saving current conversation before switching');
        saveCurrentConversationToPersistentStorage();
    }
    
    // Set new conversation as current
    currentConversationId = conversationId;
    conversationHistory = [...conversation.history]; // Deep copy to prevent reference issues
    
    console.log(`Loaded conversation with ${conversationHistory.length} messages`);
    
    // Clear and rebuild the UI
    clearMessagesArea();
    loadConversationUI();
    
    // Update conversation list to show active state
    updateConversationList();
    
    // Update chat title
    updateChatTitle(conversation.title);
    
    // Save current conversation ID to localStorage
    saveToLocalStorage(STORAGE_KEYS.CURRENT_ID, currentConversationId);
    
    console.log(`Successfully switched to conversation: ${conversation.title}`);
}

// Updated startNewChat function with proper cleanup
function startNewChat() {
    console.log('Starting new chat...');
    
    // Save current conversation if it exists and has content
    if (currentConversationId && conversationHistory.length > 0) {
        console.log('Saving current conversation before starting new chat');
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
    
    console.log('New chat started successfully');
}

// Updated clearChat function (for "Getting Started" button)
function clearChat() {
    console.log('Clearing chat (Getting Started clicked)...');
    
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
    console.log(`Attempting to delete conversation: ${conversationId}`);
    
    // Find the conversation to get its title for confirmation
    const conversation = savedConversations.find(c => c.id === conversationId);
    if (!conversation) {
        console.error(`Conversation ${conversationId} not found`);
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
            console.log(`Removed conversation from array: ${conversation.title}`);
        }
        
        // Handle if we're deleting the currently active conversation
        if (currentConversationId === conversationId) {
            console.log('Deleting currently active conversation, starting new chat');
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
        
        console.log(`Successfully deleted conversation: ${conversation.title}`);
        
    } catch (error) {
        console.error('Error deleting conversation:', error);
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
        console.error('Initialization error:', error);
    }
    
    // Initialize conversation list
    updateConversationList();
}

// Enhanced regenerate function that maintains conversation context
function regenerateResponse() {
    if (conversationHistory.length >= 2) {
        const lastUserMessage = conversationHistory[conversationHistory.length - 2];
        
        console.log('Regenerating response for:', lastUserMessage.content.substring(0, 50) + '...');
        
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
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: lastUserMessage.content })
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
                        console.log('Response regenerated successfully');
                    }
                }
            } catch (error) {
                hideTypingIndicator();
                console.error('Error regenerating response:', error);
                addMessageToUI('assistant', 'Sorry, I encountered an error while regenerating the response.', true);
            }
        }, 1000);
    }
}

function showUploadModal() {
    document.getElementById('upload-modal').classList.add('active');
    
    // Don't clear files automatically - let user decide
    updateUploadedFilesList();
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
            showNotification(`⚠️ ${file.name}: File already added`);
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
        console.log(`Added file: ${file.name} (${formatFileSize(file.size)})`);
    });

    updateUploadedFilesList();

    // Show summary notification
    if (addedCount > 0) {
        showNotification(`✅ Added ${addedCount} file${addedCount > 1 ? 's' : ''} for processing`);
    }
    
    if (errorCount > 0) {
        console.log(`${errorCount} files were rejected due to validation errors`);
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
    processBtn.innerHTML = '<span>⏳</span> Processing...';

    try {
        // Set all files to processing status
        uploadedFiles.forEach(fileObj => fileObj.status = 'processing');
        updateUploadedFilesList();

        // Create FormData for upload
        const formData = new FormData();
        uploadedFiles.forEach(fileObj => {
            formData.append('files', fileObj.file);
        });

        console.log(`Uploading ${uploadedFiles.length} files...`);

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Upload response:', result);
            
            let successCount = 0;

            if (result.success && result.results && Array.isArray(result.results)) {
                // Process individual file results
                result.results.forEach(fileResult => {
                    const fileObj = uploadedFiles.find(f => f.name === fileResult.filename);
                    if (fileObj) {
                        if (fileResult.success) {
                            fileObj.status = 'success';
                            successCount++;
                            console.log(`✅ ${fileResult.filename} processed successfully`);
                        } else {
                            fileObj.status = 'error';
                            fileObj.error = fileResult.error || 'Processing failed';
                            console.log(`❌ ${fileResult.filename} failed: ${fileObj.error}`);
                        }
                    }
                });
            } else if (result.success) {
                // Fallback: if no individual results but overall success
                console.log('No individual results, assuming all successful');
                uploadedFiles.forEach(fileObj => {
                    fileObj.status = 'success';
                    successCount++;
                });
            } else {
                // Overall failure
                console.error('Upload failed:', result.error || 'Unknown error');
                uploadedFiles.forEach(fileObj => {
                    fileObj.status = 'error';
                    fileObj.error = result.error || 'Upload failed';
                });
            }

            // Update UI with final status
            updateUploadedFilesList();

            // Refresh stats if any files were processed successfully
            if (successCount > 0) {
                console.log(`${successCount} files uploaded successfully, refreshing stats...`);
                await loadSystemStats(true); // Force refresh after upload
            }

            // Show completion message
            setTimeout(() => {
                if (successCount > 0) {
                    showNotification(`Successfully processed ${successCount} of ${uploadedFiles.length} documents!`);
                } else {
                    showNotification(`Failed to process any documents. Please check the files and try again.`);
                }
                
                // Only close modal if at least one file was successful
                if (successCount > 0) {
                    hideUploadModal();
                }
            }, 1000);

        } else {
            // HTTP error response
            const errorText = await response.text();
            console.error('Upload HTTP error:', response.status, errorText);
            throw new Error(`Upload failed with status ${response.status}: ${errorText}`);
        }

    } catch (error) {
        console.error('Upload error:', error);
        
        // Set all files to error status
        uploadedFiles.forEach(fileObj => {
            fileObj.status = 'error';
            fileObj.error = error.message || 'Network error';
        });
        updateUploadedFilesList();
        
        showNotification('Upload failed. Please check your connection and try again.');
        
    } finally {
        // Reset button state
        processBtn.disabled = false;
        processBtn.innerHTML = '<span>🚀</span> Process Files';
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
                console.log('Using cached stats (updated', Math.round(timeSinceLastUpdate / 1000), 'seconds ago)');
                return statsCache;
            }
        }

        console.log('Fetching fresh stats from server...');
        const response = await fetch('/api/stats');
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.stats) {
                // Cache the stats
                statsCache = data.stats;
                lastStatsUpdate = Date.now();
                console.log('Stats loaded and cached successfully');
                return data.stats;
            }
        }
        return null;
    } catch (error) {
        console.error('Stats error:', error);
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
        console.error('Refresh error:', error);
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
    console.log('Stats cache cleared manually');
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
        console.error('Detailed stats error:', error);
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
            console.log('Auto-saving conversation...');
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
            console.log('Sidebar expanded - container classes:', container.classList.toString());
        } else {
            // Hide sidebar
            sidebar.classList.add('collapsed');
            container.classList.add('sidebar-collapsed');
            toggleIcon.textContent = '≫';
            toggleBtn.classList.add('active');
            console.log('Sidebar collapsed - container classes:', container.classList.toString());
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
    console.log('=== Conversation State Debug ===');
    console.log('Current conversation ID:', currentConversationId);
    console.log('Conversation history length:', conversationHistory.length);
    console.log('Saved conversations count:', savedConversations.length);
    console.log('Conversation counter:', conversationCounter);
    console.log('Stats cache info:', getStatsCacheInfo());
    console.log('===============================');
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
    console.log('showEditModal called, currentMenuConversationId:', currentMenuConversationId);
    
    // Store the ID before closing menu (which clears currentMenuConversationId)
    const conversationId = currentMenuConversationId;
    closeConversationMenu();
    
    if (!conversationId) {
        console.error('No conversationId available');
        showNotification('No conversation selected');
        return;
    }
    
    const conversation = savedConversations.find(c => c.id === conversationId);
    if (!conversation) {
        console.error('Conversation not found for ID:', conversationId);
        showNotification('Conversation not found');
        return;
    }
    
    // Set the ID back for the edit operation
    currentMenuConversationId = conversationId;
    
    const modal = document.getElementById('edit-modal');
    const input = document.getElementById('edit-input');
    
    if (!modal || !input) {
        console.error('Modal or input not found:', { modal, input });
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
    console.log('showDeleteModal called, currentMenuConversationId:', currentMenuConversationId);
    
    // Store the ID before closing menu (which clears currentMenuConversationId)
    const conversationId = currentMenuConversationId;
    closeConversationMenu();
    
    if (!conversationId) {
        console.error('No conversationId available');
        showNotification('No conversation selected');
        return;
    }
    
    const conversation = savedConversations.find(c => c.id === conversationId);
    if (!conversation) {
        console.error('Conversation not found for ID:', conversationId);
        showNotification('Conversation not found');
        return;
    }
    
    // Set the ID back for the delete operation
    currentMenuConversationId = conversationId;
    
    const modal = document.getElementById('delete-modal');
    const message = document.getElementById('delete-message');
    
    if (!modal || !message) {
        console.error('Modal or message not found:', { modal, message });
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
        console.error('Error deleting conversation:', error);
        closeDeleteModal();
        showNotification('Error deleting conversation. Please try again.');
    }
}

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
});