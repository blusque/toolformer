class ToolformerChat {
    constructor() {
        this.messages = [];
        this.isLoading = false;
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        this.chatInput = document.querySelector('.chat-input');
        this.sendBtn = document.querySelector('.send-btn');
        this.exampleCards = document.querySelectorAll('.example-card');
        this.mainContainer = document.querySelector('.main-container');
        this.newChatBtn = document.querySelector('.new-chat-btn');
        this.welcomeSection = document.querySelector('.welcome-section');
        this.examplesGrid = document.querySelector('.examples-grid');
    }

    bindEvents() {
        // Input events
        this.chatInput.addEventListener('input', (e) => this.handleInputChange(e));
        this.chatInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        
        // Button events
        this.sendBtn.addEventListener('click', () => this.handleSendMessage());
        this.newChatBtn.addEventListener('click', () => this.handleNewChat());
        
        // Example card events
        this.exampleCards.forEach(card => {
            card.addEventListener('click', (e) => this.handleExampleClick(e));
        });
    }

    handleInputChange(e) {
        const value = e.target.value.trim();
        this.sendBtn.disabled = value === '' || this.isLoading;
        
        // Auto-resize textarea
        e.target.style.height = 'auto';
        e.target.style.height = e.target.scrollHeight + 'px';
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!this.sendBtn.disabled) {
                this.handleSendMessage();
            }
        }
    }

    handleExampleClick(e) {
        const card = e.currentTarget;
        const exampleText = card.querySelector('.example-text').textContent;
        this.chatInput.value = exampleText;
        this.chatInput.focus();
        this.sendBtn.disabled = false;
        
        // Auto-resize for the example text
        this.chatInput.style.height = 'auto';
        this.chatInput.style.height = this.chatInput.scrollHeight + 'px';
    }

    async handleSendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || this.isLoading) return;

        // Add user message to conversation
        this.addMessage('user', message);
        
        // Clear input and disable send button
        this.chatInput.value = '';
        this.chatInput.style.height = 'auto';
        this.setLoading(true);

        try {
            // Send message to backend
            const response = await this.sendMessageToAPI(message);
            
            // Add assistant response to conversation
            this.addMessage('assistant', response.message, response.tools_used);
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.', [], true);
        } finally {
            this.setLoading(false);
        }
    }

    async sendMessageToAPI(message) {
        const requestBody = {
            messages: [
                ...this.messages,
                { role: 'user', content: message }
            ],
            max_tokens: 150,
            temperature: 0.7
        };

        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    addMessage(role, content, tools = [], isError = false) {
        // Add to messages array
        this.messages.push({ role, content });

        // Create chat interface if it doesn't exist
        if (!this.chatContainer) {
            this.createChatInterface();
        }

        // Create message element
        const messageElement = this.createMessageElement(role, content, tools, isError);
        this.chatMessagesContainer.appendChild(messageElement);

        // Scroll to bottom
        this.scrollToBottom();
    }

    createChatInterface() {
        // Hide welcome section and examples
        this.welcomeSection.style.display = 'none';
        this.examplesGrid.style.display = 'none';

        // Create chat container
        this.chatContainer = document.createElement('div');
        this.chatContainer.className = 'chat-container';
        this.chatContainer.style.cssText = `
            flex: 1;
            width: 100%;
            max-width: 800px;
            display: flex;
            flex-direction: column;
            margin-bottom: 2rem;
        `;

        // Create messages container
        this.chatMessagesContainer = document.createElement('div');
        this.chatMessagesContainer.className = 'chat-messages';
        this.chatMessagesContainer.style.cssText = `
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            max-height: 60vh;
        `;

        this.chatContainer.appendChild(this.chatMessagesContainer);
        this.mainContainer.insertBefore(this.chatContainer, this.mainContainer.children[1]);
    }

    createMessageElement(role, content, tools = [], isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        const isUser = role === 'user';
        messageDiv.style.cssText = `
            align-self: ${isUser ? 'flex-end' : 'flex-start'};
            max-width: 80%;
            padding: 1rem;
            border-radius: 12px;
            background-color: ${isUser ? '#10a37f' : isError ? '#dc2626' : '#40414f'};
            color: ${isUser || isError ? '#fff' : '#d1d5db'};
            border: 1px solid ${isUser ? '#0d8f6f' : isError ? '#b91c1c' : '#565869'};
            word-wrap: break-word;
        `;

        // Create content container
        const contentDiv = document.createElement('div');
        contentDiv.style.cssText = 'white-space: pre-wrap; line-height: 1.5;';
        contentDiv.textContent = content;
        messageDiv.appendChild(contentDiv);

        // Add tools used indicator if available
        if (!isUser && tools && tools.length > 0) {
            const toolsDiv = document.createElement('div');
            toolsDiv.style.cssText = `
                margin-top: 0.5rem;
                padding-top: 0.5rem;
                border-top: 1px solid #565869;
                font-size: 0.75rem;
                color: #9ca3af;
            `;
            toolsDiv.innerHTML = `<strong>Tools used:</strong> ${tools.join(', ')}`;
            messageDiv.appendChild(toolsDiv);
        }

        return messageDiv;
    }

    setLoading(loading) {
        this.isLoading = loading;
        this.sendBtn.disabled = loading || this.chatInput.value.trim() === '';
        
        if (loading) {
            this.sendBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" class="animate-spin">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" opacity="0.25"/>
                    <path fill="currentColor" opacity="0.75" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                </svg>
            `;
            
            // Add spinning animation
            const style = document.createElement('style');
            style.textContent = `
                .animate-spin {
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            `;
            document.head.appendChild(style);
        } else {
            this.sendBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                </svg>
            `;
        }
    }

    scrollToBottom() {
        if (this.chatMessagesContainer) {
            this.chatMessagesContainer.scrollTop = this.chatMessagesContainer.scrollHeight;
        }
    }

    handleNewChat() {
        // Reset messages
        this.messages = [];
        
        // Remove chat interface if it exists
        if (this.chatContainer) {
            this.chatContainer.remove();
            this.chatContainer = null;
            this.chatMessagesContainer = null;
        }
        
        // Show welcome section and examples
        this.welcomeSection.style.display = 'block';
        this.examplesGrid.style.display = 'grid';
        
        // Clear and reset input
        this.chatInput.value = '';
        this.chatInput.style.height = 'auto';
        this.sendBtn.disabled = true;
        this.chatInput.focus();
    }
}

// Initialize the chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ToolformerChat();
});