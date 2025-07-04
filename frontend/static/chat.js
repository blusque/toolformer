class ToolformerChat {
    constructor() {
        this.messages = [];
        this.isLoading = false;
        this.currentStreamingMessageIndex = -1;
        this.initializeElements();
        this.bindEvents();
        this.addStreamingStyles();
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

    addStreamingStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .streaming-cursor {
                animation: blink 1s infinite;
                color: #10a37f;
                font-weight: bold;
            }
            @keyframes blink {
                0%, 50% { opacity: 1; }
                51%, 100% { opacity: 0; }
            }
        `;
        document.head.appendChild(style);
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

    updateMessage(role, content, tools = [], isError = false) {
        // Simple method for error handling - adds a new message
        this.addMessage(role, content, tools, isError);
    }

    handleInputChange(e) {
        const value = e.target.value.trim();
        console.log('Input changed:', value);
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
            // Send messages to API
            const response = await this.sendMessagesToAPI();
            if (response.error) {
                throw new Error(response.error);
            }

            // Update last assistant message with response content
            this.updateMessage('assistant', response.message, response.tools_used || []);
        } catch (error) {
            console.error('Error sending message:', error);
            this.updateMessage('assistant', 'Sorry, I encountered an error. Please try again.', [], true);
        } finally {
            this.setLoading(false);
        }
    }

    async checkForClearMessages() {
        const response = await fetch('/clear_messages', {
            method: 'GET',
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data.should_clear;
    }

    async sendMessagesToAPI() {
        const requestBody = {
            messages: [
                ...this.messages
            ],
            max_tokens: 1024,
            temperature: 0.7,
            top_p: 1.0,
            top_k: 50
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

        const data = await response.json();
        return data;
    }

    async streamMessagesFromAPI() {
        const requestBody = {
            messages: [
                ...this.messages.slice(0, -1) // Exclude the empty assistant message we just added
            ],
            max_tokens: 1024,
            temperature: 0.7
        };

        const response = await fetch('/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulatedContent = '';

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6); // Remove 'data: ' prefix
                        if (dataStr.trim() === '') continue;

                        try {
                            const data = JSON.parse(dataStr);
                            
                            if (data.error) {
                                throw new Error(data.error);
                            }
                            
                            if (data.type === 'chunk' && data.content) {
                                accumulatedContent += data.content;
                                this.updateStreamingMessage(accumulatedContent);
                            } else if (data.type === 'done') {
                                // Final update with tools used
                                this.finalizeStreamingMessage(accumulatedContent, data.tools_used || []);
                                return;
                            }
                        } catch (parseError) {
                            console.warn('Failed to parse streaming data:', parseError);
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }

    updateStreamingMessage(content) {
        // Update the content of the last assistant message
        if (this.currentStreamingMessageIndex >= 0) {
            this.messages[this.currentStreamingMessageIndex].content = content;
            
            // Update the message element in the UI
            const messageElements = this.chatMessagesContainer.querySelectorAll('.assistant-message');
            const lastMessageElement = messageElements[messageElements.length - 1];
            
            if (lastMessageElement) {
                const contentDiv = lastMessageElement.querySelector('div');
                if (contentDiv) {
                    contentDiv.textContent = content;
                    // Add cursor effect to show streaming
                    contentDiv.style.position = 'relative';
                    contentDiv.innerHTML = content + '<span class="streaming-cursor">|</span>';
                }
            }
            
            this.scrollToBottom();
        }
    }

    finalizeStreamingMessage(content, tools = []) {
        // Final update without cursor
        if (this.currentStreamingMessageIndex >= 0) {
            this.messages[this.currentStreamingMessageIndex].content = content;
            
            // Update the message element in the UI
            const messageElements = this.chatMessagesContainer.querySelectorAll('.assistant-message');
            const lastMessageElement = messageElements[messageElements.length - 1];
            
            if (lastMessageElement) {
                // Remove the old content and recreate with tools if needed
                lastMessageElement.innerHTML = '';
                
                // Create content container
                const contentDiv = document.createElement('div');
                contentDiv.style.cssText = 'white-space: pre-wrap; line-height: 1.5;';
                contentDiv.textContent = content;
                lastMessageElement.appendChild(contentDiv);

                // Add tools used indicator if available
                if (tools && tools.length > 0) {
                    const toolsDiv = document.createElement('div');
                    toolsDiv.style.cssText = `
                        margin-top: 0.5rem;
                        padding-top: 0.5rem;
                        border-top: 1px solid #565869;
                        font-size: 0.75rem;
                        color: #9ca3af;
                    `;
                    toolsDiv.innerHTML = `<strong>Tools used:</strong> ${tools.join(', ')}`;
                    lastMessageElement.appendChild(toolsDiv);
                }
            }
            
            this.scrollToBottom();
        }
        
        // Reset streaming state
        this.currentStreamingMessageIndex = -1;
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
        // Reset messages and streaming state
        this.messages = [];
        this.currentStreamingMessageIndex = -1;
        
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