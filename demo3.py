import streamlit as st
import sys
import io
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from get_embedding_function import get_embedding_function
import time

# Fix encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Constants
CHROMA_PATH = "chroma"

# Enhanced prompt template that considers chat history
PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on the provided context and chat history.

Chat History:
{chat_history}

Context from documents:
{context}

---

Current Question: {question}

Please provide a detailed answer based on the context above. If the chat history is relevant to the current question, consider it in your response. If you cannot find the answer in the provided context, say "I don't have enough information in the provided documents to answer this question."
"""

def initialize_rag_system():
    """Initialize the RAG system components"""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = Ollama(model="mistral")
    return db, model

def format_chat_history(messages):
    """Format chat history for the prompt"""
    if not messages:
        return "No previous conversation."
    
    formatted_history = []
    for message in messages[-6:]:  # Keep last 6 messages for context
        if isinstance(message, HumanMessage):
            formatted_history.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_history.append(f"Assistant: {message.content}")
    
    return "\n".join(formatted_history)

def query_rag_with_history(query_text: str, chat_history, db, model):
    """Query RAG system with chat history context"""
    
    # Search the database
    results = db.similarity_search_with_score(query_text, k=5)
    
    # Prepare context from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Format chat history
    history_text = format_chat_history(chat_history)
    
    # Create prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text, 
        question=query_text,
        chat_history=history_text
    )
    
    # Get response from model
    response_text = model.invoke(prompt)
    
    # Get sources
    sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]
    
    return response_text, sources

def apply_elegant_css():
    """Apply sophisticated, elegant CSS styling"""
    st.markdown("""
    <style>
    /* Import elegant fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Source+Serif+Pro:wght@400;600&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #34495e;
        --accent-color: #3498db;
        --text-primary: #2c3e50;
        --text-secondary: #7f8c8d;
        --background-primary: #ffffff;
        --background-secondary: #f8f9fa;
        --border-color: #e9ecef;
        --shadow-subtle: 0 2px 4px rgba(0,0,0,0.06);
        --shadow-elevated: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    /* Global typography and layout */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background-color: var(--background-secondary);
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1000px;
    }
    
    /* Elegant header */
    .elegant-header {
        text-align: center;
        padding: 3rem 0 2rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 3rem;
        background: var(--background-primary);
    }
    
    .elegant-header h1 {
        font-family: 'Source Serif Pro', serif;
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .elegant-header .subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--background-primary);
        border-right: 1px solid var(--border-color);
    }
    
    .sidebar-section {
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #f1f3f4;
    }
    
    .sidebar-section:last-child {
        border-bottom: none;
    }
    
    .sidebar-section h3 {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }
    
    /* Chat message styling */
    .chat-message {
        margin: 1.5rem 0;
        padding: 1.5rem;
        background: var(--background-primary);
        border-radius: 8px;
        box-shadow: var(--shadow-subtle);
        border: 1px solid var(--border-color);
    }
    
    .chat-message.user {
        background: #f8f9fa;
        border-left: 3px solid var(--accent-color);
    }
    
    .chat-message.assistant {
        background: var(--background-primary);
    }
    
    .message-header {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .message-content {
        color: var(--text-primary);
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-subtle);
    }
    
    .stButton > button:hover {
        background-color: var(--secondary-color);
        box-shadow: var(--shadow-elevated);
        transform: translateY(-1px);
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stChatInput > div > div > input {
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 0.75rem;
        background: var(--background-primary);
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stChatInput > div > div > input:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    
    /* Metrics styling */
    .metric-container {
        background: var(--background-primary);
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid var(--border-color);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }
    
    /* Sources styling */
    .sources-container {
        background: #f8f9fa;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .source-item {
        font-family: 'SF Mono', 'Monaco', 'Cascadia Code', monospace;
        font-size: 0.85rem;
        color: var(--text-secondary);
        padding: 0.25rem 0;
        border-bottom: 1px solid #e9ecef;
    }
    
    .source-item:last-child {
        border-bottom: none;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-loading {
        background: #e2e3e5;
        color: #383d41;
        border: 1px solid #d6d8db;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: var(--accent-color);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--background-primary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        font-weight: 500;
        color: var(--text-primary);
    }
    
    /* Toggle and slider styling */
    .stToggle {
        padding: 0.25rem 0;
    }
    
    .stSlider {
        padding: 0.5rem 0;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
    
    /* Chat input container */
    .stChatInput {
        background: var(--background-primary);
        border-top: 1px solid var(--border-color);
        padding: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def create_elegant_header():
    """Create a sophisticated header"""
    st.markdown("""
    <div class="elegant-header">
        <h1>Document Intelligence Assistant</h1>
        <div class="subtitle">Sophisticated retrieval-augmented generation with conversation memory</div>
    </div>
    """, unsafe_allow_html=True)

def create_metric_display(label, value):
    """Create an elegant metric display"""
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def display_chat_message(content, is_user=False):
    """Display a chat message with elegant styling"""
    message_type = "user" if is_user else "assistant"
    header = "You" if is_user else "Assistant"
    
    st.markdown(f"""
    <div class="chat-message {message_type}">
        <div class="message-header">{header}</div>
        <div class="message-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def display_sources_elegantly(sources):
    """Display sources in an elegant format"""
    if sources:
        st.markdown("""
        <div class="sources-container">
            <div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--text-primary);">Document Sources</div>
        """, unsafe_allow_html=True)
        
        for i, source in enumerate(sources, 1):
            st.markdown(f"""
            <div class="source-item">{i}. {source}</div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_status(message, status_type="loading"):
    """Show elegant status indicator"""
    icon = "⏳" if status_type == "loading" else "✓"
    st.markdown(f"""
    <div class="status-indicator status-{status_type}">
        <span>{icon}</span>
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Document Intelligence Assistant",
        page_icon="📖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply elegant styling
    apply_elegant_css()
    
    # Create header
    create_elegant_header()
    
    # Initialize RAG system
    if 'rag_initialized' not in st.session_state:
        show_status("Initializing document intelligence system...", "loading")
        try:
            st.session_state.db, st.session_state.model = initialize_rag_system()
            st.session_state.rag_initialized = True
            show_status("System ready", "success")
            time.sleep(1)
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = "Good day. I'm your document intelligence assistant, equipped with retrieval-augmented generation capabilities and conversation memory. How may I assist you with your document queries today?"
        st.session_state.messages.append(AIMessage(welcome_msg))
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### Session Control")
        
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.messages.append(AIMessage("Conversation cleared. Ready for new inquiries."))
            st.rerun()
        
        # Export functionality
        if len(st.session_state.messages) > 1:
            chat_export = "\n\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in st.session_state.messages
            ])
            st.download_button(
                "Export Conversation",
                chat_export,
                file_name="conversation_export.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Statistics
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### Session Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            create_metric_display("Total Messages", len(st.session_state.messages))
        with col2:
            user_count = len([m for m in st.session_state.messages if isinstance(m, HumanMessage)])
            create_metric_display("User Queries", user_count)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Settings
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### Configuration")
        
        show_sources = st.toggle("Display document sources", value=True)
        context_length = st.slider("Conversation context length", 2, 10, 6)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat area
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            display_chat_message(message.content, is_user=True)
        elif isinstance(message, AIMessage):
            display_chat_message(message.content, is_user=False)
    
    # Chat input
    user_query = st.chat_input("Enter your question...")
    
    if user_query:
        # Add user message
        st.session_state.messages.append(HumanMessage(user_query))
        display_chat_message(user_query, is_user=True)
        
        # Process query
        with st.spinner("Processing your inquiry..."):
            try:
                response, sources = query_rag_with_history(
                    user_query,
                    st.session_state.messages[:-1],
                    st.session_state.db,
                    st.session_state.model
                )
                
                # Display response
                display_chat_message(response, is_user=False)
                
                # Display sources if enabled
                if show_sources:
                    display_sources_elegantly(sources)
                
                # Add to history
                st.session_state.messages.append(AIMessage(response))
                
            except Exception as e:
                error_msg = f"I apologize, but I encountered an error processing your request: {str(e)}"
                display_chat_message(error_msg, is_user=False)
                st.session_state.messages.append(AIMessage(error_msg))

if __name__ == "__main__":
    main()