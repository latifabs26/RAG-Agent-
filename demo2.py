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

def apply_custom_css():
    """Apply custom CSS for a beautiful interface"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Custom header with gradient */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .custom-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .custom-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Chat container */
    .chat-container {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* User message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        font-weight: 500;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        color: #2d3748;
        padding: 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Chat input styling */
    .stChatInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e2e8f0;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        background: white;
        transition: all 0.3s ease;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Loading animation */
    .loading-pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Status indicators */
    .status-success {
        color: #48bb78;
        font-weight: 600;
    }
    
    .status-error {
        color: #f56565;
        font-weight: 600;
    }
    
    /* Sources styling */
    .source-item {
        background: #f7fafc;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        border-left: 3px solid #667eea;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar improvements */
    .sidebar-content {
        padding: 1rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%);
    }
    </style>
    """, unsafe_allow_html=True)

def create_custom_header():
    """Create a beautiful custom header"""
    st.markdown("""
    <div class="custom-header">
        <h1>🤖 RAG Assistant</h1>
        <p>Intelligent document chat with memory • Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, icon="📊"):
    """Create a beautiful metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 1.5rem; font-weight: 700;">{value}</div>
        <div style="font-size: 0.9rem; opacity: 0.9;">{title}</div>
    </div>
    """, unsafe_allow_html=True)

def display_sources_beautifully(sources):
    """Display sources in a beautiful format"""
    if sources:
        st.markdown("**📚 Sources:**")
        for i, source in enumerate(sources, 1):
            st.markdown(f"""
            <div class="source-item">
                <strong>{i}.</strong> {source}
            </div>
            """, unsafe_allow_html=True)

def main():
    # Streamlit page configuration
    st.set_page_config(
        page_title="RAG Assistant", 
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    apply_custom_css()
    
    # Create beautiful header
    create_custom_header()
    
    # Initialize RAG system
    if 'rag_initialized' not in st.session_state:
        with st.spinner("🚀 Initializing RAG system..."):
            try:
                st.session_state.db, st.session_state.model = initialize_rag_system()
                st.session_state.rag_initialized = True
                st.success("✅ RAG system ready!", icon="🎉")
                time.sleep(1)  # Brief pause for visual feedback
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}", icon="🚨")
                st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = "👋 Hello! I'm your intelligent RAG assistant. I can answer questions based on your documents and remember our entire conversation. What would you like to explore today?"
        st.session_state.messages.append(AIMessage(welcome_msg))
    
    # Beautiful sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.markdown("### 🎛️ Chat Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                st.session_state.messages = []
                welcome_msg = "🔄 Chat cleared! Ready for new questions."
                st.session_state.messages.append(AIMessage(welcome_msg))
                st.rerun()
        
        with col2:
            if st.button("📤 Export Chat", type="secondary", use_container_width=True):
                # Create export functionality
                chat_export = "\n".join([
                    f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                    for msg in st.session_state.messages
                ])
                st.download_button(
                    "📥 Download",
                    chat_export,
                    file_name="chat_history.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        st.markdown("### 📊 Statistics")
        
        # Beautiful metrics
        col1, col2 = st.columns(2)
        with col1:
            create_metric_card("Messages", len(st.session_state.messages), "💬")
        with col2:
            user_msgs = len([m for m in st.session_state.messages if isinstance(m, HumanMessage)])
            create_metric_card("Questions", user_msgs, "❓")
        
        st.markdown("### ⚙️ Settings")
        
        show_sources = st.toggle("📚 Show Sources", value=True, help="Display document sources for answers")
        
        max_history = st.slider(
            "🧠 Memory Length", 
            min_value=2, 
            max_value=10, 
            value=6, 
            help="How many previous messages to remember"
        )
        
        st.markdown("### 🎨 Theme")
        theme_option = st.selectbox(
            "Choose theme",
            ["🌟 Gradient", "🌙 Dark Mode", "☀️ Light Mode"],
            index=0
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages with beautiful styling
    for i, message in enumerate(st.session_state.messages):
        if isinstance(message, HumanMessage):
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {message.content}
            </div>
            """, unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.markdown(f"""
            <div class="assistant-message">
                <strong>🤖 Assistant:</strong><br>{message.content}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input with beautiful styling
    user_question = st.chat_input("💭 Ask me anything about your documents...")
    
    if user_question:
        # Add user message to chat
        st.session_state.messages.append(HumanMessage(user_question))
        
        # Display user message
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong> {user_question}
        </div>
        """, unsafe_allow_html=True)
        
        # Generate response with beautiful loading
        with st.spinner("🤔 Thinking deeply..."):
            try:
                # Simulate processing time for better UX
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Get response from RAG system
                response, sources = query_rag_with_history(
                    user_question, 
                    st.session_state.messages[:-1],
                    st.session_state.db, 
                    st.session_state.model
                )
                
                progress_bar.empty()
                
                # Display response beautifully
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant:</strong><br>{response}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources if enabled
                if show_sources:
                    with st.expander("📚 View Sources", expanded=False):
                        display_sources_beautifully(sources)
                
                # Add assistant response to chat history
                st.session_state.messages.append(AIMessage(response))
                
                # Success feedback
                st.success("✨ Response generated!", icon="🎯")
                
            except Exception as e:
                error_msg = f"😞 Sorry, I encountered an error: {str(e)}"
                st.error(error_msg, icon="🚨")
                st.session_state.messages.append(AIMessage(error_msg))

if __name__ == "__main__":
    main()