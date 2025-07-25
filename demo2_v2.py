import streamlit as st
import sys
import io
import os
import tempfile
import shutil
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function
import time

# Fix encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.pdf': 'pdf',
    '.csv': 'csv',
    '.md': 'text',
    '.markdown': 'text',
    '.txt': 'text',
}

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
    
    # Ensure chroma directory exists
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = Ollama(model="mistral")
    return db, model

def load_single_pdf(file_path):
    """Load a single PDF file"""
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        st.error(f"❌ Failed to load PDF {Path(file_path).name}: {e}")
        return []

def load_single_csv(file_path):
    """Load a single CSV file with error handling"""
    try:
        # Try with default settings first
        loader = CSVLoader(file_path=file_path)
        return loader.load()
    except Exception as e1:
        try:
            # Try with different encoding
            loader = CSVLoader(
                file_path=file_path,
                encoding='utf-8',
                csv_args={'delimiter': ','}
            )
            return loader.load()
        except Exception as e2:
            try:
                # Try with semicolon delimiter (common in European CSVs)
                loader = CSVLoader(
                    file_path=file_path,
                    encoding='utf-8',
                    csv_args={'delimiter': ';'}
                )
                return loader.load()
            except Exception as e3:
                st.error(f"❌ Failed to load CSV {Path(file_path).name}: Multiple encoding/delimiter attempts failed")
                return []

def load_single_text(file_path):
    """Load a single text file with encoding handling"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            loader = TextLoader(file_path, encoding=encoding)
            return loader.load()
        except Exception as e:
            continue
    
    st.error(f"❌ Failed to load text file {Path(file_path).name} with any encoding")
    return []

def process_uploaded_file(uploaded_file, temp_dir):
    """Process a single uploaded file and return documents"""
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension not in SUPPORTED_EXTENSIONS:
        st.warning(f"⚠️ Unsupported file type: {uploaded_file.name}")
        return []
    
    # Save uploaded file to temp directory
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load documents based on file type
    file_type = SUPPORTED_EXTENSIONS[file_extension]
    
    if file_type == 'pdf':
        documents = load_single_pdf(temp_file_path)
    elif file_type == 'csv':
        documents = load_single_csv(temp_file_path)
    elif file_type == 'text':
        documents = load_single_text(temp_file_path)
    else:
        return []
    
    # Update metadata to include original filename
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name
        doc.metadata["uploaded"] = True
    
    return documents

def split_documents(documents: list[Document]):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    
    # For CSV files, use larger chunks since they contain structured data
    csv_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_chunks = []
    
    for doc in documents:
        source = doc.metadata.get("source", "")
        
        try:
            # Use different splitter for CSV files
            if source.lower().endswith('.csv'):
                chunks = csv_splitter.split_documents([doc])
            else:
                chunks = text_splitter.split_documents([doc])
            
            all_chunks.extend(chunks)
        except Exception as e:
            st.error(f"❌ Error splitting document {source}: {e}")
    
    return all_chunks

def calculate_chunk_ids(chunks):
    """Calculate unique IDs for chunks"""
    last_source_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        
        # For PDF files, use page number
        if source.lower().endswith('.pdf'):
            page = chunk.metadata.get("page", 0)
            current_source_id = f"{source}:{page}"
        # For CSV files, use row number if available
        elif source.lower().endswith('.csv'):
            row = chunk.metadata.get("row", 0)
            current_source_id = f"{source}:{row}"
        # For other files, just use the source
        else:
            current_source_id = f"{source}:0"

        # If the source ID is the same as the last one, increment the index.
        if current_source_id == last_source_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_source_id}:{current_chunk_index}"
        last_source_id = current_source_id

        # Add it to the metadata.
        chunk.metadata["id"] = chunk_id

    return chunks

def add_documents_to_chroma(chunks: list[Document], db):
    """Add new documents to the Chroma database"""
    if not chunks:
        return 0
    
    # Calculate chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Get existing documents
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    # Only add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        return len(new_chunks)
    else:
        return 0

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
    
    /* Upload area styling */
    .upload-area {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 2px dashed #cbd5e0;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
    }
    
    /* File info styling */
    .file-info {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #48bb78;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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
    </style>
    """, unsafe_allow_html=True)

def create_custom_header():
    """Create a beautiful custom header"""
    st.markdown("""
    <div class="custom-header">
        <h1>🤖 RAG Assistant</h1>
        <p>Intelligent document chat with memory • Upload & Chat with your files</p>
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

def get_database_stats(db):
    """Get statistics about the database"""
    try:
        existing_items = db.get(include=["metadatas"])
        total_docs = len(existing_items["ids"])
        
        # Count unique sources
        sources = set()
        for metadata in existing_items["metadatas"]:
            if metadata and "source" in metadata:
                sources.add(metadata["source"])
        
        return total_docs, len(sources)
    except:
        return 0, 0

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
                time.sleep(1)
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}", icon="🚨")
                st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = "👋 Hello! I'm your intelligent RAG assistant. Upload documents using the sidebar, then ask me questions about them. I can remember our entire conversation!"
        st.session_state.messages.append(AIMessage(welcome_msg))
    
    # Beautiful sidebar with file upload
    with st.sidebar:
        st.markdown("### 📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'csv', 'txt', 'md', 'markdown'],
            accept_multiple_files=True,
            help="Supported formats: PDF, CSV, TXT, MD, Markdown"
        )
        
        if uploaded_files:
            st.markdown("### 📋 Uploaded Files")
            
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                with st.spinner("Processing uploaded files..."):
                    total_processed = 0
                    total_chunks = 0
                    
                    # Create temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        all_documents = []
                        
                        # Process each uploaded file
                        for uploaded_file in uploaded_files:
                            st.write(f"📄 Processing: {uploaded_file.name}")
                            documents = process_uploaded_file(uploaded_file, temp_dir)
                            if documents:
                                all_documents.extend(documents)
                                total_processed += 1
                        
                        if all_documents:
                            # Split documents into chunks
                            chunks = split_documents(all_documents)
                            total_chunks = len(chunks)
                            
                            # Add to database
                            new_chunks_added = add_documents_to_chroma(chunks, st.session_state.db)
                            
                            if new_chunks_added > 0:
                                st.success(f"✅ Successfully processed {total_processed} files ({new_chunks_added} new chunks added)!")
                            else:
                                st.info("ℹ️ All documents were already in the database")
                        else:
                            st.error("❌ No documents could be processed")
            
            # Display file information
            for uploaded_file in uploaded_files:
                file_size = len(uploaded_file.getbuffer()) / 1024  # KB
                st.markdown(f"""
                <div class="file-info">
                    <strong>📄 {uploaded_file.name}</strong><br>
                    <small>Size: {file_size:.1f} KB | Type: {uploaded_file.type}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
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
        
        st.markdown("### 📊 Database Statistics")
        
        # Get database stats
        total_chunks, unique_sources = get_database_stats(st.session_state.db)
        
        col1, col2 = st.columns(2)
        with col1:
            create_metric_card("Documents", unique_sources, "📚")
        with col2:
            create_metric_card("Chunks", total_chunks, "🧩")
        
        st.markdown("### 💬 Chat Statistics")
        col1, col2 = st.columns(2)
        with col1:
            create_metric_card("Messages", len(st.session_state.messages), "💬")
        with col2:
            user_msgs = len([m for m in st.session_state.messages if isinstance(m, HumanMessage)])
            create_metric_card("Questions", user_msgs, "❓")
        
        st.markdown("### ⚙️ Settings")
        show_sources = st.toggle("📚 Show Sources", value=True, help="Display document sources for answers")
    
    # Main chat area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages with beautiful styling
    for message in st.session_state.messages:
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
    
    # Chat input
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
        
        # Generate response
        with st.spinner("🤔 Thinking deeply..."):
            try:
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
                
                # Display response
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
                
                st.success("✨ Response generated!", icon="🎯")
                
            except Exception as e:
                error_msg = f"😞 Sorry, I encountered an error: {str(e)}"
                st.error(error_msg, icon="🚨")
                st.session_state.messages.append(AIMessage(error_msg))

if __name__ == "__main__":
    main()