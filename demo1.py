import streamlit as st
import sys
import io
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from get_embedding_function import get_embedding_function

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

def main():
    # Streamlit page configuration
    st.set_page_config(
        page_title="RAG Chatbot with History", 
        page_icon="🤖",
        layout="wide"
    )
    
    # Title and description
    st.title("🤖 RAG Chatbot with Memory")
    st.markdown("Ask questions about your documents! I'll remember our conversation.")
    
    # Initialize RAG system
    if 'rag_initialized' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            try:
                st.session_state.db, st.session_state.model = initialize_rag_system()
                st.session_state.rag_initialized = True
                st.success("RAG system initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize RAG system: {str(e)}")
                st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add a welcome message
        welcome_msg = "Hello! I'm your RAG assistant. I can answer questions based on your documents and remember our conversation. What would you like to know?"
        st.session_state.messages.append(AIMessage(welcome_msg))
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Chat Controls")
        
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.session_state.messages.append(AIMessage("Chat history cleared. How can I help you?"))
            st.rerun()
        
        st.header("Chat Statistics")
        st.metric("Messages", len(st.session_state.messages))
        
        st.header("Settings")
        show_sources = st.checkbox("Show Sources", value=True)
        max_history = st.slider("Max History Context", 2, 10, 6, help="Number of previous messages to consider")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
    
    # Chat input
    user_question = st.chat_input("Ask me anything about your documents...")
    
    if user_question:
        # Add user message to chat
        st.session_state.messages.append(HumanMessage(user_question))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from RAG system
                    response, sources = query_rag_with_history(
                        user_question, 
                        st.session_state.messages[:-1],  # Exclude the current question
                        st.session_state.db, 
                        st.session_state.model
                    )
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display sources if enabled
                    if show_sources and sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.write(f"{i}. {source}")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append(AIMessage(response))
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(error_msg))

if __name__ == "__main__":
    main()