"""
Streamlit UI for RAG Document Assistant.
 
This frontend provides a user-friendly interface to:
- Upload PDF documents
- Query and chat with documents
- Visualize retrieved context
- View system statistics
"""
 
import os
import requests
import streamlit as st
from typing import List, Dict, Optional
import time
 
# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
 
# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .stats-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)
 
 
def check_api_health() -> bool:
    """Check if the API is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except Exception:
        return False
 
 
def upload_pdf(file) -> Optional[Dict]:
    """Upload a PDF file to the API."""
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload failed: {str(e)}")
        return None
 
 
def query_documents(query: str, top_k: int = 5) -> Optional[Dict]:
    """Query documents without generation."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Query failed: {str(e)}")
        return None
 
 
def generate_answer(query: str, top_k: int = 5, return_context: bool = True) -> Optional[Dict]:
    """Generate an answer using RAG."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={"query": query, "top_k": top_k, "return_context": return_context}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Generation failed: {str(e)}")
        return None
 
 
def get_stats() -> Optional[Dict]:
    """Get system statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch stats: {str(e)}")
        return None
 
 
def clear_all_data() -> bool:
    """Clear all documents from the system."""
    try:
        response = requests.delete(f"{API_BASE_URL}/clear")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to clear data: {str(e)}")
        return False
 
 
def display_source(source: Dict, index: int):
    """Display a single source document."""
    st.markdown(f"""
    <div class="source-box">
        <strong>Source {index + 1}</strong>
        (Score: {source.get('score', 0):.4f}) -
        <em>{source.get('metadata', {}).get('source', 'Unknown')}</em>
        <p style="margin-top: 0.5rem;">{source.get('text', '')}</p>
    </div>
    """, unsafe_allow_html=True)
 
 
def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
 
 
def main():
    """Main application."""
    initialize_session_state()
 
    # Header
    st.markdown('<div class="main-header">📚 RAG Document Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload PDFs and chat with your documents using AI</div>', unsafe_allow_html=True)
 
    # Check API health
    if not check_api_health():
        st.error("⚠️ Cannot connect to the API backend. Please ensure the FastAPI server is running.")
        st.info(f"Expected API URL: {API_BASE_URL}")
        st.stop()
 
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
 
        # File upload section
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], key="pdf_uploader")
 
        if uploaded_file is not None:
            if st.button("📤 Upload & Process", use_container_width=True):
                with st.spinner("Processing document..."):
                    result = upload_pdf(uploaded_file)
                    if result:
                        st.success(f"✅ {result['filename']} processed successfully!")
                        st.json(result['result'])
                        st.session_state.uploaded_files.append(result['filename'])
                        # Rerun to refresh stats
                        time.sleep(1)
                        st.rerun()
 
        st.divider()
 
        # Statistics section
        st.subheader("📊 System Statistics")
        if st.button("🔄 Refresh Stats", use_container_width=True):
            st.rerun()
 
        stats_data = get_stats()
        if stats_data:
            stats = stats_data.get('stats', {})
 
            # Display key metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get('total_documents', 0))
            with col2:
                st.metric("Chunks", stats.get('total_chunks', 0))
 
            # Configuration details
            with st.expander("⚙️ Configuration"):
                config = stats.get('config', {})
                st.json(config)
 
        st.divider()
 
        # Data management
        st.subheader("🗑️ Data Management")
        if st.button("🗑️ Clear All Documents", use_container_width=True, type="secondary"):
            if st.button("⚠️ Confirm Clear", use_container_width=True):
                if clear_all_data():
                    st.success("All documents cleared!")
                    st.session_state.chat_history = []
                    st.session_state.uploaded_files = []
                    time.sleep(1)
                    st.rerun()
 
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Search", "ℹ️ About"])
 
    # Tab 1: Chat Interface (RAG with Generation)
    with tab1:
        st.header("Chat with Your Documents")
        st.markdown("Ask questions and get AI-generated answers based on your uploaded documents.")
 
        # Chat input
        col1, col2 = st.columns([4, 1])
        with col1:
            user_query = st.text_input("Ask a question:", key="chat_query", placeholder="What is this document about?")
        with col2:
            top_k = st.number_input("Top K", min_value=1, max_value=10, value=3, help="Number of context chunks to use")
 
        if st.button("🚀 Generate Answer", use_container_width=True):
            if not user_query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Generating answer..."):
                    result = generate_answer(user_query, top_k=top_k, return_context=True)
 
                    if result:
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'query': user_query,
                            'answer': result['answer'],
                            'model': result['model'],
                            'context': result.get('context_used', [])
                        })
 
                        # Display the answer
                        st.markdown(f"""
                        <div class="answer-box">
                            <strong>🤖 Answer ({result['model']}):</strong>
                            <p style="margin-top: 0.5rem; font-size: 1.1rem;">{result['answer']}</p>
                        </div>
                        """, unsafe_allow_html=True)
 
                        # Display context sources
                        if result.get('context_used'):
                            with st.expander(f"📄 View {len(result['context_used'])} Context Sources"):
                                for idx, context in enumerate(result['context_used']):
                                    st.markdown(f"**Context {idx + 1}:**")
                                    st.text(context)
                                    st.divider()
 
        # Display chat history
        if st.session_state.chat_history:
            st.divider()
            st.subheader("💭 Chat History")
 
            for idx, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.container():
                    st.markdown(f"**Q{len(st.session_state.chat_history) - idx}:** {chat['query']}")
                    st.markdown(f"**A:** {chat['answer']}")
                    st.caption(f"Model: {chat['model']}")
                    st.divider()
 
    # Tab 2: Search Interface (Retrieval Only)
    with tab2:
        st.header("Search Documents")
        st.markdown("Search for relevant passages without AI generation (pure retrieval).")
 
        col1, col2 = st.columns([4, 1])
        with col1:
            search_query = st.text_input("Search query:", key="search_query", placeholder="machine learning algorithms")
        with col2:
            search_top_k = st.number_input("Results", min_value=1, max_value=20, value=5)
 
        if st.button("🔍 Search", use_container_width=True):
            if not search_query:
                st.warning("Please enter a search query.")
            else:
                with st.spinner("Searching..."):
                    results = query_documents(search_query, top_k=search_top_k)
 
                    if results:
                        st.success(f"Found {results['count']} results")
 
                        # Display results
                        for idx, result in enumerate(results['results']):
                            display_source(result, idx)
 
    # Tab 3: About
    with tab3:
        st.header("About RAG Document Assistant")
 
        st.markdown("""
        ### 🎯 What is this?
 
        The **RAG Document Assistant** is a production-ready Retrieval-Augmented Generation (RAG) system that allows you to:
 
        - 📤 **Upload** PDF documents
        - 🔍 **Search** through your documents using semantic similarity
        - 💬 **Chat** with your documents using AI-powered question answering
        - 📊 **Visualize** the sources and context used for answers
 
        ### 🏗️ Architecture
 
        This application follows a microservices architecture:
 
        ```
        User ↔ Streamlit UI (Frontend) ↔ FastAPI (Backend) ↔ RAG Pipeline
        ```
 
        - **Frontend (Streamlit)**: This user interface
        - **Backend (FastAPI)**: RESTful API for document processing and querying
        - **RAG Pipeline**: Document chunking, embedding, vector storage, and generation
 
        ### 🧠 How it works
 
        1. **Upload**: PDFs are chunked into smaller segments
        2. **Embed**: Each chunk is converted to a vector embedding
        3. **Store**: Embeddings are stored in a vector database
        4. **Retrieve**: When you ask a question, relevant chunks are found
        5. **Generate**: An LLM uses the context to generate accurate answers
 
        ### 🛠️ Technology Stack
 
        - **Frontend**: Streamlit
        - **Backend**: FastAPI
        - **Embeddings**: Sentence Transformers
        - **Vector Store**: ChromaDB / FAISS
        - **LLM**: OpenAI / Ollama / HuggingFace
 
        ### 📝 API Endpoints
 
        - `POST /upload` - Upload PDF documents
        - `POST /query` - Search documents (retrieval only)
        - `POST /generate` - Generate answers (RAG)
        - `GET /stats` - Get system statistics
        - `DELETE /clear` - Clear all documents
 
        ---
 
        **Version**: 1.0.
        **API Base URL**: `{}`
        """.format(API_BASE_URL))
 
        # Connection test
        st.subheader("🔌 Connection Test")
        if st.button("Test API Connection"):
            if check_api_health():
                st.success("✅ API is healthy and reachable!")
            else:
                st.error("❌ Cannot reach the API")
 
 
if __name__ == "__main__":
    main()