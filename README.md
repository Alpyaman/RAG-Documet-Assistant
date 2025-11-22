# RAG Document Assistant 🚀
 
An end-to-end **Retrieval-Augmented Generation (RAG)** system for intelligent document Q&A. Upload a PDF and chat with your documents using state-of-the-art NLP and vector search.
 
This project demonstrates production-ready ML engineering skills including:
- ✅ **NLP Processing**: PDF parsing, text chunking, semantic embeddings
- ✅ **Vector Databases**: ChromaDB for efficient similarity search
- ✅ **Modern ML Stack**: Sentence Transformers, LangChain-compatible architecture
- ✅ **Clean Architecture**: Modular design, type hints, comprehensive testing
- ✅ **MLOps Ready**: Configuration management, logging, performance monitoring
 
---
 
## 🎯 Project Overview
 
### What is RAG?
 
Retrieval-Augmented Generation combines:
1. **Information Retrieval**: Find relevant document chunks using semantic search
2. **Generation**: Use retrieved context to answer questions (future: integrate with LLMs)
 
### Architecture
 
```
PDF Document
    ↓
[Ingestion] → Extract text + metadata
    ↓
[Chunking] → Split into semantic chunks
    ↓
[Embeddings] → Convert to dense vectors (384-dim)
    ↓
[Vector Store] → Store in ChromaDB
    ↓
[Retrieval] → Similarity search for relevant chunks
    ↓
[Generation] → LLM generates answer from context
    ↓
Human-like Answer
```

---
 
## 📦 Installation
 
### Prerequisites
 
- Python 3.10+
- pip or conda
 
### Setup
 
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/RAG-Documet-Assistant.git
cd RAG-Documet-Assistant
```
 
2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
 
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
 
4. **Create data directories**
```bash
mkdir -p data/pdfs data/vector_db
```
 
---
 
## 🚀 Quick Start
 
### Basic Usage
 
```python
from rag_assistant import RAGPipeline
 
# Initialize the pipeline
pipeline = RAGPipeline()
 
# Process a PDF document
result = pipeline.process_pdf("data/pdfs/my_document.pdf")
print(f"Created {result.chunks_created} chunks in {result.processing_time:.2f}s")
 
# Query the document
results = pipeline.query("What is the main topic?", top_k=5)
 
for i, result in enumerate(results, 1):
    print(f"\n--- Result {i} (Similarity: {result['similarity']:.3f}) ---")
    print(result['text'][:200])
```
 
### Run Examples
 
```bash
# Basic usage example
python examples/basic_usage.py
 
# Advanced examples (batch processing, custom configs, etc.)
python examples/advanced_usage.py
```
 
---
 
## 📚 Core Components
 
### 1. PDF Ingestion (`ingestion.py`)
 
Extracts text and metadata from PDF files.
 
```python
from rag_assistant import PDFIngestor
 
ingestor = PDFIngestor()
document = ingestor.ingest_pdf("document.pdf")
 
print(f"Extracted {len(document.content)} characters")
print(f"Pages: {document.metadata['page_count']}")
```
 
**Features:**
- Robust error handling for corrupted PDFs
- Metadata extraction (author, title, page count)
- Batch processing support
 
### 2. Text Chunking (`chunking.py`)
 
Splits documents into semantically meaningful chunks.
 
```python
from rag_assistant import TextChunker
from rag_assistant.chunking import ChunkingStrategy
 
# Fixed-size chunking
chunker = TextChunker(
    chunk_size=1000,
    chunk_overlap=200,
    strategy=ChunkingStrategy.FIXED_SIZE
)
 
chunks = chunker.chunk_document(document)
print(f"Created {len(chunks)} chunks")
 
# Get statistics
stats = chunker.get_chunk_stats(chunks)
print(f"Avg chunk size: {stats['avg_chunk_size']:.0f} chars")
```
 
**Chunking Strategies:**
- `FIXED_SIZE`: Simple character-based chunking (recommended)
- `SENTENCE`: Preserves sentence boundaries
- `PARAGRAPH`: Splits on paragraph breaks
 
### 3. Embeddings (`embeddings.py`)
 
Converts text chunks into dense vector representations.
 
```python
from rag_assistant import EmbeddingGenerator
 
embedder = EmbeddingGenerator(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"  # or "cuda" for GPU
)
 
# Embed chunks
embeddings, chunk_ids = embedder.embed_chunks(chunks)
print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
 
# Embed a query
query_embedding = embedder.embed_text("What is machine learning?")
```
 
**Recommended Models:**
- `all-MiniLM-L6-v2`: Fast, 384-dim (best for most use cases)
- `all-mpnet-base-v2`: High quality, 768-dim (slower but more accurate)
- `multi-qa-MiniLM-L6-cos-v1`: Optimized for Q&A tasks
 
### 4. Vector Store (`vector_store.py`)

Persistent storage and similarity search with ChromaDB.
 
```python
from rag_assistant import VectorStore
 
store = VectorStore(
    persist_directory="./data/vector_db",
    collection_name="my_documents"
)
 
# Add embeddings
store.add_embeddings(embeddings, chunks)
 
# Search
results = store.search(query_embedding, top_k=5)
 
# Filter by metadata
results = store.search(
    query_embedding,
    top_k=5,
    filter_metadata={"filename": "report.pdf"}
)
```
 
### 5. Pipeline (`pipeline.py`)
 
Orchestrates the complete workflow.
 
```python
from rag_assistant import RAGPipeline
 
pipeline = RAGPipeline()
 
# Process single PDF
result = pipeline.process_pdf("document.pdf")
 
# Process entire directory
result = pipeline.process_directory("./data/pdfs")
 
# Query with filters
results = pipeline.query(
    "What are the key findings?",
    top_k=3,
    filter_metadata={"author": "John Doe"}
)
 
# Get statistics
stats = pipeline.get_stats()
```
 
---
 
## ⚙️ Configuration
 
### Using Environment Variables
 
Create a `.env` file:
 
```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB_PATH=./data/vector_db
BATCH_SIZE=32
DEVICE=cpu
# LLM Settings (Phase 2)
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
OPENAI_API_KEY=your-key-here
```
 
### Programmatic Configuration
 
```python
from rag_assistant.config import RAGConfig
from rag_assistant import RAGPipeline
 
config = RAGConfig(
    chunk_size=500,
    chunk_overlap=100,
    embedding_model_name="all-mpnet-base-v2",
    batch_size=64,
    device="cuda"
)
 
pipeline = RAGPipeline(config=config)
```
 
---
 
## 🧪 Testing
 
Run the test suite:
 
```bash
# Install test dependencies
pip install pytest pytest-cov
 
# Run all tests
pytest tests/ -v
 
# Run with coverage
pytest tests/ --cov=src/rag_assistant --cov-report=html
```
 
---
 
## 📊 Performance Optimization
 
### GPU Acceleration
 
```python
pipeline = RAGPipeline(
    embedder=EmbeddingGenerator(device="cuda")
)
```
 
### Batch Processing
 
```python
config = RAGConfig(batch_size=128)  # Increase for faster processing
pipeline = RAGPipeline(config=config)
```
 
### Caching Embeddings
 
```python
from rag_assistant.embeddings import EmbeddingCache
 
cache = EmbeddingCache()
 
# Save embeddings
cache.save(chunk_ids, embeddings, cache_key="document_v1")
 
# Load later
cached_data = cache.load("document_v1")
```
 
---
 
## 🎓 Learning Outcomes
 
This project demonstrates:
 
1. **NLP Fundamentals**
   - Text preprocessing and chunking strategies
   - Semantic embeddings and similarity search
   - Document retrieval techniques
 
2. **ML Engineering**
   - Modular, production-ready code architecture
   - Configuration management with Pydantic
   - Comprehensive logging and error handling
 
3. **Modern ML Stack**
   - Vector databases (ChromaDB)
   - Sentence Transformers for embeddings
   - LangChain-compatible design
 
4. **Software Engineering Best Practices**
   - Type hints and dataclasses
   - Unit testing with pytest
   - Clean code principles (SOLID)
 
---
 
## 🚀 Next Steps (Phase 2+)
 
### Phase 2: LLM Integration
- [ ] Integrate OpenAI/Anthropic API for answer generation
- [ ] Implement prompt engineering templates
- [ ] Add conversation history management
 
### Phase 3: API & Deployment
- [ ] Build FastAPI REST API
- [ ] Create Streamlit/Gradio UI
- [ ] Dockerize the application
- [ ] Deploy to AWS/GCP/Azure
 
### Phase 4: Advanced Features
- [ ] Multi-document querying
- [ ] Hybrid search (keyword + semantic)
- [ ] Re-ranking with cross-encoders
- [ ] Streaming responses
 
### Phase 5: MLOps
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Model versioning (MLflow)
- [ ] Monitoring and observability
- [ ] A/B testing framework
 
---
 
## 📖 Additional Resources
 
### Documentation
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://docs.trychroma.com/)
- [LangChain](https://python.langchain.com/)
 
### Learning Materials
- [RAG Paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
 
---
 
## 🤝 Contributing
 
Contributions are welcome! Please:
 
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
 
---
 
## 📝 License
 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 
---
 
## 🙋 FAQ
 
**Q: Why ChromaDB instead of Pinecone/Weaviate?**
A: ChromaDB is free, runs locally, and perfect for learning. Easy to swap for production alternatives later.
 
**Q: Can I use this with other document types (Word, HTML, etc.)?**
A: Yes! Extend the `PDFIngestor` class or create new ingestors for other formats.
 
**Q: How do I improve retrieval quality?**
A: Try:
- Different chunking strategies
- Larger/better embedding models
- Adjust chunk size/overlap
- Add re-ranking with cross-encoders
 
**Q: Is this production-ready?**
A: This is Phase 1 (pipeline). For production, add API layer, authentication, rate limiting, monitoring, etc.
 
---
 
## 📧 Contact
 
For questions or feedback:
- Create an issue on GitHub
- Email: your.email@example.com
- LinkedIn: [Your Profile]
 ---

**Built with ❤️ for learning ML Engineering and NLP**