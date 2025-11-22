"""
Unit tests for RAG Pipeline components.
 
Run with: pytest tests/test_pipeline.py -v
"""
 
import pytest
from pathlib import Path
import sys
import tempfile
 
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
 
from rag_assistant import (
    PDFIngestor,
    TextChunker,
    EmbeddingGenerator,
    VectorStore,
    RAGPipeline,
)
from rag_assistant.chunking import ChunkingStrategy, Chunk 
 
class TestPDFIngestor:
    """Test PDF ingestion functionality."""
 
    def test_ingestor_initialization(self):
        """Test that PDFIngestor initializes correctly."""
        ingestor = PDFIngestor()
        assert ingestor is not None
        assert ingestor.encoding == "utf-8"
 
    def test_invalid_file_raises_error(self):
        """Test that non-existent file raises FileNotFoundError."""
        ingestor = PDFIngestor()
 
        with pytest.raises(FileNotFoundError):
            ingestor.ingest_pdf("nonexistent.pdf")
 
    def test_non_pdf_file_raises_error(self):
        """Test that non-PDF file raises ValueError."""
        ingestor = PDFIngestor()
 
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            with pytest.raises(ValueError):
                ingestor.ingest_pdf(f.name)
 
 
class TestTextChunker:
    """Test text chunking functionality."""
 
    def test_chunker_initialization(self):
        """Test that TextChunker initializes with correct parameters."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
 
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.strategy == ChunkingStrategy.FIXED_SIZE
 
    def test_invalid_overlap_raises_error(self):
        """Test that overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)
 
    def test_fixed_size_chunking(self):
        """Test fixed-size chunking strategy."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
 
        text = "This is a test. " * 50  # ~750 characters
        chunks = chunker.chunk_text(text, source="test")
 
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.text) <= 100 for chunk in chunks)
 
    def test_sentence_chunking(self):
        """Test sentence-based chunking strategy."""
        chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=20,
            strategy=ChunkingStrategy.SENTENCE
        )
 
        text = "First sentence. Second sentence. Third sentence. " * 10
        chunks = chunker.chunk_text(text, source="test")
 
        assert len(chunks) > 0
 
    def test_paragraph_chunking(self):
        """Test paragraph-based chunking strategy."""
        chunker = TextChunker(
            chunk_size=200,
            chunk_overlap=50,
            strategy=ChunkingStrategy.PARAGRAPH
        )
 
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunker.chunk_text(text, source="test")
 
        assert len(chunks) > 0
 
    def test_chunk_stats(self):
        """Test chunk statistics calculation."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
 
        text = "Test text. " * 100
        chunks = chunker.chunk_text(text, source="test")
 
        stats = chunker.get_chunk_stats(chunks)
 
        assert "total_chunks" in stats
        assert "avg_chunk_size" in stats
        assert stats["total_chunks"] == len(chunks)
 
 
class TestEmbeddingGenerator:
    """Test embedding generation functionality."""
 
    @pytest.fixture
    def embedder(self):
        """Create an embedding generator for tests."""
        return EmbeddingGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
 
    def test_embedder_initialization(self, embedder):
        """Test that EmbeddingGenerator initializes correctly."""
        assert embedder is not None
        assert embedder.embedding_dim == 384  # MiniLM-L6 dimension
 
    def test_embed_single_text(self, embedder):
        """Test embedding a single text."""
        text = "This is a test sentence."
        embedding = embedder.embed_text(text)
 
        assert embedding.shape == (384,)
        assert embedding.dtype.kind == 'f'  # Float type
 
    def test_embed_batch(self, embedder):
        """Test embedding multiple texts."""
        texts = ["First text.", "Second text.", "Third text."]
        embeddings = embedder.embed_batch(texts)
 
        assert embeddings.shape == (3, 384)
 
    def test_embed_chunks(self, embedder):
        """Test embedding chunks."""
        chunks = [
            Chunk(
                text=f"Chunk {i}",
                metadata={"index": i},
                chunk_id=f"chunk_{i}",
                start_char=i*10,
                end_char=(i+1)*10
            )
            for i in range(3)
        ]
 
        embeddings, chunk_ids = embedder.embed_chunks(chunks, show_progress=False)
 
        assert embeddings.shape == (3, 384)
        assert len(chunk_ids) == 3
        assert chunk_ids == ["chunk_0", "chunk_1", "chunk_2"]
 
    def test_compute_similarity(self, embedder):
        """Test similarity computation."""
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a mat."
        text3 = "The weather is nice today."
 
        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)
        emb3 = embedder.embed_text(text3)
 
        # Similar texts should have higher similarity
        sim_12 = embedder.compute_similarity(emb1, emb2)
        sim_13 = embedder.compute_similarity(emb1, emb3)
 
        assert sim_12 > sim_13  # text1 and text2 are more similar
 
    def test_model_info(self, embedder):
        """Test getting model information."""
        info = embedder.get_model_info()
 
        assert "model_name" in info
        assert "embedding_dimension" in info
        assert info["embedding_dimension"] == 384
 
 
class TestVectorStore:
    """Test vector store functionality."""
 
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(
                persist_directory=tmpdir,
                collection_name="test_collection"
            )
            yield store
 
    def test_vector_store_initialization(self, temp_db):
        """Test that VectorStore initializes correctly."""
        assert temp_db is not None
        assert temp_db.collection_name == "test_collection"
 
    def test_add_and_retrieve_embeddings(self, temp_db):
        """Test adding and retrieving embeddings."""
        import numpy as np
 
        # Create dummy data
        embeddings = np.random.rand(3, 384).astype(np.float32)
        chunks = [
            Chunk(
                text=f"Test chunk {i}",
                metadata={"index": i},
                chunk_id=f"test_{i}",
                start_char=i*10,
                end_char=(i+1)*10
            )
            for i in range(3)
        ]
 
        # Add embeddings
        temp_db.add_embeddings(embeddings, chunks)
 
        # Verify count
        assert temp_db.collection.count() == 3
 
    def test_search(self, temp_db):
        """Test similarity search."""
        import numpy as np
 
        # Add test data
        embeddings = np.random.rand(5, 384).astype(np.float32)
        chunks = [
            Chunk(
                text=f"Document {i}",
                metadata={"doc_id": i},
                chunk_id=f"doc_{i}",
                start_char=0,
                end_char=10
            )
            for i in range(5)
        ]
 
        temp_db.add_embeddings(embeddings, chunks)
 
        # Search
        query_embedding = np.random.rand(384).astype(np.float32)
        results = temp_db.search(query_embedding, top_k=3)
 
        assert len(results) == 3
        assert all('text' in r for r in results)
        assert all('metadata' in r for r in results)
 
    def test_collection_stats(self, temp_db):
        """Test getting collection statistics."""
        stats = temp_db.get_collection_stats()
 
        assert "collection_name" in stats
        assert "total_documents" in stats
        assert stats["collection_name"] == "test_collection"
 
 
class TestRAGPipeline:
    """Test end-to-end RAG pipeline."""
 
    @pytest.fixture
    def temp_pipeline(self):
        """Create a pipeline with temporary storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_store = VectorStore(
                persist_directory=tmpdir,
                collection_name="test_pipeline"
            )
 
            pipeline = RAGPipeline(vector_store=vector_store)
            yield pipeline
 
    def test_pipeline_initialization(self, temp_pipeline):
        """Test that RAGPipeline initializes correctly."""
        assert temp_pipeline is not None
        assert temp_pipeline.ingestor is not None
        assert temp_pipeline.chunker is not None
        assert temp_pipeline.embedder is not None
        assert temp_pipeline.vector_store is not None
 
    def test_pipeline_stats(self, temp_pipeline):
        """Test getting pipeline statistics."""
        stats = temp_pipeline.get_stats()
 
        assert "pipeline" in stats
        assert "model" in stats
        assert "vector_store" in stats
 
 
if __name__ == "__main__":
    pytest.main([__file__, "-v"])