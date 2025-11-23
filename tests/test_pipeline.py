"""
Unit tests for RAG Pipeline components.

Run with: pytest tests/test_pipeline.py -v
"""

import pytest
from pathlib import Path
import sys
import tempfile
import shutil
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_assistant import (
    PDFIngestor,
    TextChunker,
    EmbeddingGenerator,
    VectorStore,
    RAGPipeline,
)
from rag_assistant.chunking import ChunkingStrategy, Chunk


# --- Helper for Windows Cleanup ---
def robust_rmtree(path):
    """
    Robustly remove a directory, ignoring errors.
    Useful for Windows where file locks (like ChromaDB's) prevent deletion.
    """
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


class TestPDFIngestor:
    """Test PDF ingestion functionality."""

    def test_ingestor_initialization(self):
        ingestor = PDFIngestor()
        assert ingestor is not None
        assert ingestor.encoding == "utf-8"

    def test_invalid_file_raises_error(self):
        ingestor = PDFIngestor()
        with pytest.raises(FileNotFoundError):
            ingestor.ingest_pdf("nonexistent.pdf")

    def test_non_pdf_file_raises_error(self):
        ingestor = PDFIngestor()
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            with pytest.raises(ValueError):
                ingestor.ingest_pdf(f.name)


class TestTextChunker:
    """Test text chunking functionality."""

    def test_chunker_initialization(self):
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.strategy == ChunkingStrategy.FIXED_SIZE

    def test_invalid_overlap_raises_error(self):
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_fixed_size_chunking(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 50
        chunks = chunker.chunk_text(text, source="test")
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.text) <= 100 for chunk in chunks)

    def test_sentence_chunking(self):
        chunker = TextChunker(
            chunk_size=100, chunk_overlap=20, strategy=ChunkingStrategy.SENTENCE
        )
        text = "First sentence. Second sentence. Third sentence. " * 10
        chunks = chunker.chunk_text(text, source="test")
        assert len(chunks) > 0

    def test_paragraph_chunking(self):
        chunker = TextChunker(
            chunk_size=200, chunk_overlap=50, strategy=ChunkingStrategy.PARAGRAPH
        )
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunker.chunk_text(text, source="test")
        assert len(chunks) > 0

    def test_chunk_stats(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "Test text. " * 100
        chunks = chunker.chunk_text(text, source="test")
        stats = chunker.get_chunk_stats(chunks)
        assert "total_chunks" in stats
        assert "avg_chunk_size" in stats


class TestEmbeddingGenerator:
    """Test embedding generation functionality."""

    @pytest.fixture
    def embedder(self):
        return EmbeddingGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )

    def test_embedder_initialization(self, embedder):
        assert embedder is not None
        assert embedder.embedding_dim == 384

    def test_embed_single_text(self, embedder):
        text = "This is a test sentence."
        embedding = embedder.embed_text(text)
        assert embedding.shape == (384,)
        assert embedding.dtype.kind == "f"

    def test_embed_batch(self, embedder):
        texts = ["First text.", "Second text.", "Third text."]
        embeddings = embedder.embed_batch(texts)
        assert embeddings.shape == (3, 384)

    def test_embed_chunks(self, embedder):
        chunks = [
            Chunk(
                text=f"Chunk {i}",
                metadata={"index": i},
                chunk_id=f"chunk_{i}",
                start_char=i * 10,
                end_char=(i + 1) * 10,
            )
            for i in range(3)
        ]
        embeddings, chunk_ids = embedder.embed_chunks(chunks, show_progress=False)
        assert embeddings.shape == (3, 384)
        assert len(chunk_ids) == 3

    def test_compute_similarity(self, embedder):
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a mat."
        text3 = "The weather is nice today."
        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)
        emb3 = embedder.embed_text(text3)
        sim_12 = embedder.compute_similarity(emb1, emb2)
        sim_13 = embedder.compute_similarity(emb1, emb3)
        assert sim_12 > sim_13

    def test_model_info(self, embedder):
        info = embedder.get_model_info()
        assert "model_name" in info
        assert "embedding_dimension" in info
        assert info["embedding_dimension"] == 384


class TestVectorStore:
    """Test vector store functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing with robust cleanup."""
        # Manually create directory instead of using context manager
        tmpdir = tempfile.mkdtemp()
        store = VectorStore(
            persist_directory=tmpdir, collection_name="test_collection"
        )
        yield store
        
        # Teardown
        store.close()
        # Robust cleanup that ignores Windows file lock errors
        robust_rmtree(tmpdir)

    def test_vector_store_initialization(self, temp_db):
        assert temp_db is not None
        assert temp_db.collection_name == "test_collection"

    def test_add_and_retrieve_embeddings(self, temp_db):
        import numpy as np
        embeddings = np.random.rand(3, 384).astype(np.float32)
        chunks = [
            Chunk(
                text=f"Test chunk {i}",
                metadata={"index": i},
                chunk_id=f"test_{i}",
                start_char=i * 10,
                end_char=(i + 1) * 10,
            )
            for i in range(3)
        ]
        temp_db.add_embeddings(embeddings, chunks)
        assert temp_db.collection.count() == 3

    def test_search(self, temp_db):
        import numpy as np
        embeddings = np.random.rand(5, 384).astype(np.float32)
        chunks = [
            Chunk(
                text=f"Document {i}",
                metadata={"doc_id": i},
                chunk_id=f"doc_{i}",
                start_char=0,
                end_char=10,
            )
            for i in range(5)
        ]
        temp_db.add_embeddings(embeddings, chunks)
        query_embedding = np.random.rand(384).astype(np.float32)
        results = temp_db.search(query_embedding, top_k=3)
        assert len(results) == 3
        assert all("text" in r for r in results)
        assert all("metadata" in r for r in results)

    def test_collection_stats(self, temp_db):
        stats = temp_db.get_collection_stats()
        assert "collection_name" in stats
        assert "total_documents" in stats
        assert stats["collection_name"] == "test_collection"


class TestRAGPipeline:
    """Test end-to-end RAG pipeline."""

    @pytest.fixture
    def temp_pipeline(self):
        """Create a pipeline with temporary storage and MOCKED generator."""
        # Manually create directory
        tmpdir = tempfile.mkdtemp()
        vector_store = VectorStore(
            persist_directory=tmpdir, collection_name="test_pipeline"
        )

        # --- KEY FIX: Mock the LLM Generator ---
        # This prevents the pipeline from trying to connect to OpenAI/Ollama
        mock_generator = MagicMock()
        mock_generator.get_model_info.return_value = {
            "provider": "mock",
            "model_name": "test-model"
        }
        mock_generator.generate.return_value = "This is a mock answer."

        # Initialize pipeline with the mock generator
        pipeline = RAGPipeline(
            vector_store=vector_store,
            generator=mock_generator
        )
        
        yield pipeline
        
        # Teardown
        vector_store.close()
        robust_rmtree(tmpdir)

    def test_pipeline_initialization(self, temp_pipeline):
        assert temp_pipeline is not None
        assert temp_pipeline.ingestor is not None
        assert temp_pipeline.chunker is not None
        assert temp_pipeline.embedder is not None
        assert temp_pipeline.vector_store is not None
        # Ensure generator is our mock
        assert isinstance(temp_pipeline.generator, MagicMock)

    def test_pipeline_stats(self, temp_pipeline):
        stats = temp_pipeline.get_stats()
        assert "pipeline" in stats
        assert "embedding_model" in stats  # Note: key might be 'embedding_model' or 'model' depending on your implementation
        assert "vector_store" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])