"""
RAG Pipeline - Orchestrates the complete document processing workflow.
 
This module ties together all components to provide an end-to-end
RAG document processing pipeline.
"""
 
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
 
from .ingestion import PDFIngestor
from .chunking import TextChunker, ChunkingStrategy
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .config import RAGConfig, get_config
 
 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
 
 
@dataclass
class PipelineResult:
    """Results from pipeline execution."""
 
    documents_processed: int
    chunks_created: int
    embeddings_stored: int
    processing_time: float
    errors: List[str]
 
    def __repr__(self) -> str:
        return (
            f"PipelineResult(\n"
            f"  documents_processed={self.documents_processed},\n"
            f"  chunks_created={self.chunks_created},\n"
            f"  embeddings_stored={self.embeddings_stored},\n"
            f"  processing_time={self.processing_time:.2f}s,\n"
            f"  errors={len(self.errors)}\n"
            f")"
        )
 
 
class RAGPipeline:
    """
    End-to-end RAG document processing pipeline.
 
    Orchestrates:
    1. PDF ingestion
    2. Text chunking
    3. Embedding generation
    4. Vector storage
 
    Example:
        >>> pipeline = RAGPipeline()
        >>> result = pipeline.process_pdf("document.pdf")
        >>> print(f"Processed {result.chunks_created} chunks")
 
        >>> # Query the processed documents
        >>> results = pipeline.query("What is the main topic?", top_k=3)
        >>> for result in results:
        ...     print(result['text'][:100])
    """
 
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        ingestor: Optional[PDFIngestor] = None,
        chunker: Optional[TextChunker] = None,
        embedder: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Initialize the RAG pipeline.
 
        Args:
            config: Configuration object (uses defaults if not provided)
            ingestor: Custom PDF ingestor (optional)
            chunker: Custom text chunker (optional)
            embedder: Custom embedding generator (optional)
            vector_store: Custom vector store (optional)
        """
        self.config = config or get_config()
 
        # Initialize components
        self.ingestor = ingestor or PDFIngestor()
 
        self.chunker = chunker or TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            strategy=ChunkingStrategy.FIXED_SIZE,
        )
 
        self.embedder = embedder or EmbeddingGenerator(
            model_name=self.config.embedding_model_name,
            device=self.config.device,
            batch_size=self.config.batch_size,
        )
 
        self.vector_store = vector_store or VectorStore(
            persist_directory=self.config.vector_db_path,
            collection_name=self.config.collection_name,
        )
 
        logger.info("RAG Pipeline initialized successfully")
 
    def process_pdf(self, pdf_path: str | Path) -> PipelineResult:
        """
        Process a single PDF through the complete pipeline.
 
        Args:
            pdf_path: Path to PDF file
 
        Returns:
            PipelineResult with processing statistics
        """
        import time
        start_time = time.time()
 
        errors = []
 
        try:
            # Step 1: Ingest PDF
            logger.info(f"Step 1/4: Ingesting PDF - {pdf_path}")
            document = self.ingestor.ingest_pdf(pdf_path)
 
            # Step 2: Chunk text
            logger.info("Step 2/4: Chunking text")
            chunks = self.chunker.chunk_document(document)
 
            if not chunks:
                raise ValueError("No chunks created from document")
 
            # Step 3: Generate embeddings
            logger.info("Step 3/4: Generating embeddings")
            embeddings, chunk_ids = self.embedder.embed_chunks(chunks)
 
            # Step 4: Store in vector database
            logger.info("Step 4/4: Storing embeddings in vector database")
            self.vector_store.add_embeddings(embeddings, chunks)
 
            processing_time = time.time() - start_time
 
            result = PipelineResult(
                documents_processed=1,
                chunks_created=len(chunks),
                embeddings_stored=len(embeddings),
                processing_time=processing_time,
                errors=errors,
            )
 
            logger.info(f"Pipeline completed successfully in {processing_time:.2f}s")
            logger.info(str(result))
 
            return result
 
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
 
            return PipelineResult(
                documents_processed=0,
                chunks_created=0,
                embeddings_stored=0,
                processing_time=processing_time,
                errors=errors,
            )
 
    def process_directory(self, directory_path: str | Path) -> PipelineResult:
        """
        Process all PDFs in a directory.
 
        Args:
            directory_path: Path to directory containing PDFs
 
        Returns:
            PipelineResult with aggregate statistics
        """
        import time
        start_time = time.time()
 
        directory_path = Path(directory_path)
        pdf_files = list(directory_path.glob("*.pdf"))
 
        logger.info(f"Found {len(pdf_files)} PDF files to process")
 
        total_docs = 0
        total_chunks = 0
        total_embeddings = 0
        errors = []
 
        for pdf_file in pdf_files:
            logger.info(f"\nProcessing {pdf_file.name}...")
 
            try:
                result = self.process_pdf(pdf_file)
 
                total_docs += result.documents_processed
                total_chunks += result.chunks_created
                total_embeddings += result.embeddings_stored
                errors.extend(result.errors)
 
            except Exception as e:
                error_msg = f"Failed to process {pdf_file.name}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
 
        processing_time = time.time() - start_time
 
        result = PipelineResult(
            documents_processed=total_docs,
            chunks_created=total_chunks,
            embeddings_stored=total_embeddings,
            processing_time=processing_time,
            errors=errors,
        )
 
        logger.info("\n" + "="*60)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(str(result))
        logger.info("="*60)
 
        return result
 
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        return_embeddings: bool = False
    ) -> List[Dict]:
        """
        Query the vector store for relevant document chunks.
 
        Args:
            query_text: Query string
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            return_embeddings: Whether to include embeddings in results
 
        Returns:
            List of result dictionaries sorted by relevance
        """
        logger.info(f"Querying: '{query_text}' (top_k={top_k})")
 
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query_text)
 
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
 
        logger.info(f"Found {len(results)} relevant chunks")
 
        if return_embeddings:
            for result in results:
                result['query_embedding'] = query_embedding
 
        return results
 
    def get_stats(self) -> Dict:
        """
        Get statistics about the pipeline and stored data.
 
        Returns:
            Dictionary with pipeline statistics
        """
        vector_stats = self.vector_store.get_collection_stats()
        model_info = self.embedder.get_model_info()
 
        return {
            "pipeline": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "batch_size": self.config.batch_size,
            },
            "model": model_info,
            "vector_store": vector_stats,
        }
 
    def clear_all_data(self) -> None:
        """
        Clear all data from the vector store.
 
        WARNING: This is destructive and cannot be undone.
        """
        logger.warning("Clearing all pipeline data...")
        self.vector_store.reset()
        logger.info("All data cleared successfully")
 
    def __repr__(self) -> str:
        stats = self.vector_store.get_collection_stats()
        return (
            f"RAGPipeline(\n"
            f"  documents={stats['total_documents']},\n"
            f"  model={self.config.embedding_model_name},\n"
            f"  chunk_size={self.config.chunk_size}\n"
            f")"
        )