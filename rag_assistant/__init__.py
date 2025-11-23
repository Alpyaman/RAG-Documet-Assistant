"""
RAG Document Assistant - An end-to-end Retrieval-Augmented Generation system.
This package provides tools for building a document Q&A system using:
- PDF ingestion and processing
- Text chunking strategies
- Semantic embeddings
- Vector database storage (ChromaDB)
"""

__version__ = "0.1.0"
__author__ = "Alp Yaman"

from .chunking import TextChunker
from .config import RagConfig
from .embeddings import EmbeddingGenerator
from .ingestion import PDFIngestor
from .pipeline import RAGPipeline
from .vector_store import VectorStore

__all__ = [
    "PDFIngestor",
    "TextChunker",
    "EmbeddingGenerator",
    "VectorStore",
    "RAGPipeline",
    "RagConfig",
]
