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

from .ingestion import PDFIngestor
from .chunking import TextChunker
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .pipeline import RAGPipeline
from .config import RagConfig

__all__ = [
    "PDFIngestor",
    "TextChunker",
    "EmbeddingGenerator",
    "VectorStore",
    "RAGPipeline",
    "RAGConfig",
]