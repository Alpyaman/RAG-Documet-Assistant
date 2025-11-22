"""
Text Chunking Module - Splits text into manageable chunks for embedding and retrieval.

Provides multiple chunking strategies optimized for different document types and retrieval patterns.
"""

import logging
import re
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum

from .ingestion import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Available text chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    metadata: Dict[str, any]
    chunk_id: str
    start_char: int
    end_char: int

    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Chunk(id={self.chunk_id}, length={len(self.text)}, preview={preview})"
    
class TextChunker:
    """
    Splits documents into chunks for embedding and retrieval.

    Supports multiple chunking strategies:
    - Fixed size: Simple character-based chunking with overlap
    - Sentence: Split on sentence boundaries
    - Paragraph: Split on paragraph boundaries
    - Semantic: Intelligent splitting based on content structure

    Example:
        >>> chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        >>> chunks = chunker.chunk_document(documnet)
        >>> print(f"Created {len(chunks)} chunks")
    """

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Number of overlapping characters between chunks
            strategy: Chunking strategy to use
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        logger.info(f"TextChunker initialized with strategy={strategy.value}, chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks based on the configured strategy.
        
        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(document)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            chunks = self._chunk_by_sentence(document)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._chunk_by_paragraph(document)
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented yet")
        
        logger.info(f"Created {len(chunks)} chunks from document (source: {document.metadata.get('filename', 'unknown')})")

        return chunks
    
    def chunk_text(self, text: str, source: str = "text") -> List[Chunk]:
        """
        Chunk raw text directly (convenience method).

        Args:
            text: Raw text to chunk
            source: Source identifier for metadata

        Returns:
            List of Chunk objects
        """
        doc = Document(
            content=text,
            metadata={"source": source},
            source=source
        )
        return self.chunk_document(doc)
    
    def _chunk_fixed_size(self, document: Document) -> List[Chunk]:
        """
        Split text into fixed-size chunks with overlap.

        This is the most straightforward approach and works well for most use cases.
        """
        text = document.content
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Extract chunk
            chunk_text = text[start:end]

            # Create chunk metadata
            chunk_metadata = {
                **document.metadata,
                "chunk_index": chunk_index,
                "chunk_strategy": "fixed_size",
                "total_chars": len(chunk_text),
            }

            chunk_id = f"{document.metadata.get('filename', 'doc')}_{chunk_index}"

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    start_char=start,
                    end_char=end,
                )
            )

            # Move to next chunk with overlap
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1

        return chunks
    
    def _chunk_by_sentence(self, document: Document) -> List[Chunk]:
        """
        Split text into chunks at sentence boundaries.
        Preserves sentence integrity for better sementic coherence.
        """
        text = document.content

        # Simple sentence splitting (can be improved with spacy/NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        char_position = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk_size, create new chunk
            if current_size + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_id = f"{document.metadata.get('filename', 'doc')}_{chunk_index}"

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_index,
                            "chunk_strategy": "sentence",
                            "sentence_count": len(current_chunk),
                        },
                        chunk_id=chunk_id,
                        start_char=char_position - current_size,
                        end_char=char_position
                    )
                )

                # Handle overlap: keep last few sentences
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break

                current_chunk = overlap_sentences
                current_size = overlap_size
                chunk_index += 1
            
            current_chunk.append(sentence)
            current_size += sentence_len
            char_position += sentence_len

        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = f"{document.metadata.get('filename', 'doc')}_{chunk_index}"

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "chunk_strategy": "sentence",
                        "sentence_count": len(current_chunk),
                    },
                    chunk_id=chunk_id,
                    start_char=char_position - current_size,
                    end_char=char_position,
                )
            )
        
        return chunks
    
    def _chunk_by_paragraph(self, document: Document) -> List[Chunk]:
        """
        Split text into chunks at paragraph boundaries.
        Best for well-structured documents with clear paragraph breaks.
        """
        text = document.content

        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\ns*\n', text)

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        char_position = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            para_len = len(paragraph)

            # If paragraph alone exceeds chunk_size, split it
            if para_len > self.chunk_size:
                # Fallback to fixed-size chunking for this paragraph
                para_doc = Document(
                    content=paragraph,
                    metadata=document.metadata,
                    source=document.source
                )
                para_chunks = self._chunk_fixed_size(para_doc)
                chunks.extend(para_chunks)
                continue

            # If adding this paragraph exceeds chunk_size, create new chunk
            if current_size + para_len > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_id = f"{document.metadata.get('filename', 'doc')}_{chunk_index}"

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_index,
                            "chunk_strategy": "paragraph",
                            "paragraph_count": len(current_chunk),
                        },
                        chunk_id=chunk_id,
                        start_char=char_position - current_size,
                        end_char=char_position,
                    )
                )

                current_chunk = []
                current_size = 0
                chunk_index += 1

            current_chunk.append(paragraph)
            current_size += para_len
            char_position += para_len

        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_id = f"{document.metadata.get('filename', 'doc')}_{chunk_index}"

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "chunk_strategy": "paragraph",
                        "paragraph_count": len(current_chunk),
                    },
                    chunk_id=chunk_id,
                    start_char=char_position - current_size,
                    end_char=char_position,
                )
            )

        return chunks
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, any]:
        """
        Calculate statistics about the chunks.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary with statistics.
        """
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.text) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_characters": sum(chunk_sizes),
            "strategy": chunks[0].metadata.get("chunk_strategy", "unknown"),
        }