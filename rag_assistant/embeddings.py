"""
Embeddings Generation Module - Converts text chunks into dense vector representations.

Uses sentence-transformers for generation semantic embeddings optimized for similarity search and retrieval tasks.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from .chunking import Chunk


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates semantic embeddings for text chunks using pre-trained models.

    Features:
    - Bath processing for efficiency.
    - GPU acceleration support.
    - Multiple model options.
    - Normalized embeddings for cosine similarity.

    Example:
        >>> generator = EmbeddingGenerator()
        >>> embeddings = generator.embed_chunks(chunks)
        >>> print(f"Generated {len(embeddings)} embeddings")
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None, batch_size: int = 32, normalize_embeddings: bool = True,):
        """
        Initialize the embedding generator.

        Args:
            model_name: HuggingFace model identifier
                - all-MiniLM-L6-v2: Fast, 384-dim (recommended for most use case)
                - all-mpnet-base-v2: Higher quality, 768-dim (slower but more accurate)
                - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A tasks
            device: Device to run model on ('cpu', 'cuda', or None for auto-detect)
            batch_size: Number of texts to process in parallel
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model_name = model_name

        logger.info(f"Loading embedding model: {model_name} on {device}")

        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def embed_chunks(self, chunks: List[Chunk], show_progress: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of text chunks to embed.
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (embeddings array, chunk IDs)
            - embeddings: numpy array of shape (n_chunks, embedding_dim)
            - chunk_ids: List of chunk identifiers
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return np.array([]), []

        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        logger.info(f"Generating embeddings for {len(texts)} chunks...")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )

            logger.info(f"Successfully generated {len(embeddings)} embeddings with shape {embeddings.shape}")

            return embeddings, chunk_ids

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text (e.g., a query).

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )

            return embedding

        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Generate embedding for a batch of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            raise

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1 (higher is more similar)
        """
        # If embeddings are normalized, dot product = cosine similarity
        if self.normalize_embeddings:
            return float(np.dot(embedding1, embedding2))

        # Otherwise, compute cosine similarity manually
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalized": self.normalize_embeddings,
            "max_seq_length": self.model.max_seq_length,
        }


    @staticmethod
    def list_recommended_models() -> dict:
        """
        Get a list of recommended models for different use cases.

        Returns:
            Dictionary mapping use case to model name
        """
        return {
            "fast_general": "sentence-transformers/all-MiniLM-L6-v2",
            "high_quality": "sentence-transformers/all-mpnet-base-v2",
            "qa_optimized": "sentence-transformers/mutli-qa-MiniLM-L6-cos-v1",
            "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "legal_financial": "sentece-transformers/all-mpnet-base-v2",
        }


class EmbeddingCache:
    """
    Simple cache for storing and retrieving embeddings to avoid recomputation.

    This is useful when processing the same documents multiple times.
    """

    def __init__(self, cache_dir: str = "./data/embedding_cache"):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Embedding cache initialized at {self.cache_dir}")

    def save(self, chunk_ids: List[str], embeddings: np.ndarray, cache_key: str) -> None:
        """
        Save embeddings to cache.

        Args:
            chunk_ids: List of chunk identifiers
            embeddings: Array of embeddings
            cache_key: Unique key for this embedding set
        """
        cache_file = self.cache_dir / f"{cache_key}.npz"

        np.savez(
            cache_file,
            embeddings=embeddings,
            chunk_ids=chunk_ids
        )

        logger.info(f"Saved {len(embeddings)} embeddings to cache: {cache_key}")

    def load(self, cache_key: str) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        Load embeddings from cache.

        Args:
            cache_key: Cache key to retrieve.

        Returns:
            Tuple of (embeddings, chunk_ids) if found, None otherwise
        """
        cache_file = self.cache_dir / f"{cache_key}.npz"

        if not cache_file.exists():
            return None

        try:
            data = np.load(cache_file, allow_pickle=True)
            embeddings = data['embeddings']
            chunk_ids = data['chunk_ids'].tolist()

            logger.info(f"Loaded {len(embeddings)} embeddings from cache: {cache_key}")
            return embeddings, chunk_ids

        except Exception as e:
            logger.error(f"Failed to load cache {cache_key}: {e}")
            return None

    def clear(self) -> None:
        """Delete all cached embeddings."""
        import shutil
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Embedding cache cleared")