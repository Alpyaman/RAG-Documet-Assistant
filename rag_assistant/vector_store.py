"""
Vector Store Module - Manages document embeddings in ChromaDB.

Provides persistent storage and efficient similarity search for
document embeddings using ChromaDB.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from .chunking import Chunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector embeddings storage and retrieval using ChromaDB.

    Features:
    - Persistent storage of embeddings
    - Efficient similarity search
    - Metadata filtering
    - Batch operations
    - Collection management

    Example:
        >>> store = VectorStore()
        >>> store.add_embeddings(embeddings, chunks)
        >>> results = store.search("What is the main topic?", top_k=5)
    """

    def __init__(
        self,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "document_embeddings",
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name

        logger.info(f"Initializing ChromaDB at {self.persist_directory}")

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"description": "RAG document embeddings"}
        )

        logger.info(
            f"Vector store initialized. "
            f"Collection: {collection_name}, "
            f"Documents: {self.collection.count()}"
        )

    def add_embeddings(
        self, embeddings: np.ndarray, chunks: List[Chunk], batch_size: int = 100
    ) -> None:
        """
        Add embeddings and their associated chunks to the vector store.

        Args:
            embeddings: Array of embedding vectors (n_chunks, embedding_dim)
            chunks: List of Chunk objects
            batch_size: Number of embeddings to add per batch
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings but {len(chunks)} chunks"
            )

        if len(chunks) == 0:
            logger.warning("No chunks to add")
            return

        logger.info(f"Adding {len(chunks)} embeddings to vector store...")

        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Convert numpy array to list of lists for ChromaDB
        embeddings_list = embeddings.tolist()

        # Add in batches to avoid memory issues
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))

            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings_list[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

            logger.debug(f"Added batch {i//batch_size + 1}: {batch_end - i} documents")

        logger.info(
            f"Successfully added {len(chunks)} documents. "
            f"Total in collection: {self.collection.count()}"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search for similar documents using a query embedding.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"filename": "report.pdf"})

        Returns:
            List of result dictionaries with 'id', 'text', 'metadata', 'distance'
        """
        # Convert numpy array to list for ChromaDB
        query_list = query_embedding.tolist()

        # Build where clause for metadata filtering
        where = filter_metadata if filter_metadata else None

        try:
            results = self.collection.query(
                query_embeddings=[query_list],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "similarity": 1
                        - results["distances"][0][i],  # Convert distance to similarity
                    }
                )

            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def search_by_text(
        self, query_text: str, top_k: int = 5, filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search using text (ChromaDB will handle embedding internally).

        Args:
            query_text: Query string
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of result dictionaries
        """
        where = filter_metadata if filter_metadata else None

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "similarity": 1 - results["distances"][0][i],
                    }
                )

            logger.info(f"Text search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            raise

    def get_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a specific chunk by its ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Dictionary with chunk data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[chunk_id], include=["documents", "metadatas", "embeddings"]
            )

            if not results["ids"]:
                return None

            return {
                "id": results["ids"][0],
                "text": results["documents"][0],
                "metadata": results["metadatas"][0],
                "embedding": results["embeddings"][0],
            }

        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None

    def delete_by_metadata(self, metadata_filter: Dict) -> int:
        """
        Delete documents matching metadata filter.

        Args:
            metadata_filter: Metadata to match (e.g., {"filename": "old.pdf"})

        Returns:
            Number of documents deleted
        """
        try:
            # First get IDs matching the filter
            results = self.collection.get(where=metadata_filter, include=[])

            ids_to_delete = results["ids"]

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents")

            return len(ids_to_delete)

        except Exception as e:
            logger.error(f"Failed to delete by metadata: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the entire collection and all its data."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")

            # Recreate empty collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document embeddings"},
            )

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()

        # Sample a few documents to get metadata info
        sample = self.collection.peek(limit=min(10, count))

        unique_files = set()
        if sample["metadatas"]:
            for metadata in sample["metadatas"]:
                if "filename" in metadata:
                    unique_files.add(metadata["filename"])

        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "persist_directory": str(self.persist_directory),
            "sample_unique_files": list(unique_files),
        }

    def list_documents(self, limit: int = 10) -> List[Dict]:
        """
        List documents in the collection.

        Args:
            limit: Maximum number of documents to return

        Returns:
            List of document dictionaries
        """
        try:
            results = self.collection.peek(limit=limit)

            documents = []
            for i in range(len(results["ids"])):
                documents.append(
                    {
                        "id": results["ids"][i],
                        "text_preview": results["documents"][i][:100] + "...",
                        "metadata": results["metadatas"][i],
                    }
                )

            return documents

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def reset(self) -> None:
        """
        Reset the vector store (clear all data).

        WARNING: This deletes all stored embeddings.
        """
        logger.warning("Resetting vector store - all data will be lost!")
        self.delete_collection()

    def close(self) -> None:
        """
        Close the ChromaDB client and release resources.

        This is important for proper cleanup, especially on Windows
        where file locks can prevent cleanup of temporary directories.
        """
        try:
            import gc
            import time

            # ChromaDB PersistentClient doesn't have an explicit close,
            # but we can delete our references to allow garbage collection
            if hasattr(self, "collection"):
                del self.collection
            if hasattr(self, "client"):
                del self.client

            # Force garbage collection to release file handles immediately.
            gc.collect()

            # Small delay to ensure file handles are released (Windows issue)
            time.sleep(0.1)

            logger.debug("VectorStore closed")
        except Exception as e:
            logger.warning(f"Error during close: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup

    def __repr__(self) -> str:
        count = self.collection.count()
        return f"VectorStore(collection='{self.collection_name}', documents={count})"
