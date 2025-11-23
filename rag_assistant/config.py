"""
Configuration settings for RAG Document Assistant.
Uses Pydantic for validation and environment variable management.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class RagConfig(BaseSettings):
    """Configuration for RAG pipeline components."""

    # Chunking Configuration
    chunk_size: int = Field(
        default=1000, description="Maximum characters per text chunk.", ge=100, le=5000
    )

    chunk_overlap: int = Field(
        default=200,
        description="Number of overlapping characters between chunks.",
        ge=0,
        le=500,
    )

    # Embedding Model Configuration
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Hugging Face model for embeddings.",
    )

    embedding_dimension: int = Field(
        default=384, description="Dimension of embedding vectors."
    )

    # Vector Store Configuration
    vector_db_path: str = Field(
        default="./data/vector_db", description="Path to ChromaDB storage"
    )

    collection_name: str = Field(
        default="document_embeddings", description="Name of ChromaDB collection."
    )

    # Processing Configuration
    batch_size: int = Field(
        default=32, description="Batch size for embedding generation."
    )

    device: str = Field(
        default="cpu", description="Device for model inference (e.g., 'cpu', 'cuda')."
    )

    # Data Paths
    pdf_storage_path: str = Field(
        default="./data/pdfs", description="Directory for uploaded PDFs."
    )

    # LLM Generation Configuration
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: 'openai', 'ollama', or 'huggingface'",
    )

    llm_model_name: str = Field(
        default="gpt-3.5-turbo", description="Model name for the LLM provider."
    )

    llm_temperature: float = Field(
        default=0.7,
        description="Sampling temperature for generation (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
    )

    llm_max_tokens: int = Field(
        default=512, description="Maximum tokens in LLM response.", ge=50, le=4096
    )

    openai_api_key: str = Field(
        default="", description="OpenAI API key (optional, can use env var)."
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL for local LLM."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_config() -> RagConfig:
    """Factory function to get configuration instance."""
    return RagConfig()
