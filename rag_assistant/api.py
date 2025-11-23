"""
FastAPI Web API for RAG Document Assistant.

This module provides RESTful endpoints to:
- Upload PDF documents
- Query the vector store
- Generate answers using RAG
- Manage the document collection
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline
from .config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and cleanup."""
    global pipeline

    logger.info("Initializing RAG Pipeline...")
    config = get_config()

    # Ensure data directories exist
    os.makedirs(config.pdf_storage_path, exist_ok=True)
    os.makedirs(config.vector_db_path, exist_ok=True)

    # Initialize the pipeline
    pipeline = RAGPipeline(config=config)
    logger.info("RAG Pipeline initialized successfully")

    yield

    # Cleanup (if needed)
    logger.info("Shutting down RAG Pipeline...")


# Create FastAPI app
app = FastAPI(
    title="RAG Document Assistant API",
    description="A production-ready RAG system for document Q&A",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for querying documents."""

    query: str = Field(..., description="The query text", min_length=1)
    top_k: int = Field(
        default=5, description="Number of results to return", ge=1, le=20
    )
    filter_metadata: Optional[Dict] = Field(
        default=None, description="Optional metadata filters"
    )


class QueryResponse(BaseModel):
    """Response model for query results."""

    query: str
    results: List[Dict]
    count: int


class GenerateRequest(BaseModel):
    """Request model for generating answers."""

    query: str = Field(..., description="The question to answer", min_length=1)
    top_k: int = Field(
        default=5, description="Number of context chunks to use", ge=1, le=20
    )
    return_context: bool = Field(
        default=False, description="Include context chunks in response"
    )


class GenerateResponse(BaseModel):
    """Response model for generated answers."""

    query: str
    answer: str
    model: str
    context_used: Optional[List[str]] = None


class UploadResponse(BaseModel):
    """Response model for file upload."""

    filename: str
    status: str
    result: Dict


class StatsResponse(BaseModel):
    """Response model for statistics."""

    stats: Dict


# API Endpoints


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "RAG Document Assistant API",
        "version": "1.0.0",
    }


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.

    The document will be:
    1. Saved to storage
    2. Chunked into smaller pieces
    3. Embedded using the configured model
    4. Stored in the vector database

    Args:
        file: PDF file to upload

    Returns:
        Processing result with statistics
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    try:
        # Save uploaded file
        config = get_config()
        file_path = Path(config.pdf_storage_path) / file.filename

        logger.info(f"Saving uploaded file: {file.filename}")
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process the PDF through pipeline
        logger.info(f"Processing PDF: {file.filename}")
        result = pipeline.process_pdf(file_path)

        if result.errors:
            logger.error(f"Errors during processing: {result.errors}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Processing failed: {result.errors[0]}",
            )

        return UploadResponse(
            filename=file.filename,
            status="success",
            result={
                "documents_processed": result.documents_processed,
                "chunks_created": result.chunks_created,
                "embeddings_stored": result.embeddings_stored,
                "processing_time": round(result.processing_time, 2),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file: {str(e)}",
        )


@app.post("/query", response_model=QueryResponse, tags=["Search"])
async def query_documents(request: QueryRequest):
    """
    Query the document collection (retrieval only, no generation).

    Returns the most relevant document chunks based on semantic similarity.

    Args:
        request: Query parameters

    Returns:
        List of relevant document chunks
    """
    try:
        logger.info(f"Query request: '{request.query}' (top_k={request.top_k})")

        results = pipeline.query(
            query_text=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
            return_embeddings=False,
        )

        return QueryResponse(query=request.query, results=results, count=len(results))

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_answer(request: GenerateRequest):
    """
    Generate an answer to a question using RAG (Retrieval-Augmented Generation).

    This endpoint:
    1. Retrieves relevant document chunks
    2. Uses an LLM to generate a coherent answer

    Args:
        request: Generation parameters

    Returns:
        Generated answer with optional context
    """
    try:
        logger.info(f"Generate request: '{request.query}'")

        result = pipeline.generate_answer(
            query_text=request.query,
            top_k=request.top_k,
            return_context=request.return_context,
        )

        return GenerateResponse(
            query=request.query,
            answer=result.answer,
            model=result.model,
            context_used=result.context_used if request.return_context else None,
        )

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {str(e)}",
        )


@app.get("/stats", response_model=StatsResponse, tags=["Management"])
async def get_statistics():
    """
    Get statistics about the RAG pipeline and document collection.

    Returns:
        Pipeline configuration and collection statistics
    """
    try:
        stats = pipeline.get_stats()
        return StatsResponse(stats=stats)

    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}",
        )


@app.delete("/clear", tags=["Management"])
async def clear_all_data():
    """
    Clear all documents from the vector store.

    WARNING: This is destructive and cannot be undone!

    Returns:
        Success message
    """
    try:
        logger.warning("Clear all data request received")
        pipeline.clear_all_data()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "All documents cleared from vector store",
            },
        )

    except Exception as e:
        logger.error(f"Failed to clear data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear data: {str(e)}",
        )


# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "rag_assistant.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
