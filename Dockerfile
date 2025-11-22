# Multi-stage Docker build for RAG Document Assistant
# Stage 1: Base image with dependencies
FROM python:3.10-slim as base
 
# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
 
# Set working directory
WORKDIR /app
 
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
 
# Stage 2: Dependencies installation
FROM base as dependencies
 
# Copy requirements file
COPY requirements.txt .
 
# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
 
# Stage 3: Final application image
FROM dependencies as final
 
# Copy application code
COPY . .
 
# Install the package in development mode
RUN pip install -e .
 
# Create necessary directories
RUN mkdir -p /app/data/pdfs /app/data/vector_db
 
# Expose port
EXPOSE 8000
 
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1
 
# Run the FastAPI application
CMD ["uvicorn", "rag_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]