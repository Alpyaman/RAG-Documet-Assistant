# RAG Document Assistant - API Guide
 
## Phase 3: Web API Deployment
 
This guide explains how to run and use the RAG Document Assistant as a web service.
 
## Quick Start
 
### Option 1: Run Locally (Development)
 
```bash
# Make the script executable (first time only)
chmod +x run_server.sh
 
# Start the server
./run_server.sh local
```
 
The server will start at `http://localhost:8000` with hot reload enabled.
 
### Option 2: Run with Docker (Production-like)
 
```bash
# Build and run with Docker
./run_server.sh docker
 
# View logs
./run_server.sh logs
 
# Stop the container
./run_server.sh stop
```
 
### Option 3: Use Docker Compose
 
```bash
# Copy environment variables
cp .env.example .env
 
# Edit .env with your configuration
nano .env
 
# Start the service
docker-compose up -d
 
# View logs
docker-compose logs -f
 
# Stop the service
docker-compose down
```
 
## API Endpoints
 
Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
 
### 1. Health Check
 
**Endpoint**: `GET /`
 
Check if the API is running.
 
```bash
curl http://localhost:8000/
```
 
**Response**:
```json
{
  "status": "healthy",
  "service": "RAG Document Assistant API",
  "version": "1.0.0"
}
```
 
### 2. Upload PDF
 
**Endpoint**: `POST /upload`
 
Upload and process a PDF document.
 
```bash
curl -X POST \
  http://localhost:8000/upload \
  -F "file=@/path/to/document.pdf"
```
 
**Response**:
```json
{
  "filename": "document.pdf",
  "status": "success",
  "result": {
    "documents_processed": 1,
    "chunks_created": 45,
    "embeddings_stored": 45,
    "processing_time": 12.34
  }
}
```
 
### 3. Query Documents (Retrieval Only)
 
**Endpoint**: `POST /query`
 
Search for relevant document chunks without generating an answer.
 
```bash
curl -X POST \
  http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "top_k": 5
  }'
```
 
**Response**:
```json
{
  "query": "What is the main topic?",
  "results": [
    {
      "text": "The main topic discusses...",
      "score": 0.8567,
      "metadata": {
        "source": "document.pdf",
        "page": 1
      }
    }
  ],
  "count": 5
}
```
 
### 4. Generate Answer (RAG)
 
**Endpoint**: `POST /generate`
 
Generate an answer using Retrieval-Augmented Generation.
 
```bash
curl -X POST \
  http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "top_k": 5,
    "return_context": false
  }'
```
 
**Response**:
```json
{
  "query": "What are the key findings?",
  "answer": "The key findings indicate that...",
  "model": "gpt-3.5-turbo"
}
```
 
### 5. Get Statistics
 
**Endpoint**: `GET /stats`
 
Get information about the pipeline and document collection.
 
```bash
curl http://localhost:8000/stats
```
 
**Response**:
```json
{
  "stats": {
    "pipeline": {
      "chunk_size": 1000,
      "chunk_overlap": 200,
      "batch_size": 32
    },
    "embedding_model": {
      "name": "sentence-transformers/all-MiniLM-L6-v2",
      "dimension": 384
    },
    "vector_store": {
      "total_documents": 45,
      "collection_name": "document_embeddings"
    }
  }
}
```
 
### 6. Clear All Data
 
**Endpoint**: `DELETE /clear`
 
Remove all documents from the vector store.
 
⚠️ **Warning**: This is destructive and cannot be undone!
 
```bash
curl -X DELETE http://localhost:8000/clear
```
 
**Response**:
```json
{
  "status": "success",
  "message": "All documents cleared from vector store"
}
```
 
## Python Client Example
 
```python
import requests
 
BASE_URL = "http://localhost:8000"
 
# Upload a PDF
with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f, "application/pdf")}
    response = requests.post(f"{BASE_URL}/upload", files=files)
    print(response.json())
 
# Generate an answer
payload = {
    "query": "What is the conclusion?",
    "top_k": 5,
    "return_context": True
}
response = requests.post(f"{BASE_URL}/generate", json=payload)
result = response.json()
print(f"Answer: {result['answer']}")
```
 
For more examples, see `examples/api_usage.py`.
 
## Configuration
 
### Environment Variables
 
Create a `.env` file from the template:
 
```bash
cp .env.example .env
```
 
Key settings:
 
| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai/ollama/huggingface) | ollama |
| `LLM_MODEL_NAME` | Model to use | llama2 |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) | - |
| `CHUNK_SIZE` | Maximum characters per chunk | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 |
| `DEVICE` | Device for inference (cpu/cuda) | cpu |
 
### Using Different LLM Providers
 
#### OpenAI (GPT Models)
 
```bash
# In .env
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-3.5-turbo
OPENAI_API_KEY=sk-...
```
 
#### Ollama (Local Models)
 
```bash
# 1. Install and start Ollama
# Visit: https://ollama.ai
 
# 2. Pull a model
ollama pull llama2
 
# 3. Configure .env
LLM_PROVIDER=ollama
LLM_MODEL_NAME=llama2
OLLAMA_BASE_URL=http://localhost:11434
```
 
#### HuggingFace (Local Models)
 
```bash
# In .env
LLM_PROVIDER=huggingface
LLM_MODEL_NAME=google/flan-t5-base
DEVICE=cpu  # or "cuda" if you have a GPU
```
 
## Testing the API
 
Run the example script:
 
```bash
python examples/api_usage.py
```
 
Or use the interactive documentation:
 
1. Start the server: `./run_server.sh local`
2. Open browser: http://localhost:8000/docs
3. Try out the endpoints directly in the UI
 
## Deployment Checklist
 
For production deployment:
 
- [ ] Set secure environment variables
- [ ] Configure proper LLM provider and API keys
- [ ] Set up persistent volume for data
- [ ] Configure reverse proxy (nginx, Caddy)
- [ ] Enable HTTPS/TLS
- [ ] Set up monitoring and logging
- [ ] Configure CORS if needed
- [ ] Add authentication/authorization
- [ ] Set resource limits (CPU, memory)
- [ ] Configure backup strategy for vector DB
 
## Troubleshooting
 
### Server won't start
 
1. Check if port 8000 is available:
   ```bash
   lsof -i :8000
   ```
 
2. Check logs:
   ```bash
   # For local
   Check terminal output
 
   # For Docker
   docker logs rag-assistant-api
   ```
 
### Upload fails
 
1. Ensure PDF is valid and not corrupted
2. Check file size limits
3. Verify disk space in data directory
 
### Generation errors
 
1. Verify LLM provider is configured correctly
2. Check API keys (for OpenAI)
3. Ensure Ollama is running (for Ollama)
4. Check model is downloaded
 
### Out of memory
 
1. Reduce `BATCH_SIZE` in .env
2. Use smaller embedding model
3. Reduce `CHUNK_SIZE`
4. Increase Docker memory limits
 
## Next Steps
 
- Add authentication (JWT, API keys)
- Implement rate limiting
- Add caching layer (Redis)
- Set up monitoring (Prometheus, Grafana)
- Add async processing for large files
- Implement webhooks for processing notifications
- Add multi-user support with user isolation
 
## Support
 
For issues and questions:
- Check the [main README](README.md)
- Review [examples](examples/)
- Open an issue on GitHub