# RAG Document Assistant - UI Startup Script
# This script starts the Streamlit frontend
 
echo "🚀 Starting RAG Document Assistant UI..."
echo ""
echo "Make sure the FastAPI backend is running on http://localhost:8000"
echo "If not, start it with: uvicorn rag_assistant.api:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "Starting Streamlit UI on http://localhost:8501..."
echo ""
 
# Set the API base URL (default to localhost)
export API_BASE_URL=${API_BASE_URL:-http://localhost:8000}
 
# Run Streamlit
streamlit run rag_assistant/ui.py