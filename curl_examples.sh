# RAG Document Assistant - cURL Examples
# Quick reference for testing API endpoints
 
BASE_URL="http://localhost:8000"
 
echo "======================================"
echo "RAG Document Assistant - API Examples"
echo "======================================"
echo ""
echo "Make sure the server is running:"
echo "  ./run_server.sh local"
echo ""
 
# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
 
# 1. Health Check
echo -e "${BLUE}1. Health Check${NC}"
echo "GET /"
echo ""
curl -X GET "$BASE_URL/" | jq
echo -e "\n"
 
# 2. Upload PDF
echo -e "${BLUE}2. Upload PDF${NC}"
echo "POST /upload"
echo ""
echo "# Example (replace with your PDF path):"
echo "curl -X POST $BASE_URL/upload -F \"file=@/path/to/document.pdf\" | jq"
echo -e "\n"
 
# 3. Query Documents
echo -e "${BLUE}3. Query Documents (Retrieval Only)${NC}"
echo "POST /query"
echo ""
cat << 'EOF' | jq
curl -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "top_k": 5
  }'
EOF
echo ""
echo "# Try it:"
echo 'curl -X POST '"$BASE_URL"'/query -H "Content-Type: application/json" -d '\''{"query": "What is the main topic?", "top_k": 5}'\'' | jq'
echo -e "\n"
 
# 4. Generate Answer
echo -e "${BLUE}4. Generate Answer (RAG)${NC}"
echo "POST /generate"
echo ""
cat << 'EOF'
curl -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "top_k": 5,
    "return_context": true
  }'
EOF
echo ""
echo "# Try it:"
echo 'curl -X POST '"$BASE_URL"'/generate -H "Content-Type: application/json" -d '\''{"query": "What are the key findings?", "top_k": 5, "return_context": false}'\'' | jq'
echo -e "\n"
 
# 5. Get Statistics
echo -e "${BLUE}5. Get Statistics${NC}"
echo "GET /stats"
echo ""
curl -X GET "$BASE_URL/stats" | jq
echo -e "\n"
 
# 6. Clear All Data
echo -e "${BLUE}6. Clear All Data (DESTRUCTIVE)${NC}"
echo "DELETE /clear"
echo ""
echo "# WARNING: This deletes all documents!"
echo "curl -X DELETE $BASE_URL/clear | jq"
echo -e "\n"
 
# Additional Examples
echo -e "${BLUE}Additional Examples${NC}"
echo ""
 
echo "# Upload multiple PDFs:"
echo 'for pdf in *.pdf; do'
echo '  echo "Uploading $pdf..."'
echo '  curl -X POST '"$BASE_URL"'/upload -F "file=@$pdf"'
echo 'done'
echo ""
 
echo "# Query with metadata filter:"
echo 'curl -X POST '"$BASE_URL"'/query -H "Content-Type: application/json" -d '\''{"query": "search term", "top_k": 3, "filter_metadata": {"source": "document.pdf"}}'\'' | jq'
echo ""
 
echo "# Generate answer with more context:"
echo 'curl -X POST '"$BASE_URL"'/generate -H "Content-Type: application/json" -d '\''{"query": "your question", "top_k": 10, "return_context": true}'\'' | jq'
echo ""
 
echo "======================================"
echo "API Documentation: $BASE_URL/docs"
echo "======================================"