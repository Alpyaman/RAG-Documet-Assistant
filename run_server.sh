 # RAG Document Assistant Server Runner
# This script helps you run the server either locally or in Docker
 
set -e  # Exit on error
 
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
 
# Configuration
IMAGE_NAME="rag-assistant"
CONTAINER_NAME="rag-assistant-api"
PORT=8000
 
# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  RAG Document Assistant Server${NC}"
    echo -e "${BLUE}========================================${NC}"
}
 
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}
 
print_error() {
    echo -e "${RED}✗ $1${NC}"
}
 
print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}
 
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"
}
 
run_local() {
    print_header
    print_info "Starting server locally..."
 
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
 
    # Activate virtual environment
    source venv/bin/activate
 
    # Install dependencies
    print_info "Installing dependencies..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    pip install -q -e .
 
    # Create data directories
    mkdir -p data/pdfs data/vector_db
 
    print_success "Dependencies installed"
    print_info "Starting FastAPI server on port $PORT..."
    echo ""
 
    # Run the server
    uvicorn rag_assistant.api:app --host 0.0.0.0 --port $PORT --reload
}
 
build_docker() {
    print_header
    print_info "Building Docker image..."
 
    check_docker
 
    docker build -t $IMAGE_NAME:latest .
 
    print_success "Docker image built: $IMAGE_NAME:latest"
}
 
run_docker() {
    print_header
    print_info "Running Docker container..."
 
    check_docker
 
    # Stop existing container if running
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        print_info "Stopping existing container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    fi
 
    # Check if image exists, if not build it
    if [[ "$(docker images -q $IMAGE_NAME:latest 2> /dev/null)" == "" ]]; then
        print_info "Image not found. Building..."
        build_docker
    fi
 
    # Run the container
    print_info "Starting container on port $PORT..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:8000 \
        -v $(pwd)/data:/app/data \
        -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
        $IMAGE_NAME:latest
 
    print_success "Container started: $CONTAINER_NAME"
    print_info "API available at: http://localhost:$PORT"
    print_info "API docs at: http://localhost:$PORT/docs"
    echo ""
    print_info "View logs with: docker logs -f $CONTAINER_NAME"
    print_info "Stop with: docker stop $CONTAINER_NAME"
}
 
stop_docker() {
    print_header
    print_info "Stopping Docker container..."
 
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
        print_success "Container stopped and removed"
    else
        print_info "Container is not running"
    fi
}
 
show_logs() {
    print_header
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        docker logs -f $CONTAINER_NAME
    else
        print_error "Container is not running"
        exit 1
    fi
}
 
show_usage() {
    print_header
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  local       - Run server locally (development mode)"
    echo "  build       - Build Docker image"
    echo "  docker      - Run server in Docker container"
    echo "  stop        - Stop Docker container"
    echo "  logs        - Show Docker container logs"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 local          # Run locally with hot reload"
    echo "  $0 docker         # Run in Docker (production-like)"
    echo "  $0 build          # Build Docker image only"
    echo "  $0 logs           # View container logs"
    echo ""
}
 
# Main script
case "${1:-}" in
    local)
        run_local
        ;;
    build)
        build_docker
        ;;
    docker)
        run_docker
        ;;
    stop)
        stop_docker
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        show_usage
        exit 1
        ;;
esac