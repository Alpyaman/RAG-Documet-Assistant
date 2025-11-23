"""
Example: Using the RAG Document Assistant API

This script demonstrates how to interact with the FastAPI endpoints.
Make sure the server is running first:
    ./run_server.sh local
    # or
    ./run_server.sh docker
"""

import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def health_check():
    """Check if the API is healthy."""
    print_section("Health Check")

    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200

def upload_pdf(pdf_path: str):
    """Upload a PDF file to the API."""
    print_section(f"Uploading PDF: {pdf_path}")

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        return None

    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f, "application/pdf")}
        response = requests.post(f"{BASE_URL}/upload", files=files)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        return result
    else:
        print(f"Error: {response.text}")
        return None

def query_documents(query: str, top_k: int = 5):
    """Query the document collection (retrieval only)."""
    print_section(f"Query: {query}")

    payload = {"query": query, "top_k": top_k}

    response = requests.post(
        f"{BASE_URL}/query", json=payload, headers={"Content-Type": "application/json"}
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nFound {result['count']} results:")

        for i, doc in enumerate(result["results"], 1):
            print(f"\n--- Result {i} ---")
            print(f"Score: {doc.get('score', 'N/A'):.4f}")
            print(f"Text: {doc['text'][:200]}...")
            if "metadata" in doc:
                print(f"Metadata: {doc['metadata']}")

        return result
    else:
        print(f"Error: {response.text}")
        return None

def generate_answer(query: str, top_k: int = 5, return_context: bool = False):
    """Generate an answer using RAG."""
    print_section(f"Generate Answer: {query}")

    payload = {"query": query, "top_k": top_k, "return_context": return_context}

    response = requests.post(
        f"{BASE_URL}/generate",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()

        print(f"\nQuestion: {result['query']}")
        print(f"\nAnswer ({result['model']}):")
        print(f"{result['answer']}")

        if return_context and result.get("context_used"):
            print(f"\nContext Used ({len(result['context_used'])} chunks):")
            for i, context in enumerate(result["context_used"], 1):
                print(f"\n  {i}. {context[:150]}...")

        return result
    else:
        print(f"Error: {response.text}")
        return None

def get_statistics():
    """Get pipeline statistics."""
    print_section("Pipeline Statistics")

    response = requests.get(f"{BASE_URL}/stats")

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        stats = response.json()["stats"]
        print(json.dumps(stats, indent=2))
        return stats
    else:
        print(f"Error: {response.text}")
        return None

def clear_all_data():
    """Clear all documents (use with caution!)."""
    print_section("Clear All Data")

    confirm = input("Are you sure you want to delete all documents? (yes/no): ")
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    response = requests.delete(f"{BASE_URL}/clear")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def main():
    """Run the example workflow."""
    print("\n" + "=" * 60)
    print("  RAG Document Assistant - API Example")
    print("=" * 60)

    # 1. Health check
    if not health_check():
        print("\nError: API is not healthy. Make sure the server is running.")
        print("Start the server with: ./run_server.sh local")
        return

    # 2. Get initial statistics
    get_statistics()

    # 3. Upload a PDF (update this path to your test PDF)
    # pdf_path = "path/to/your/document.pdf"
    # upload_result = upload_pdf(pdf_path)

    # If you have uploaded PDFs, you can query them:

    # 4. Query documents (retrieval only)
    # query_documents("What is the main topic of the document?", top_k=3)

    # 5. Generate answers (RAG)
    # generate_answer("What are the key findings?", top_k=5, return_context=True)

    # 6. Get updated statistics
    # get_statistics()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("=" * 60)

if __name__ == "__main__":
    main()
