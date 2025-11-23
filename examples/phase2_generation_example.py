"""
Phase 2 Example: RAG with LLM Generation

This example demonstrates the complete RAG pipeline with answer generation.

Usage:
    # With OpenAI (requires OPENAI_API_KEY env var):
    python examples/phase2_generation_example.py --provider openai
 
    # With Ollama (requires Ollama running locally):
    python examples/phase2_generation_example.py --provider ollama --model llama2
 
    # With HuggingFace (local, no API key needed):
    python examples/phase2_generation_example.py --provider huggingface --model google/flan-t5-base
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_assistant.pipeline import RAGPipeline
from rag_assistant.config import RagConfig


def main():
    parser = argparse.ArgumentParser(description="Phase 2: RAG with LLM Generation")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "ollama", "huggingface"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (defaults based on provider)"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to PDF file to process (optional)"
    )

    args = parser.parse_args()

    # Create configuration
    print(f"\n{'='*60}")
    print("Phase 2: RAG Document Assistant with LLM Generation")
    print(f"{'='*60}\n")

    config = RagConfig(
        llm_provider=args.provider,
        llm_model_name=args.model if args.model else get_default_model(args.provider),
    )

    print("Configuration:")
    print(f"  LLM Provider: {config.llm_provider}")
    print(f"  LLM Model: {config.llm_model_name}")
    print(f"  Embedding Model: {config.embedding_model_name}")
    print(f"  Chunk Size: {config.chunk_size}")
    print()

    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    try:
        pipeline = RAGPipeline(config=config)
        print("✓ Pipeline initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        print("\nTroubleshooting:")
        if args.provider == "openai":
            print("  - Make sure OPENAI_API_KEY is set in your environment")
            print("  - Example: export OPENAI_API_KEY='your-key-here'")
        elif args.provider == "ollama":
            print("  - Make sure Ollama is installed and running")
            print("  - Install: https://ollama.ai")
            print(f"  - Pull model: ollama pull {args.model or 'llama2'}")
        elif args.provider == "huggingface":
            print("  - Make sure transformers and torch are installed")
            print("  - Install: pip install transformers torch")
        return

    # Process PDF if provided
    if args.pdf:
        print(f"Processing PDF: {args.pdf}")
        result = pipeline.process_pdf(args.pdf)
        print(result)
        print()
    else:
        # Check if there's already data in the vector store
        stats = pipeline.get_stats()
        if stats['vector_store']['total_documents'] == 0:
            print("⚠ No documents in vector store. Process some PDFs first!")
            print("  Example: python examples/basic_usage.py")
            print("  Or use --pdf flag to process a document")
            return

        print(f"Found {stats['vector_store']['total_documents']} chunks in vector store\n")

    # Demo queries
    print(f"{'='*60}")
    print("DEMO: Retrieval vs Generation")
    print(f"{'='*60}\n")

    demo_questions = [
        "What is the main topic of the document?",
        "What are the key findings?",
        "What is the conclusion?",
    ]

    for i, question in enumerate(demo_questions, 1):
        print(f"\n[Question {i}] {question}\n")

        # Phase 1: Retrieval only (for comparison)
        print("Phase 1 (Retrieval Only):")
        print("-" * 40)
        search_results = pipeline.query(question, top_k=3)

        if search_results:
            for j, result in enumerate(search_results[:2], 1):
                snippet = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                print(f"  [{j}] {snippet}")
                print(f"      (score: {result.get('score', 'N/A'):.3f})\n")
        else:
            print("  No results found\n")

        # Phase 2: Retrieval + Generation
        print("Phase 2 (Retrieval + Generation):")
        print("-" * 40)

        try:
            answer_result = pipeline.generate_answer(question, top_k=3)

            print(f"Answer:\n{answer_result.answer}\n")
            print(f"Model: {answer_result.model}")
            if answer_result.total_tokens:
                print(f"Tokens: {answer_result.total_tokens}")
            print()

        except Exception as e:
            print(f"✗ Generation failed: {e}\n")
            if args.provider == "openai":
                print("  Check your OpenAI API key and quota")
            elif args.provider == "ollama":
                print("  Make sure Ollama is running and the model is pulled")

        print("-" * 60)

    # Show pipeline statistics
    print(f"\n{'='*60}")
    print("Pipeline Statistics")
    print(f"{'='*60}\n")

    stats = pipeline.get_stats()
    print(f"Embedding Model: {stats['embedding_model']['model_name']}")
    print(f"LLM Model: {stats['llm_model']['model_name']}")
    print(f"LLM Provider: {stats['llm_model']['provider']}")
    print(f"Total Chunks: {stats['vector_store']['total_documents']}")
    print()


def get_default_model(provider: str) -> str:
    """Get default model for a provider."""
    defaults = {
        "openai": "gpt-3.5-turbo",
        "ollama": "llama2",
        "huggingface": "google/flan-t5-base",
    }
    return defaults.get(provider, "gpt-3.5-turbo")


if __name__ == "__main__":
    main()