"""
Advanced Usage Example - RAG Document Assistant
 
Demonstrates advanced features:
1. Batch processing multiple PDFs
2. Different chunking strategies
3. Custom configurations
4. Metadata filtering
5. Performance optimization
"""
 
import sys
from pathlib import Path
import time
 
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
 
from rag_assistant import (
    RAGPipeline,
    TextChunker,
    VectorStore,
)
from rag_assistant.chunking import ChunkingStrategy
from rag_assistant.config import RAGConfig
from rich.console import Console
 
 
console = Console()
 
 
def example_batch_processing():
    """Demonstrate batch processing of multiple PDFs."""
 
    console.print("\n[bold cyan]Example: Batch Processing[/bold cyan]\n")
 
    pipeline = RAGPipeline()
 
    # Process all PDFs in a directory
    pdf_directory = "./data/pdfs"
 
    if not Path(pdf_directory).exists():
        console.print(f"[yellow]Directory not found: {pdf_directory}[/yellow]")
        console.print("[dim]Create ./data/pdfs/ and add PDF files to test[/dim]\n")
        return
 
    console.print(f"Processing all PDFs in: [cyan]{pdf_directory}[/cyan]\n")
 
    result = pipeline.process_directory(pdf_directory)
 
    console.print("\n[bold green]Batch Processing Complete[/bold green]")
    console.print(f"  Documents: {result.documents_processed}")
    console.print(f"  Chunks: {result.chunks_created}")
    console.print(f"  Time: {result.processing_time:.2f}s")
 
    if result.errors:
        console.print("\n[yellow]Errors encountered:[/yellow]")
        for error in result.errors:
            console.print(f"  - {error}")
 
 
def example_custom_chunking():
    """Demonstrate different chunking strategies."""
 
    console.print("\n[bold cyan]Example: Custom Chunking Strategies[/bold cyan]\n")
 
    strategies = [
        (ChunkingStrategy.FIXED_SIZE, "Fixed Size (1000 chars)"),
        (ChunkingStrategy.SENTENCE, "Sentence-based"),
        (ChunkingStrategy.PARAGRAPH, "Paragraph-based"),
    ]
 
    for strategy, name in strategies:
        console.print(f"\n[yellow]Testing {name}[/yellow]")
 
        chunker = TextChunker(
            chunk_size=1000,
            chunk_overlap=200,
            strategy=strategy
        )
 
        pipeline = RAGPipeline(
            chunker=chunker,
            vector_store=VectorStore(collection_name=f"chunks_{strategy.value}")
        )
 
        # Process a sample PDF if available
        sample_pdf = "./data/pdfs/sample_document.pdf"
        if Path(sample_pdf).exists():
            result = pipeline.process_pdf(sample_pdf)
            console.print(f"  Created {result.chunks_created} chunks")
 
            # Get chunk statistics
            stats = chunker.get_chunk_stats(chunker.chunk_text("Sample text " * 200))
            console.print(f"  Avg chunk size: {stats.get('avg_chunk_size', 0):.0f} chars")
 
 
def example_custom_config():
    """Demonstrate custom configuration."""
 
    console.print("\n[bold cyan]Example: Custom Configuration[/bold cyan]\n")
 
    # Create custom configuration
    custom_config = RAGConfig(
        chunk_size=500,  # Smaller chunks
        chunk_overlap=100,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64,  # Larger batch size
        device="cpu",
        collection_name="custom_collection"
    )
 
    console.print("[bold]Custom Configuration:[/bold]")
    console.print(f"  Chunk Size: {custom_config.chunk_size}")
    console.print(f"  Overlap: {custom_config.chunk_overlap}")
    console.print(f"  Model: {custom_config.embedding_model_name}")
    console.print(f"  Batch Size: {custom_config.batch_size}")
 
    # Initialize pipeline with custom config
    pipeline = RAGPipeline(config=custom_config)
 
    console.print(f"\n[green]✓ Pipeline initialized with custom config[/green] {pipeline}")
 
 
def example_metadata_filtering():
    """Demonstrate metadata filtering in queries."""
 
    console.print("\n[bold cyan]Example: Metadata Filtering[/bold cyan]\n")
 
    pipeline = RAGPipeline()
 
    # Query with metadata filter
    query = "What are the main conclusions?"
 
    console.print(f"Query: [cyan]{query}[/cyan]\n")
 
    # Search without filter
    all_results = pipeline.query(query, top_k=5)
    console.print(f"Results without filter: {len(all_results)}")
 
    # Search with metadata filter (if documents are processed)
    if all_results:
        sample_filename = all_results[0]['metadata'].get('filename')
 
        if sample_filename:
            filtered_results = pipeline.query(
                query,
                top_k=5,
                filter_metadata={"filename": sample_filename}
            )
 
            console.print(f"Results filtered by '{sample_filename}': {len(filtered_results)}")
 
 
def example_performance_comparison():
    """Compare performance of different configurations."""
 
    console.print("\n[bold cyan]Example: Performance Comparison[/bold cyan]\n")
 
    sample_pdf = "./data/pdfs/sample_document.pdf"
 
    if not Path(sample_pdf).exists():
        console.print("[yellow]Sample PDF not found, skipping performance test[/yellow]")
        return
 
    configs = [
        ("Small chunks", {"chunk_size": 500, "chunk_overlap": 50}),
        ("Medium chunks", {"chunk_size": 1000, "chunk_overlap": 200}),
        ("Large chunks", {"chunk_size": 2000, "chunk_overlap": 400}),
    ]
 
    results = []
 
    for name, params in configs:
        console.print(f"\n[yellow]Testing: {name}[/yellow]")
 
        config = RAGConfig(**params, collection_name=f"perf_{params['chunk_size']}")
        pipeline = RAGPipeline(config=config)
 
        start = time.time()
        result = pipeline.process_pdf(sample_pdf)
        elapsed = time.time() - start
 
        results.append({
            "name": name,
            "chunks": result.chunks_created,
            "time": elapsed,
        })
 
        console.print(f"  Chunks: {result.chunks_created}")
        console.print(f"  Time: {elapsed:.2f}s")
 
    # Summary
    console.print("\n[bold]Performance Summary:[/bold]")
    for r in results:
        console.print(f"  {r['name']}: {r['chunks']} chunks in {r['time']:.2f}s")
 
 
def main():
    """Run all advanced examples."""
 
    console.print("[bold cyan]RAG Document Assistant - Advanced Examples[/bold cyan]")
 
    examples = [
        ("Batch Processing", example_batch_processing),
        ("Custom Chunking", example_custom_chunking),
        ("Custom Configuration", example_custom_config),
        ("Metadata Filtering", example_metadata_filtering),
        ("Performance Comparison", example_performance_comparison),
    ]
 
    console.print("\n[bold]Available Examples:[/bold]")
    for i, (name, _) in enumerate(examples, 1):
        console.print(f"  {i}. {name}")
 
    console.print("\n[dim]Running all examples...[/dim]\n")
    console.print("=" * 60)
 
    for name, example_func in examples:
        try:
            example_func()
            console.print("\n" + "=" * 60)
        except Exception as e:
            console.print(f"\n[red]Error in {name}: {e}[/red]")
            import traceback
            traceback.print_exc()
 
    console.print("\n[green]✓ All examples completed[/green]\n")
 
 
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()