"""
Basic Usage Example - RAG Document Assistant
 
This example demonstrates the core functionality of the RAG pipeline:
1. Processing a PDF document
2. Querying the processed document
3. Retrieving relevant chunks
"""
 
import sys
from pathlib import Path
 
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
 
from rag_assistant import RAGPipeline
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
 
 
console = Console()
 
 
def main():
    """Run the basic RAG pipeline example."""
 
    console.print("\n[bold cyan]RAG Document Assistant - Basic Usage[/bold cyan]\n")
 
    # Initialize the pipeline
    console.print("[yellow]Initializing RAG Pipeline...[/yellow]")
    pipeline = RAGPipeline()
 
    console.print("[green]✓ Pipeline initialized successfully[/green]\n")
 
    # Example 1: Process a single PDF
    console.print(Panel.fit(
        "[bold]Example 1: Process a PDF Document[/bold]",
        border_style="blue"
    ))
 
    pdf_path = "./data/pdfs/sample_document.pdf"
 
    # Check if example PDF exists
    if not Path(pdf_path).exists():
        console.print(
            f"[yellow]Note: Example PDF not found at {pdf_path}[/yellow]\n"
            f"[dim]Place your PDF in ./data/pdfs/ to test[/dim]\n"
        )
    else:
        console.print(f"Processing: [cyan]{pdf_path}[/cyan]")
 
        result = pipeline.process_pdf(pdf_path)
 
        # Display results
        table = Table(title="Processing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
 
        table.add_row("Documents Processed", str(result.documents_processed))
        table.add_row("Chunks Created", str(result.chunks_created))
        table.add_row("Embeddings Stored", str(result.embeddings_stored))
        table.add_row("Processing Time", f"{result.processing_time:.2f}s")
        table.add_row("Errors", str(len(result.errors)))
 
        console.print(table)
 
    # Example 2: Query the document
    console.print(f"\n{Panel.fit('[bold]Example 2: Query the Document[/bold]', border_style='blue')}\n")
 
    sample_queries = [
        "What is the main topic of the document?",
        "Can you summarize the key points?",
        "What are the conclusions?",
    ]
 
    for query in sample_queries:
        console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
 
        results = pipeline.query(query, top_k=3)
 
        if not results:
            console.print("[yellow]No results found. Process a document first.[/yellow]")
            continue
 
        console.print(f"\n[green]Found {len(results)} relevant chunks:[/green]\n")
 
        for i, result in enumerate(results, 1):
            similarity = result['similarity']
            text_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
 
            console.print(
                Panel(
                    f"[bold]Chunk #{i}[/bold] (Similarity: {similarity:.3f})\n\n{text_preview}",
                    border_style="green",
                    expand=False
                )
            )
 
    # Example 3: Get pipeline statistics
    console.print(f"\n{Panel.fit('[bold]Example 3: Pipeline Statistics[/bold]', border_style='blue')}\n")
 
    stats = pipeline.get_stats()
 
    console.print("[bold]Pipeline Configuration:[/bold]")
    console.print(f"  Chunk Size: {stats['pipeline']['chunk_size']}")
    console.print(f"  Chunk Overlap: {stats['pipeline']['chunk_overlap']}")
    console.print(f"  Batch Size: {stats['pipeline']['batch_size']}")
 
    console.print("\n[bold]Embedding Model:[/bold]")
    console.print(f"  Model: {stats['model']['model_name']}")
    console.print(f"  Dimension: {stats['model']['embedding_dimension']}")
    console.print(f"  Device: {stats['model']['device']}")
 
    console.print("\n[bold]Vector Store:[/bold]")
    console.print(f"  Collection: {stats['vector_store']['collection_name']}")
    console.print(f"  Total Documents: {stats['vector_store']['total_documents']}")
 
    console.print("\n[green]✓ Example completed successfully[/green]\n")
 
 
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()