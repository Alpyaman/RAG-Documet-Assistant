"""
PDF Ingestion Module - Handles PDF document reading and text extraction.

This module provides robust PDF processing with error handling,
metadata extraction, and support for various PDF formats.
"""

import logging
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a processed document with metadata."""

    content: str
    metadata: Dict[str, any]
    source: str

    def __repr__(self) -> str:
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"Document(source={self.source}, length={len(self.content)}, preview={preview})"


class PDFIngestor:
    """
    Handles PDF document ingestion and text extraction.

    Features:
    - Extracts text from all pages
    - Preserves metadata (author, title, page count)
    - Handles corrupted PDFs gracefully
    - Supports batch processing

    Example:
        >>> ingestor = PDFIngestor()
        >>> doc = ingestor.ingest_pdf("report.pdf")
        >>> print(doc.content[:100])
    """

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize the PDF ingestor.

        Args:
            encoding: Text encoding to use (default: utf-8)
        """
        self.encoding = encoding
        logger.info("PDFIngestor initialized")

    def ingest_pdf(self, pdf_path: str | Path) -> Document:
        """
        Extract text and metadata from a single PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Document object containing text and metadata

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            PdfReadError: If PDF is corrupted or unreadable
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"File must be a PDF, got: {pdf_path.suffix}")

        logger.info(f"Processing PDF: {pdf_path.name}")

        try:
            reader = PdfReader(str(pdf_path))

            # Extract text from all pages
            text_content = []
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")

            full_text = "\n\n".join(text_content)

            # Extract metadata
            metadata = self._extract_metadata(reader, pdf_path)

            logger.info(
                f"Successfully extracted {len(full_text)} characters "
                f"from {metadata['page_count']} pages"
            )

            return Document(
                content=full_text,
                metadata=metadata,
                source=str(pdf_path)
            )

        except PdfReadError as e:
            logger.error(f"Failed to read PDF {pdf_path.name}: {e}")
            raise

    def ingest_directory(self, directory_path: str | Path) -> List[Document]:
        """
        Process all PDF files in a directory.

        Args:
            directory_path: Path to directory containing PDFs

        Returns:
            List of Document objects
        """
        directory_path = Path(directory_path)

        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        pdf_files = list(directory_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")

        documents = []
        for pdf_file in pdf_files:
            try:
                doc = self.ingest_pdf(pdf_file)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Skipping {pdf_file.name} due to error: {e}")

        logger.info(f"Successfully processed {len(documents)}/{len(pdf_files)} PDFs")
        return documents

    def _extract_metadata(self, reader: PdfReader, pdf_path: Path) -> Dict[str, any]:
        """
        Extract metadata from PDF.

        Args:
            reader: PdfReader instance
            pdf_path: Path to PDF file

        Returns:
            Dictionary of metadata
        """
        metadata = {
            "filename": pdf_path.name,
            "page_count": len(reader.pages),
            "file_size_bytes": pdf_path.stat().st_size,
        }

        # Try to extract PDF metadata (may not always be present)
        if reader.metadata:
            try:
                metadata.update({
                    "title": reader.metadata.get("/Title", "Unknown"),
                    "author": reader.metadata.get("/Author", "Unknown"),
                    "creator": reader.metadata.get("/Creator", "Unknown"),
                    "producer": reader.metadata.get("/Producer", "Unknown"),
                })
            except Exception as e:
                logger.debug(f"Could not extract all metadata: {e}")

        return metadata

    def get_page_content(self, pdf_path: str | Path, page_number: int) -> str:
        """
        Extract text from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)

        Returns:
            Text content of the specified page
        """
        pdf_path = Path(pdf_path)
        reader = PdfReader(str(pdf_path))
 
        if page_number < 1 or page_number > len(reader.pages):
            raise ValueError(
                f"Page number {page_number} out of range (1-{len(reader.pages)})"
            )

        return reader.pages[page_number - 1].extract_text()