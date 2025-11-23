"""
Setup script for RAG Document Assistant.
 
Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

setup(
    name="rag-document-assistant",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="End-to-end RAG system for intelligent document Q&A",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RAG-Documet-Assistant",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "pypdf2>=3.0.1",
        "chromadb>=0.4.22",
        "sentence-transformers>=2.3.1",
        "faiss-cpu>=1.7.4",
        "torch>=2.1.2",
        "transformers>=4.36.2",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.3",
        "pydantic-settings>=2.1.0",
        "tqdm>=4.66.1",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="rag nlp embeddings vector-database chromadb sentence-transformers",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/RAG-Documet-Assistant/issues",
        "Source": "https://github.com/yourusername/RAG-Documet-Assistant",
    },
)
