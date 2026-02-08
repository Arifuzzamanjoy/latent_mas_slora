"""
RAG (Retrieval-Augmented Generation) Module for Document Intelligence

Provides:
- Document loading and chunking
- Embedding-based retrieval
- Context injection for multi-agent reasoning
"""

from .document_store import DocumentStore, Document, DocumentChunk
from .retriever import RAGRetriever, RetrievalResult
from .rag_pipeline import RAGPipeline

__all__ = [
    "DocumentStore",
    "Document", 
    "DocumentChunk",
    "RAGRetriever",
    "RetrievalResult",
    "RAGPipeline",
]
