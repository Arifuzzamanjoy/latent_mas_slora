"""
Document Store - Manages document loading, chunking, and indexing

Supports multiple formats:
- Plain text files
- JSON documents
- PDF (with optional PyPDF2)
- Markdown files

Optimized for integration with latent memory system.
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading


@dataclass
class DocumentChunk:
    """A chunk of a document for retrieval"""
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Any] = None  # Will be set by retriever
    
    def __hash__(self):
        return hash(self.chunk_id)
    
    def __eq__(self, other):
        if isinstance(other, DocumentChunk):
            return self.chunk_id == other.chunk_id
        return False


@dataclass
class Document:
    """A document in the store"""
    doc_id: str
    title: str
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[DocumentChunk] = field(default_factory=list)
    
    @property
    def num_chunks(self) -> int:
        return len(self.chunks)


class DocumentStore:
    """
    Document storage and management for RAG.
    
    Features:
    - Flexible document loading from files/strings
    - Intelligent text chunking with overlap
    - Metadata filtering
    - Thread-safe operations
    
    Example:
        store = DocumentStore()
        store.add_document("My doc content", title="Report", metadata={"year": 2024})
        store.load_directory("/path/to/docs")
        chunks = store.get_all_chunks()
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        self._documents: Dict[str, Document] = {}
        self._chunks: Dict[str, DocumentChunk] = {}
        self._lock = threading.RLock()
    
    def _generate_id(self, text: str, prefix: str = "doc") -> str:
        """Generate a unique ID from text content"""
        hash_val = hashlib.md5(text.encode()).hexdigest()[:12]
        return f"{prefix}_{hash_val}"
    
    def _chunk_text(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Uses sentence-aware splitting when possible.
        """
        chunks = []
        
        # Clean text
        text = text.strip()
        if len(text) < self.min_chunk_size:
            if text:
                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_c0",
                    doc_id=doc_id,
                    text=text,
                    start_char=0,
                    end_char=len(text),
                )
                chunks.append(chunk)
            return chunks
        
        # Split by paragraphs first, then combine to target size
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_start = 0
        pos = 0
        chunk_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                pos += 2  # Account for \n\n
                continue
            
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size and current_chunk:
                # Save current chunk
                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_c{chunk_idx}",
                    doc_id=doc_id,
                    text=current_chunk.strip(),
                    start_char=current_start,
                    end_char=pos,
                )
                chunks.append(chunk)
                chunk_idx += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap_text
                current_start = pos - len(overlap_text)
            
            current_chunk += ("\n\n" if current_chunk else "") + para
            pos += len(para) + 2
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_c{chunk_idx}",
                doc_id=doc_id,
                text=current_chunk.strip(),
                start_char=current_start,
                end_char=len(text),
            )
            chunks.append(chunk)
        
        return chunks
    
    def add_document(
        self,
        content: str,
        title: Optional[str] = None,
        doc_id: Optional[str] = None,
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Add a document to the store.
        
        Args:
            content: Document text content
            title: Document title (auto-generated if None)
            doc_id: Document ID (auto-generated if None)
            source: Source identifier
            metadata: Additional metadata
            
        Returns:
            Created Document object
        """
        with self._lock:
            doc_id = doc_id or self._generate_id(content)
            title = title or f"Document {doc_id[:8]}"
            metadata = metadata or {}
            
            # Check for duplicate
            if doc_id in self._documents:
                print(f"[RAG] Document {doc_id} already exists, updating...")
                # Remove old chunks
                old_doc = self._documents[doc_id]
                for chunk in old_doc.chunks:
                    self._chunks.pop(chunk.chunk_id, None)
            
            # Chunk the content
            chunks = self._chunk_text(content, doc_id)
            
            # Create document
            doc = Document(
                doc_id=doc_id,
                title=title,
                content=content,
                source=source,
                metadata=metadata,
                chunks=chunks,
            )
            
            # Store
            self._documents[doc_id] = doc
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk
            
            print(f"[RAG] Added document '{title}' with {len(chunks)} chunks")
            return doc
    
    def load_file(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Document]:
        """
        Load a document from file.
        
        Supports: .txt, .md, .json, .pdf (if PyPDF2 installed)
        """
        path = Path(path)
        
        if not path.exists():
            print(f"[RAG] File not found: {path}")
            return None
        
        metadata = metadata or {}
        metadata["filename"] = path.name
        metadata["filepath"] = str(path.absolute())
        
        suffix = path.suffix.lower()
        
        try:
            if suffix in ['.txt', '.md']:
                content = path.read_text(encoding='utf-8')
            elif suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle different JSON structures
                    if isinstance(data, str):
                        content = data
                    elif isinstance(data, dict):
                        content = data.get('content', data.get('text', json.dumps(data, indent=2)))
                    elif isinstance(data, list):
                        content = '\n\n'.join(
                            item.get('content', item.get('text', str(item))) 
                            if isinstance(item, dict) else str(item)
                            for item in data
                        )
                    else:
                        content = str(data)
            elif suffix == '.pdf':
                content = self._load_pdf(path)
            else:
                # Try as plain text
                content = path.read_text(encoding='utf-8')
            
            return self.add_document(
                content=content,
                title=path.stem,
                source=str(path),
                metadata=metadata,
            )
            
        except Exception as e:
            print(f"[RAG] Error loading {path}: {e}")
            return None
    
    def _load_pdf(self, path: Path) -> str:
        """Load text from PDF file"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text.strip()
        except ImportError:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(path))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
                return text.strip()
            except ImportError:
                raise ImportError("PDF support requires: pip install pypdf or pip install PyPDF2")
    
    def load_directory(
        self,
        directory: Union[str, Path],
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Path to directory
            extensions: File extensions to load (default: ['.txt', '.md', '.json'])
            recursive: Whether to search subdirectories
            
        Returns:
            List of loaded documents
        """
        directory = Path(directory)
        extensions = extensions or ['.txt', '.md', '.json', '.pdf']
        
        if not directory.exists():
            print(f"[RAG] Directory not found: {directory}")
            return []
        
        documents = []
        pattern = '**/*' if recursive else '*'
        
        for path in directory.glob(pattern):
            if path.is_file() and path.suffix.lower() in extensions:
                doc = self.load_file(path)
                if doc:
                    documents.append(doc)
        
        print(f"[RAG] Loaded {len(documents)} documents from {directory}")
        return documents
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID"""
        return self._documents.get(doc_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a chunk by ID"""
        return self._chunks.get(chunk_id)
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all chunks in the store"""
        return list(self._chunks.values())
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents in the store"""
        return list(self._documents.values())
    
    @property
    def documents(self) -> Dict[str, Document]:
        """Access all documents (read-only view)"""
        return dict(self._documents)
    
    @property
    def chunks(self) -> Dict[str, DocumentChunk]:
        """Access all chunks (read-only view)"""
        return dict(self._chunks)
    
    @property
    def num_documents(self) -> int:
        """Number of documents in store"""
        return len(self._documents)
    
    @property
    def num_chunks(self) -> int:
        """Number of chunks in store"""
        return len(self._chunks)
    
    def __len__(self) -> int:
        """Return number of documents"""
        return len(self._documents)
    
    def search_metadata(
        self,
        filters: Dict[str, Any],
    ) -> List[Document]:
        """Filter documents by metadata"""
        results = []
        for doc in self._documents.values():
            match = all(
                doc.metadata.get(key) == value
                for key, value in filters.items()
            )
            if match:
                results.append(doc)
        return results
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and its chunks"""
        with self._lock:
            if doc_id not in self._documents:
                return False
            
            doc = self._documents.pop(doc_id)
            for chunk in doc.chunks:
                self._chunks.pop(chunk.chunk_id, None)
            
            print(f"[RAG] Removed document {doc_id}")
            return True
    
    def clear(self) -> None:
        """Clear all documents"""
        with self._lock:
            self._documents.clear()
            self._chunks.clear()
            print("[RAG] Cleared all documents")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        total_chars = sum(len(doc.content) for doc in self._documents.values())
        return {
            "num_documents": len(self._documents),
            "num_chunks": len(self._chunks),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars / len(self._chunks) if self._chunks else 0,
        }
    
    def export_to_json(self, path: Union[str, Path]) -> None:
        """Export store to JSON file"""
        path = Path(path)
        data = {
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "content": doc.content,
                    "source": doc.source,
                    "metadata": doc.metadata,
                }
                for doc in self._documents.values()
            ]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[RAG] Exported {len(data['documents'])} documents to {path}")
    
    def import_from_json(self, path: Union[str, Path]) -> int:
        """Import documents from JSON file"""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for doc_data in data.get("documents", []):
            self.add_document(
                content=doc_data["content"],
                title=doc_data.get("title"),
                doc_id=doc_data.get("doc_id"),
                source=doc_data.get("source", "json_import"),
                metadata=doc_data.get("metadata", {}),
            )
            count += 1
        
        print(f"[RAG] Imported {count} documents from {path}")
        return count
