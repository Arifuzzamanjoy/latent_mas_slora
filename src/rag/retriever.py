"""
RAG Retriever - Embedding-based document retrieval

Supports multiple embedding backends:
- Sentence Transformers (local)
- HuggingFace embeddings
- Simple TF-IDF fallback

Integrates with latent memory for context-aware retrieval.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field

from .document_store import DocumentStore, DocumentChunk


@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    chunks: List[DocumentChunk]
    scores: List[float]
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_context(self, max_length: int = 4000, separator: str = "\n\n---\n\n") -> str:
        """Get concatenated context from retrieved chunks"""
        context = ""
        for chunk, score in zip(self.chunks, self.scores):
            addition = f"[Source: {chunk.doc_id}] (relevance: {score:.2f})\n{chunk.text}"
            if len(context) + len(addition) + len(separator) > max_length:
                break
            context += (separator if context else "") + addition
        return context
    
    def __len__(self) -> int:
        return len(self.chunks)


class RAGRetriever:
    """
    Embedding-based retriever for RAG.
    
    Features:
    - Multiple embedding backends
    - Efficient batch processing
    - Semantic + keyword hybrid search
    - Integration with latent memory
    
    Example:
        retriever = RAGRetriever(store)
        retriever.build_index()
        results = retriever.retrieve("What is machine learning?", top_k=5)
        context = results.get_context()
    """
    
    def __init__(
        self,
        store: DocumentStore,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cuda",
        use_hybrid: bool = True,
    ):
        self.store = store
        self.embedding_model_name = embedding_model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_hybrid = use_hybrid
        
        self._embedding_model = None
        self._chunk_embeddings: Dict[str, torch.Tensor] = {}
        self._embedding_matrix: Optional[torch.Tensor] = None
        self._chunk_ids: List[str] = []
        self._indexed = False
        
        # TF-IDF fallback
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
    
    def _load_embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            import warnings
            import logging
            
            # Suppress harmless warnings about position_ids
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            warnings.filterwarnings("ignore", message=".*position_ids.*")
            
            print(f"[RAG] Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            if self.device == "cuda":
                self._embedding_model = self._embedding_model.to(self.device)
            print("[RAG] Embedding model loaded")
        except ImportError:
            print("[RAG] sentence-transformers not found, using TF-IDF fallback")
            self._init_tfidf()
    
    def _init_tfidf(self):
        """Initialize TF-IDF fallback"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
            )
        except ImportError:
            print("[RAG] sklearn not found, retrieval will be limited")
    
    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Embed a list of texts"""
        self._load_embedding_model()
        
        if self._embedding_model is not None:
            embeddings = self._embedding_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,
            )
            return embeddings.to(self.device)
        elif self._tfidf_vectorizer is not None:
            # TF-IDF fallback
            if self._tfidf_matrix is None:
                # Need to fit first
                all_texts = [c.text for c in self.store.get_all_chunks()]
                self._tfidf_vectorizer.fit(all_texts)
            
            tfidf = self._tfidf_vectorizer.transform(texts)
            return torch.tensor(tfidf.toarray(), dtype=torch.float32, device=self.device)
        else:
            raise RuntimeError("No embedding backend available")
    
    def build_index(self, force: bool = False) -> None:
        """
        Build embedding index for all chunks.
        
        Args:
            force: Rebuild even if already indexed
        """
        if self._indexed and not force:
            print("[RAG] Index already built, use force=True to rebuild")
            return
        
        chunks = self.store.get_all_chunks()
        if not chunks:
            # Silently skip - no documents loaded (expected for some evaluations)
            return
        
        print(f"[RAG] Building index for {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        # Compute embeddings
        self._load_embedding_model()
        
        if self._embedding_model is not None:
            embeddings = self._embedding_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,
                batch_size=32,
            )
            self._embedding_matrix = embeddings.to(self.device)
            
            # Store individual embeddings
            for chunk_id, emb in zip(self._chunk_ids, embeddings):
                self._chunk_embeddings[chunk_id] = emb
        
        elif self._tfidf_vectorizer is not None:
            # TF-IDF fallback
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
            self._embedding_matrix = torch.tensor(
                self._tfidf_matrix.toarray(),
                dtype=torch.float32,
                device=self.device,
            )
        
        self._indexed = True
        print(f"[RAG] Index built successfully")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filters: Metadata filters to apply
            
        Returns:
            RetrievalResult with chunks and scores
        """
        if not self._indexed:
            self.build_index()
        
        if self._embedding_matrix is None or len(self._chunk_ids) == 0:
            return RetrievalResult(chunks=[], scores=[], query=query)
        
        # Embed query
        query_embedding = self._embed_texts([query])[0]
        
        # Compute similarities
        if self._embedding_model is not None:
            # Cosine similarity (embeddings are normalized)
            similarities = torch.matmul(self._embedding_matrix, query_embedding)
        else:
            # TF-IDF cosine similarity
            query_norm = query_embedding / (query_embedding.norm() + 1e-8)
            matrix_norm = self._embedding_matrix / (self._embedding_matrix.norm(dim=1, keepdim=True) + 1e-8)
            similarities = torch.matmul(matrix_norm, query_norm)
        
        # Apply hybrid search boost if enabled
        if self.use_hybrid:
            keyword_scores = self._keyword_boost(query)
            similarities = 0.7 * similarities + 0.3 * keyword_scores
        
        # Get top-k
        scores, indices = torch.topk(similarities, min(top_k * 2, len(self._chunk_ids)))
        
        # Filter by threshold and metadata
        chunks = []
        final_scores = []
        
        for score, idx in zip(scores.tolist(), indices.tolist()):
            if score < score_threshold:
                continue
            
            chunk_id = self._chunk_ids[idx]
            chunk = self.store.get_chunk(chunk_id)
            
            if chunk is None:
                continue
            
            # Apply metadata filters
            if filters:
                doc = self.store.get_document(chunk.doc_id)
                if doc and not all(doc.metadata.get(k) == v for k, v in filters.items()):
                    continue
            
            chunks.append(chunk)
            final_scores.append(score)
            
            if len(chunks) >= top_k:
                break
        
        return RetrievalResult(
            chunks=chunks,
            scores=final_scores,
            query=query,
            metadata={"top_k": top_k, "threshold": score_threshold},
        )
    
    def _keyword_boost(self, query: str) -> torch.Tensor:
        """Compute keyword-based boost scores"""
        query_words = set(query.lower().split())
        scores = []
        
        for chunk_id in self._chunk_ids:
            chunk = self.store.get_chunk(chunk_id)
            if chunk:
                chunk_words = set(chunk.text.lower().split())
                overlap = len(query_words & chunk_words)
                score = overlap / (len(query_words) + 1)
                scores.append(score)
            else:
                scores.append(0.0)
        
        return torch.tensor(scores, dtype=torch.float32, device=self.device)
    
    def retrieve_with_context(
        self,
        query: str,
        latent_context: Optional[torch.Tensor] = None,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve with optional latent context from agents.
        
        This allows integrating latent memory state into retrieval,
        making it context-aware based on prior agent reasoning.
        """
        # Base retrieval
        results = self.retrieve(query, top_k=top_k * 2)
        
        if latent_context is not None and len(results.chunks) > 0:
            # Re-rank based on latent context similarity
            chunk_embeddings = torch.stack([
                self._chunk_embeddings.get(c.chunk_id, torch.zeros_like(latent_context))
                for c in results.chunks
            ])
            
            # Project latent to embedding space (simple linear projection)
            if latent_context.shape[-1] != chunk_embeddings.shape[-1]:
                # Dimension mismatch - use mean pooling
                projected = latent_context.mean(dim=-1, keepdim=True).expand_as(chunk_embeddings[0])
            else:
                projected = latent_context
            
            # Compute context-aware scores
            context_scores = torch.matmul(chunk_embeddings, projected.squeeze())
            
            # Combine with original scores
            original_scores = torch.tensor(results.scores, device=self.device)
            combined = 0.6 * original_scores + 0.4 * context_scores
            
            # Re-sort
            sorted_indices = combined.argsort(descending=True)[:top_k]
            
            return RetrievalResult(
                chunks=[results.chunks[i] for i in sorted_indices],
                scores=combined[sorted_indices].tolist(),
                query=query,
                metadata={"context_aware": True},
            )
        
        # Return top-k from base retrieval
        return RetrievalResult(
            chunks=results.chunks[:top_k],
            scores=results.scores[:top_k],
            query=query,
        )
    
    def update_index(self, doc_id: str) -> None:
        """Update index for a specific document"""
        doc = self.store.get_document(doc_id)
        if doc is None:
            return
        
        self._load_embedding_model()
        
        for chunk in doc.chunks:
            if self._embedding_model is not None:
                emb = self._embedding_model.encode(
                    chunk.text,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
                self._chunk_embeddings[chunk.chunk_id] = emb.to(self.device)
                
                # Add to matrix if not present
                if chunk.chunk_id not in self._chunk_ids:
                    self._chunk_ids.append(chunk.chunk_id)
                    if self._embedding_matrix is not None:
                        self._embedding_matrix = torch.cat([
                            self._embedding_matrix,
                            emb.unsqueeze(0).to(self.device),
                        ], dim=0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "indexed": self._indexed,
            "num_chunks_indexed": len(self._chunk_ids),
            "embedding_dim": self._embedding_matrix.shape[-1] if self._embedding_matrix is not None else 0,
            "using_semantic": self._embedding_model is not None,
            "using_tfidf": self._tfidf_vectorizer is not None,
        }
