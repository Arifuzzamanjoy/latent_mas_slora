"""
RAG Pipeline - Integrates retrieval with multi-agent reasoning

Combines document retrieval with LatentMAS pipeline for
context-aware, grounded responses.
"""

import torch
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from .document_store import DocumentStore, Document
from .retriever import RAGRetriever, RetrievalResult


@dataclass
class RAGResult:
    """Result from RAG-enhanced pipeline"""
    question: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_scores: List[float]
    context_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """
    RAG-Enhanced Pipeline for Document Intelligence.
    
    Integrates with LatentMAS for:
    - Document-grounded responses
    - Citation tracking
    - Multi-hop reasoning over documents
    
    Example:
        rag = RAGPipeline(system)
        rag.load_documents("/path/to/docs")
        result = rag.query("What does the report say about Q3 earnings?")
    """
    
    def __init__(
        self,
        system=None,  # LatentMASSystem - optional for standalone use
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        max_context_length: int = 3000,
    ):
        self.system = system
        self.top_k = top_k
        self.max_context_length = max_context_length
        
        # Initialize document store and retriever
        self.store = DocumentStore(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        device = system.device if system else "cuda"
        self.retriever = RAGRetriever(
            store=self.store,
            embedding_model=embedding_model,
            device=device,
        )
        
        self._indexed = False
    
    def load_documents(
        self,
        source: Union[str, Path, List[str], List[Path]],
        **kwargs,
    ) -> int:
        """
        Load documents from files or directories.
        
        Args:
            source: File path, directory path, or list of paths
            **kwargs: Additional arguments for document loading
            
        Returns:
            Number of documents loaded
        """
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.is_dir():
                docs = self.store.load_directory(source, **kwargs)
                return len(docs)
            else:
                doc = self.store.load_file(source, **kwargs)
                return 1 if doc else 0
        elif isinstance(source, list):
            count = 0
            for path in source:
                count += self.load_documents(path, **kwargs)
            return count
        else:
            raise ValueError(f"Invalid source type: {type(source)}")
    
    def add_text(
        self,
        text: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Add text content directly"""
        return self.store.add_document(text, title=title, metadata=metadata)
    
    def build_index(self, force: bool = False) -> None:
        """Build or rebuild the retrieval index"""
        self.retriever.build_index(force=force)
        self._indexed = True
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results (default: self.top_k)
            filters: Metadata filters
            
        Returns:
            RetrievalResult with chunks and scores
        """
        if not self._indexed:
            self.build_index()
        
        return self.retriever.retrieve(
            query=query,
            top_k=top_k or self.top_k,
            filters=filters,
        )
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_citations: bool = True,
        pipeline: str = "hierarchical",
        **kwargs,
    ) -> RAGResult:
        """
        Run RAG-enhanced query through the multi-agent system.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            include_citations: Whether to include source citations
            pipeline: Pipeline type for system.run()
            **kwargs: Additional arguments for system.run()
            
        Returns:
            RAGResult with answer and sources
        """
        # Retrieve relevant context
        retrieval = self.retrieve(question, top_k=top_k)
        
        # Build context string
        context = retrieval.get_context(max_length=self.max_context_length)
        
        # Construct augmented prompt
        if context:
            augmented_question = self._build_rag_prompt(question, context, include_citations)
        else:
            augmented_question = question
        
        # Run through multi-agent system if available
        if self.system is not None:
            result = self.system.run(augmented_question, pipeline=pipeline, **kwargs)
            answer = result.final_answer
            metadata = {
                "agent_outputs": result.agent_outputs,
                "total_tokens": result.total_tokens,
                "latency_ms": result.total_latency_ms,
            }
        else:
            # Standalone mode - return context only
            answer = f"Retrieved context:\n\n{context}"
            metadata = {}
        
        return RAGResult(
            question=question,
            answer=answer,
            retrieved_chunks=[
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "score": score,
                }
                for chunk, score in zip(retrieval.chunks, retrieval.scores)
            ],
            retrieval_scores=retrieval.scores,
            context_used=context,
            metadata=metadata,
        )
    
    def _build_rag_prompt(
        self,
        question: str,
        context: str,
        include_citations: bool,
    ) -> str:
        """Build RAG-augmented prompt"""
        citation_instruction = (
            "\n\nWhen answering, cite your sources using [Source: doc_id] format."
            if include_citations else ""
        )
        
        return f"""Use the following retrieved context to answer the question.
If the context doesn't contain relevant information, say so and provide your best answer.

RETRIEVED CONTEXT:
{context}

QUESTION: {question}{citation_instruction}

ANSWER:"""
    
    def multi_hop_query(
        self,
        question: str,
        max_hops: int = 3,
        **kwargs,
    ) -> RAGResult:
        """
        Multi-hop reasoning over documents.
        
        Iteratively retrieves and reasons until answer is found
        or max hops reached.
        """
        current_query = question
        all_chunks = []
        all_scores = []
        all_context = []
        
        for hop in range(max_hops):
            # Retrieve for current query
            retrieval = self.retrieve(current_query, top_k=self.top_k)
            
            # Accumulate results
            for chunk, score in zip(retrieval.chunks, retrieval.scores):
                if chunk not in all_chunks:
                    all_chunks.append(chunk)
                    all_scores.append(score)
            
            context = retrieval.get_context(max_length=self.max_context_length // max_hops)
            all_context.append(context)
            
            # Generate follow-up query if system available
            if self.system is not None and hop < max_hops - 1:
                follow_up_prompt = f"""Based on this context about "{question}":

{context}

What additional information would help answer the original question?
If you have enough information, respond with "SUFFICIENT".
Otherwise, provide a follow-up search query."""

                result = self.system.run(follow_up_prompt, pipeline="hierarchical")
                
                if "SUFFICIENT" in result.final_answer.upper():
                    break
                
                # Use response as next query
                current_query = result.final_answer.strip()
        
        # Final answer with all accumulated context
        combined_context = "\n\n---\n\n".join(all_context)
        
        return self.query(
            question,
            **kwargs,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "store": self.store.get_stats(),
            "retriever": self.retriever.get_stats(),
            "indexed": self._indexed,
            "top_k": self.top_k,
        }
    
    def clear(self) -> None:
        """Clear all documents and reset index"""
        self.store.clear()
        self._indexed = False
