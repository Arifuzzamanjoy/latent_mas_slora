"""
LatentMAS-SLoRA Evaluation Suite

Comprehensive evaluation comparing:
1. Traditional 4-Agent RAG (text-based reasoning)
2. LatentMAS-SLoRA (latent collaboration + dynamic LoRA)

Using MIRAGE (Medical Information Retrieval-Augmented Generation Evaluation) benchmark.

Evaluation Methods:
- ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) for lexical similarity
- BERTScore for semantic similarity
- MCQ Accuracy for answer correctness
"""

from pathlib import Path

EVALUATION_DIR = Path(__file__).parent

# Lazy imports to avoid circular dependencies
__all__ = [
    "EVALUATION_DIR",
    # Systems
    "Traditional4AgentRAG",
    "LatentMASEvaluator",
    # Dataset
    "MIRAGEDataset",
    "MIRAGE_SAMPLE_QUESTIONS",
    # Metrics
    "RougeEvaluator",
    "BERTScoreEvaluator",
    "ComprehensiveEvaluator",
    "AnswerExtractor",
]

def __getattr__(name):
    """Lazy loading of evaluation components."""
    if name == "Traditional4AgentRAG":
        from .traditional_4agent_rag import Traditional4AgentRAG
        return Traditional4AgentRAG
    elif name == "LatentMASEvaluator":
        from .full_latent_mas import LatentMASEvaluator
        return LatentMASEvaluator
    elif name == "MIRAGEDataset":
        from .mirage_dataset import MIRAGEDataset
        return MIRAGEDataset
    elif name == "MIRAGE_SAMPLE_QUESTIONS":
        from .mirage_dataset import MIRAGE_SAMPLE_QUESTIONS
        return MIRAGE_SAMPLE_QUESTIONS
    elif name == "RougeEvaluator":
        from .evaluation_metrics import RougeEvaluator
        return RougeEvaluator
    elif name == "BERTScoreEvaluator":
        from .evaluation_metrics import BERTScoreEvaluator
        return BERTScoreEvaluator
    elif name == "ComprehensiveEvaluator":
        from .evaluation_metrics import ComprehensiveEvaluator
        return ComprehensiveEvaluator
    elif name == "AnswerExtractor":
        from .evaluation_metrics import AnswerExtractor
        return AnswerExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
