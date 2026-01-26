"""
Domain Profiles for Semantic Routing

Defines domain characteristics for intelligent LoRA/pipeline selection.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List


class Domain(Enum):
    """Supported domains for routing"""
    CODE = "code"
    MATH = "math"
    MEDICAL = "medical"
    REASONING = "reasoning"
    GENERAL = "general"


@dataclass
class DomainProfile:
    """
    Domain profile for semantic routing.
    
    Contains exemplar prompts for embedding similarity
    and keywords for fast matching.
    """
    domain: Domain
    description: str
    exemplar_prompts: List[str]
    keywords: List[str]
    negative_keywords: List[str] = field(default_factory=list)
    weight: float = 1.0

    @property
    def name(self) -> str:
        return self.domain.value


# Domain profiles with exemplar prompts for embedding similarity
DOMAIN_PROFILES = {
    Domain.CODE: DomainProfile(
        domain=Domain.CODE,
        description="Programming, software development, algorithms, debugging",
        exemplar_prompts=[
            "Write a Python function to sort a list",
            "Debug this JavaScript code",
            "Implement a binary search tree",
            "How do I use async/await?",
            "Create a REST API endpoint",
            "Fix the bug in this code snippet",
            "Optimize this algorithm for better time complexity",
            "Write unit tests for this function",
            "Implement the observer design pattern",
            "Refactor this code for better readability",
        ],
        keywords=[
            "code", "function", "class", "method", "variable", "loop", "array",
            "python", "javascript", "java", "c++", "rust", "golang", "typescript",
            "debug", "error", "bug", "fix", "implement", "algorithm", "data structure",
            "api", "endpoint", "database", "sql", "git", "docker",
            "import", "export", "module", "package", "library", "framework",
            "def ", "return", "if ", "for ", "while ", "class ", "async", "await",
            "try:", "except:", "print(", "console.log",
        ],
        negative_keywords=["patient", "diagnosis", "symptom", "medicine", "治疗"],
    ),
    
    Domain.MATH: DomainProfile(
        domain=Domain.MATH,
        description="Mathematics, algebra, calculus, statistics, proofs",
        exemplar_prompts=[
            "Solve the quadratic equation x² - 5x + 6 = 0",
            "Find the derivative of sin(x²)",
            "Calculate the integral of e^x",
            "Prove that √2 is irrational",
            "Find all values of x where log₂(x) = 3",
            "What is the probability of rolling two sixes?",
            "Simplify the expression (x² - 4)/(x - 2)",
            "Find the eigenvalues of this matrix",
            "Solve the system of linear equations",
            "Calculate the limit as x approaches infinity",
        ],
        keywords=[
            "solve", "calculate", "compute", "find", "prove", "derive", "evaluate",
            "equation", "formula", "theorem", "proof", "integral", "derivative",
            "limit", "sum", "product", "series", "matrix", "vector", "eigenvalue",
            "probability", "permutation", "combination", "factorial",
            "log", "ln", "sin", "cos", "tan", "sqrt", "^2", "²", "³",
            "x =", "y =", "f(x)", "∫", "∑", "∏", "lim", "→", "≤", "≥",
        ],
        negative_keywords=["code", "function", "debug", "patient", "class "],
    ),
    
    Domain.MEDICAL: DomainProfile(
        domain=Domain.MEDICAL,
        description="Medicine, healthcare, diagnosis, treatment, pharmacology",
        exemplar_prompts=[
            "What are the symptoms of diabetes?",
            "Explain the mechanism of action of metformin",
            "Differential diagnosis for chest pain",
            "What is the treatment for hypertension?",
            "Describe the anatomy of the heart",
            "Side effects of ibuprofen",
            "How does insulin regulate blood sugar?",
            "Explain the stages of wound healing",
            "What causes autoimmune diseases?",
            "Interpret these blood test results",
        ],
        keywords=[
            "patient", "diagnosis", "symptom", "treatment", "disease", "condition",
            "medication", "drug", "dose", "prescription", "side effect",
            "anatomy", "physiology", "pathology", "pharmacology", "clinical",
            "blood", "heart", "lung", "liver", "kidney", "brain", "bone",
            "infection", "virus", "bacteria", "inflammation", "cancer", "tumor",
            "surgery", "therapy", "prognosis", "chronic", "acute",
            "mg", "ml", "iv", "oral", "injection",
        ],
        negative_keywords=["code", "algorithm", "function", "debug", "def "],
    ),
    
    Domain.REASONING: DomainProfile(
        domain=Domain.REASONING,
        description="Logic, critical thinking, problem solving, analysis",
        exemplar_prompts=[
            "If all A are B and some B are C, what can we conclude?",
            "Analyze the logical fallacy in this argument",
            "What is the best strategy for this problem?",
            "Compare and contrast these two approaches",
            "What are the pros and cons of this decision?",
            "Identify the assumptions in this statement",
            "How would you approach solving this puzzle?",
            "Evaluate the validity of this conclusion",
            "What evidence supports this claim?",
            "Break down this complex problem into steps",
        ],
        keywords=[
            "why", "how", "explain", "analyze", "evaluate", "compare", "contrast",
            "reason", "logic", "argument", "premise", "conclusion", "therefore",
            "assume", "hypothesis", "evidence", "valid", "invalid", "fallacy",
            "strategy", "approach", "method", "solution", "problem", "puzzle",
            "pros", "cons", "advantage", "disadvantage", "trade-off",
        ],
        negative_keywords=[],
        weight=0.8,  # Lower weight, acts as fallback
    ),
    
    Domain.GENERAL: DomainProfile(
        domain=Domain.GENERAL,
        description="General knowledge and conversation",
        exemplar_prompts=[
            "Tell me about the history of Rome",
            "What is the capital of France?",
            "Explain how airplanes fly",
            "Who wrote Romeo and Juliet?",
            "What is climate change?",
        ],
        keywords=[],
        negative_keywords=[],
        weight=0.5,  # Lowest priority
    ),
}
