"""
Domain Profiles for Semantic Routing

Defines domain characteristics for intelligent LoRA/pipeline selection.
"""

from dataclasses import dataclass, field
from enum import Enum
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
        # Keywords must be terms that are unlikely in ordinary prose. Words like
        # "if", "for", "while", "class", "return", "error", "fix", "method" and
        # "library" were here and fired on 164/400 and 127/400 GSM8K word
        # problems ("If there are 30 sheets...", "...for his drawing"), which is
        # what routed grade-school arithmetic to CodeExpert.
        keywords=[
            "code",
            "codebase",
            "compiler",
            "runtime",
            "python",
            "javascript",
            "typescript",
            "golang",
            "c++",
            "rust",
            "sql",
            "regex",
            "debug",
            "debugging",
            "refactor",
            "recursion",
            "algorithm",
            "data structure",
            "api",
            "endpoint",
            "database",
            "git",
            "docker",
            "unit test",
            "stack trace",
            "syntax error",
            "null pointer",
            "def ",
            "async",
            "await",
            "try:",
            "except:",
            "print(",
            "console.log",
            "()",
        ],
        negative_keywords=["patient", "diagnosis", "symptom", "medicine"],
    ),
    Domain.MATH: DomainProfile(
        domain=Domain.MATH,
        description="Mathematics, algebra, calculus, statistics, proofs",
        # Half symbolic, half word problem. The list used to be symbolic only
        # ("Find the derivative of sin(x²)"), so the centroid sat far from
        # grade-school arithmetic and GSM8K items scored closer to REASONING
        # than to MATH. Exemplars have to span the workload, not just the
        # prettiest examples of the subject.
        exemplar_prompts=[
            "Solve the quadratic equation x² - 5x + 6 = 0",
            "Find the derivative of sin(x²)",
            "Calculate the integral of e^x",
            "Prove that √2 is irrational",
            "What is the probability of rolling two sixes?",
            "Find the eigenvalues of this matrix",
            "Janet sells 6 eggs a day at $2 each. How much does she make in a week?",
            "A shirt costs $15 and is on sale for 20% off. What is the final price?",
            "Tom read 12 pages on Monday and twice as many on Tuesday. "
            "How many pages did he read in total?",
            "There are 30 students in a class. Three-fifths play sports. "
            "How many do not play sports?",
            "A train travels 60 miles in 1.5 hours. How far does it go in 4 hours?",
            "If 4 boxes hold 96 pencils, how many pencils are in 7 boxes?",
        ],
        keywords=[
            "solve",
            "calculate",
            "compute",
            "find",
            "prove",
            "derive",
            "evaluate",
            "equation",
            "formula",
            "theorem",
            "proof",
            "integral",
            "derivative",
            "limit",
            "sum",
            "product",
            "series",
            "matrix",
            "vector",
            "eigenvalue",
            "probability",
            "permutation",
            "combination",
            "factorial",
            "log",
            "ln",
            "sin",
            "cos",
            "tan",
            "sqrt",
            "^2",
            "²",
            "³",
            "x =",
            "y =",
            "f(x)",
            "∫",
            "∑",
            "∏",
            "lim",
            "→",
            "≤",
            "≥",
        ],
        negative_keywords=["patient", "diagnosis", "compiler"],
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
            "patient",
            "diagnosis",
            "symptom",
            "treatment",
            "disease",
            "condition",
            "medication",
            "drug",
            "dose",
            "prescription",
            "side effect",
            "anatomy",
            "physiology",
            "pathology",
            "pharmacology",
            "clinical",
            "blood",
            "heart",
            "lung",
            "liver",
            "kidney",
            "brain",
            "bone",
            "infection",
            "virus",
            "bacteria",
            "inflammation",
            "cancer",
            "tumor",
            "surgery",
            "therapy",
            "prognosis",
            "chronic",
            "acute",
            "mg",
            "ml",
            "oral",
            "injection",
            "intravenous",
        ],
        negative_keywords=["compiler", "def ", "javascript"],
    ),
    Domain.REASONING: DomainProfile(
        domain=Domain.REASONING,
        description="Logic, critical thinking, problem solving, analysis",
        # Formal logic plus the science and commonsense multiple choice that
        # actually arrives here (ARC and similar). Purely formal exemplars left
        # the centroid unable to recognise "Which of these is a physical
        # change?" as anything but GENERAL.
        exemplar_prompts=[
            "If all A are B and some B are C, what can we conclude?",
            "Analyze the logical fallacy in this argument",
            "Compare and contrast these two approaches",
            "Identify the assumptions in this statement",
            "Evaluate the validity of this conclusion",
            "Which of these is an example of a physical change?",
            "Which statement best explains why ice floats on water?",
            "A student observes that plants near a window grow taller. "
            "What is the most likely reason?",
            "Which of the following best describes the role of the sun in the water cycle?",
            "What happens to the volume of a gas when it is heated?",
            "Which tool would be most useful for measuring mass?",
        ],
        # Deliberately empty: "why", "how", "explain", "method", "problem",
        # "solution" and friends were here and match essentially every question
        # in every domain. This domain is carried by its exemplars.
        keywords=[],
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
