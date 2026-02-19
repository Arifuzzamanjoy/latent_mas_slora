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
    FINANCE = "finance"
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
            "Which cranial nerve is responsible for taste sensation?",
            "What is the drug of choice for absence seizures?",
            "A 45-year-old woman presents with fatigue and weight gain",
            "Which protein is mutated in sickle cell disease?",
            "What is the most likely diagnosis for elevated TSH?",
        ],
        keywords=[
            "patient", "diagnosis", "symptom", "treatment", "disease", "condition",
            "medication", "drug", "dose", "prescription", "side effect", "side effects",
            "anatomy", "physiology", "pathology", "pharmacology", "clinical",
            "blood", "heart", "lung", "liver", "kidney", "brain", "bone",
            "infection", "virus", "bacteria", "inflammation", "cancer", "tumor",
            "surgery", "therapy", "prognosis", "chronic", "acute",
            "mg", "ml", "iv", "oral", "injection",
            # Medical exam terms
            "cranial", "nerve", "tongue", "taste", "sensation", "artery", "vein",
            "seizure", "hypertension", "diabetes", "thyroid", "TSH", "hemoglobin",
            "mutation", "protein", "enzyme", "deficiency", "syndrome",
            "presents with", "year-old", "history of", "most likely", "drug of choice",
            "laboratory", "elevated", "decreased", "normal", "abnormal",
            # Singular and plural drug names
            "antibiotic", "antibiotics", "painkiller", "painkillers", "vaccine", "vaccines",
        ],
        negative_keywords=["code", "algorithm", "function", "debug", "def ", "python", "javascript"],
        weight=1.2,  # Boost medical domain weight
    ),
    
    Domain.FINANCE: DomainProfile(
        domain=Domain.FINANCE,
        description="Finance, cryptocurrency, trading, investing, markets, blockchain",
        exemplar_prompts=[
            "What is the current price of Bitcoin?",
            "Tell me about TRON TRX cryptocurrency",
            "What is Ethereum and how does it work?",
            "What is the market cap of BTC?",
            "Explain blockchain technology",
            "What are the top cryptocurrencies by market cap?",
            "How does Bitcoin mining work?",
            "What is the total supply of Dogecoin?",
            "Compare Solana and Cardano",
            "What is DeFi and yield farming?",
            "How do I stake cryptocurrency?",
            "What is a crypto wallet?",
            "Explain NFTs and their use cases",
            "What is the stock price of Apple?",
            "How do trading bots work?",
        ],
        keywords=[
            # Cryptocurrency terms
            "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
            "blockchain", "coin", "token", "wallet", "mining", "staking",
            "defi", "nft", "altcoin", "hodl", "airdrop", "ico", "ido",
            # Popular cryptocurrencies
            "tron", "trx", "solana", "sol", "cardano", "ada", "dogecoin", "doge",
            "ripple", "xrp", "polkadot", "dot", "avalanche", "avax", "bnb",
            "polygon", "matic", "chainlink", "link", "litecoin", "ltc",
            "shiba", "pepe", "usdt", "usdc", "stablecoin",
            # Finance terms
            "stock", "trading", "invest", "market", "price", "exchange",
            "portfolio", "dividend", "bull", "bear", "volume", "liquidity",
            "market cap", "circulating supply", "total supply", "max supply",
            # Platforms
            "binance", "coinbase", "kraken", "uniswap", "opensea",
        ],
        negative_keywords=["patient", "diagnosis", "symptom", "medicine", "treatment", "disease"],
        weight=1.3,  # High priority for finance/crypto queries
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
