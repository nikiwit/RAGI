"""
Type definitions and enums for the RAGI system.
"""

from enum import Enum

class QueryType(Enum):
    """Enum for different types of queries."""
    FACTUAL = "factual"           # Asking for specific facts
    PROCEDURAL = "procedural"     # How to do something
    CONCEPTUAL = "conceptual"     # Understanding concepts
    EXPLORATORY = "exploratory"   # Open-ended exploration
    COMPARATIVE = "comparative"   # Comparing things
    CONVERSATIONAL = "conversational"  # Social conversation
    COMMAND = "command"           # System commands
    IDENTITY = "identity"         # Questions about the assistant itself
    # Domain-specific query types (customizable for your use case)
    ACADEMIC = "academic"         # Academic-related queries (courses, exams, learning)
    ADMINISTRATIVE = "administrative"  # Administrative processes
    FINANCIAL = "financial"       # Financial matters, payments
    UNKNOWN = "unknown"           # Unclassified

class DocumentRelevance(Enum):
    """Enum for document relevance levels."""
    HIGH = "high"       # Directly relevant
    MEDIUM = "medium"   # Somewhat relevant
    LOW = "low"         # Tangentially relevant
    NONE = "none"       # Not relevant

class RetrievalStrategy(Enum):
    """Enum for retrieval strategies."""
    SEMANTIC = "semantic"         # Semantic similarity search
    KEYWORD = "keyword"           # Keyword-based search
    HYBRID = "hybrid"             # Combined semantic and keyword
    MMR = "mmr"                   # Maximum Marginal Relevance for diversity
    FAQ_MATCH = "faq_match"       # Direct matching for FAQ content