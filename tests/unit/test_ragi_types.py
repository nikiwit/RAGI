"""
Unit tests for ragi_types module.
"""

import pytest
from ragi_types import (
    QueryType,
    DocumentRelevance,
    RetrievalStrategy,
)


class TestQueryType:
    """Test QueryType enum."""

    def test_all_query_types_exist(self):
        """Test that all expected query types are defined."""
        expected_types = [
            "FACTUAL",
            "PROCEDURAL",
            "CONCEPTUAL",
            "EXPLORATORY",
            "COMPARATIVE",
            "CONVERSATIONAL",
            "COMMAND",
            "IDENTITY",
            "ACADEMIC",
            "ADMINISTRATIVE",
            "FINANCIAL",
            "UNKNOWN",
        ]

        for type_name in expected_types:
            assert hasattr(QueryType, type_name)

    def test_query_type_values(self):
        """Test query type enum values."""
        assert QueryType.FACTUAL.value == "factual"
        assert QueryType.PROCEDURAL.value == "procedural"
        assert QueryType.CONVERSATIONAL.value == "conversational"
        assert QueryType.COMMAND.value == "command"

    def test_query_type_comparison(self):
        """Test query type comparison."""
        assert QueryType.FACTUAL == QueryType.FACTUAL
        assert QueryType.FACTUAL != QueryType.PROCEDURAL

    def test_query_type_in_enum(self):
        """Test query type membership."""
        assert QueryType.FACTUAL in QueryType
        assert QueryType.UNKNOWN in QueryType

    def test_query_type_iteration(self):
        """Test iterating over query types."""
        query_types = list(QueryType)
        assert len(query_types) > 0
        assert all(isinstance(qt, QueryType) for qt in query_types)

    def test_domain_specific_types(self):
        """Test domain-specific query types."""
        assert QueryType.ACADEMIC in QueryType
        assert QueryType.ADMINISTRATIVE in QueryType
        assert QueryType.FINANCIAL in QueryType


class TestDocumentRelevance:
    """Test DocumentRelevance enum."""

    def test_all_relevance_levels_exist(self):
        """Test that all relevance levels are defined."""
        expected_levels = ["HIGH", "MEDIUM", "LOW", "NONE"]

        for level in expected_levels:
            assert hasattr(DocumentRelevance, level)

    def test_relevance_values(self):
        """Test relevance enum values."""
        assert DocumentRelevance.HIGH.value == "high"
        assert DocumentRelevance.MEDIUM.value == "medium"
        assert DocumentRelevance.LOW.value == "low"
        assert DocumentRelevance.NONE.value == "none"

    def test_relevance_comparison(self):
        """Test relevance level comparison."""
        assert DocumentRelevance.HIGH == DocumentRelevance.HIGH
        assert DocumentRelevance.HIGH != DocumentRelevance.LOW

    def test_relevance_iteration(self):
        """Test iterating over relevance levels."""
        relevance_levels = list(DocumentRelevance)
        assert len(relevance_levels) == 4
        assert all(isinstance(rel, DocumentRelevance) for rel in relevance_levels)


class TestRetrievalStrategy:
    """Test RetrievalStrategy enum."""

    def test_retrieval_strategy_exists(self):
        """Test that RetrievalStrategy enum exists."""
        assert RetrievalStrategy is not None

    def test_strategy_values(self):
        """Test that strategies have expected values."""
        # This test depends on what's actually in the enum
        strategies = list(RetrievalStrategy)
        assert len(strategies) > 0
        assert all(isinstance(s, RetrievalStrategy) for s in strategies)


class TestEnumUsage:
    """Test practical enum usage scenarios."""

    def test_query_type_string_conversion(self):
        """Test converting query type to string."""
        query_type = QueryType.FACTUAL
        assert str(query_type) == "QueryType.FACTUAL"
        assert query_type.value == "factual"

    def test_query_type_from_value(self):
        """Test creating query type from value."""
        query_type = QueryType("factual")
        assert query_type == QueryType.FACTUAL

    def test_relevance_string_conversion(self):
        """Test converting relevance to string."""
        relevance = DocumentRelevance.HIGH
        assert relevance.value == "high"

    def test_invalid_query_type_raises_error(self):
        """Test that invalid query type raises ValueError."""
        with pytest.raises(ValueError):
            QueryType("invalid_type")

    def test_invalid_relevance_raises_error(self):
        """Test that invalid relevance raises ValueError."""
        with pytest.raises(ValueError):
            DocumentRelevance("invalid_relevance")

    def test_enum_immutability(self):
        """Test that enums are immutable."""
        with pytest.raises(AttributeError):
            QueryType.FACTUAL = "something_else"

    def test_enum_hashable(self):
        """Test that enums can be used as dict keys."""
        query_dict = {
            QueryType.FACTUAL: "factual_handler",
            QueryType.PROCEDURAL: "procedural_handler",
        }

        assert query_dict[QueryType.FACTUAL] == "factual_handler"
        assert len(query_dict) == 2

    def test_enum_in_set(self):
        """Test that enums can be used in sets."""
        query_set = {QueryType.FACTUAL, QueryType.PROCEDURAL, QueryType.FACTUAL}

        assert len(query_set) == 2
        assert QueryType.FACTUAL in query_set
        assert QueryType.CONCEPTUAL not in query_set
