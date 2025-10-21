"""
Pytest configuration and shared fixtures for RAGI tests.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from ragi_types import QueryType, DocumentRelevance


# ============================================================================
# Session-scoped fixtures (run once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session")
def sample_documents_dir(test_data_dir: Path) -> Generator[Path, None, None]:
    """Create temporary directory with sample documents."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ragi_test_docs_"))

    # Create sample documents
    (temp_dir / "sample1.txt").write_text(
        "Machine learning is a subset of artificial intelligence. "
        "It focuses on building systems that learn from data."
    )

    (temp_dir / "sample2.txt").write_text(
        "Deep learning uses neural networks with multiple layers. "
        "It is particularly effective for image and speech recognition."
    )

    (temp_dir / "sample3.md").write_text(
        "# Python Programming\n\n"
        "Python is a high-level programming language. "
        "It is known for its simplicity and readability."
    )

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Return test configuration dictionary."""
    return {
        "chunk_size": 100,
        "chunk_overlap": 20,
        "retriever_k": 3,
        "search_type": "hybrid",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "llm_model": "qwen2.5:3b-instruct",
    }


# ============================================================================
# Function-scoped fixtures (run for each test)
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ragi_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_data_dir(temp_dir: Path) -> Path:
    """Create temporary data directory."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def temp_vector_store(temp_dir: Path) -> Path:
    """Create temporary vector store directory."""
    vector_dir = temp_dir / "vector_store"
    vector_dir.mkdir(exist_ok=True)
    return vector_dir


@pytest.fixture
def mock_config(temp_data_dir: Path, temp_vector_store: Path, monkeypatch):
    """Mock configuration with temporary paths."""
    monkeypatch.setattr(config, "DATA_PATH", str(temp_data_dir))
    monkeypatch.setattr(config, "PERSIST_PATH", str(temp_vector_store))
    monkeypatch.setattr(config, "CHUNK_SIZE", 100)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 20)
    return config


@pytest.fixture
def sample_text() -> str:
    """Return sample text for testing."""
    return (
        "Artificial intelligence (AI) is intelligence demonstrated by machines. "
        "Machine learning is a subset of AI that focuses on learning from data. "
        "Deep learning is a subset of machine learning using neural networks."
    )


@pytest.fixture
def sample_queries() -> Dict[str, QueryType]:
    """Return sample queries with expected types."""
    return {
        "What is machine learning?": QueryType.FACTUAL,
        "How do I train a neural network?": QueryType.PROCEDURAL,
        "Explain deep learning": QueryType.CONCEPTUAL,
        "Tell me about AI": QueryType.EXPLORATORY,
        "help": QueryType.COMMAND,
        "who are you": QueryType.IDENTITY,
        "hello": QueryType.CONVERSATIONAL,
    }


@pytest.fixture
def sample_document_metadata() -> Dict[str, Any]:
    """Return sample document metadata."""
    return {
        "source": "/path/to/document.pdf",
        "page": 1,
        "title": "Sample Document",
        "content_type": "kb_page",
        "timestamp": 1234567890,
    }


@pytest.fixture
def mock_ollama_response() -> str:
    """Return mock Ollama response."""
    return "This is a sample response from the language model."


# ============================================================================
# Markers and skip conditions
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers",
        "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Auto-mark tests in integration/ directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# ============================================================================
# Helper functions for tests
# ============================================================================

def create_test_document(path: Path, content: str, filename: str = "test.txt"):
    """Helper to create a test document."""
    file_path = path / filename
    file_path.write_text(content)
    return file_path


def assert_valid_query_type(query_type):
    """Assert that query_type is valid."""
    assert isinstance(query_type, QueryType)
    assert query_type in QueryType


def assert_valid_document_relevance(relevance):
    """Assert that relevance is valid."""
    assert isinstance(relevance, DocumentRelevance)
    assert relevance in DocumentRelevance


# ============================================================================
# Skip conditions
# ============================================================================

# Skip if Ollama is not running
def is_ollama_running() -> bool:
    """Check if Ollama is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


skip_if_no_ollama = pytest.mark.skipif(
    not is_ollama_running(),
    reason="Ollama is not running"
)


# Skip if no GPU available
def has_gpu() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available() or (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        )
    except:
        return False


skip_if_no_gpu = pytest.mark.skipif(
    not has_gpu(),
    reason="GPU not available"
)


# Skip if internet not available
def has_internet() -> bool:
    """Check if internet is available."""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False


skip_if_no_internet = pytest.mark.skipif(
    not has_internet(),
    reason="Internet not available"
)
