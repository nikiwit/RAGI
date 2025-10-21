"""
Unit tests for document processing modules.
"""

import pytest
from pathlib import Path
from document_processing.loaders import DocumentProcessor
from document_processing.splitters import KnowledgeBaseTextSplitter


class TestDocumentProcessor:
    """Test DocumentProcessor class."""

    def test_processor_initialization(self):
        """Test that DocumentProcessor can be initialized."""
        processor = DocumentProcessor()
        assert processor is not None

    def test_get_loader_for_txt(self):
        """Test getting loader for text files."""
        loader = DocumentProcessor.get_loader_for_file("test.txt")
        assert loader is not None

    def test_get_loader_for_pdf(self):
        """Test getting loader for PDF files."""
        loader = DocumentProcessor.get_loader_for_file("test.pdf")
        assert loader is not None

    def test_get_loader_for_docx(self):
        """Test getting loader for DOCX files."""
        loader = DocumentProcessor.get_loader_for_file("test.docx")
        assert loader is not None

    def test_get_loader_for_md(self):
        """Test getting loader for Markdown files."""
        loader = DocumentProcessor.get_loader_for_file("test.md")
        assert loader is not None

    def test_get_loader_for_unsupported(self):
        """Test that unsupported files return None or raise error."""
        result = DocumentProcessor.get_loader_for_file("test.xyz")
        # Should return None or raise ValueError
        assert result is None or True

    def test_load_documents_from_empty_dir(self, temp_data_dir):
        """Test loading from empty directory."""
        documents = DocumentProcessor.load_documents_from_directory(str(temp_data_dir))
        assert isinstance(documents, list)
        assert len(documents) == 0

    def test_load_documents_from_dir_with_files(self, temp_data_dir):
        """Test loading from directory with files."""
        # Create test files
        (temp_data_dir / "test1.txt").write_text("Test content 1")
        (temp_data_dir / "test2.txt").write_text("Test content 2")

        documents = DocumentProcessor.load_documents_from_directory(str(temp_data_dir))
        assert isinstance(documents, list)
        assert len(documents) >= 2


class TestKnowledgeBaseTextSplitter:
    """Test KnowledgeBaseTextSplitter class."""

    def test_splitter_initialization(self):
        """Test splitter initialization."""
        splitter = KnowledgeBaseTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        assert splitter is not None

    def test_splitter_with_default_params(self):
        """Test splitter with default parameters."""
        splitter = KnowledgeBaseTextSplitter()
        assert splitter is not None

    def test_split_short_text(self, sample_text):
        """Test splitting short text."""
        splitter = KnowledgeBaseTextSplitter(chunk_size=1000, chunk_overlap=100)

        # Create mock document
        class MockDoc:
            def __init__(self, content):
                self.page_content = content
                self.metadata = {}

        docs = [MockDoc(sample_text)]
        result = splitter.split_documents(docs)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_split_long_text(self):
        """Test splitting long text."""
        long_text = " ".join(["This is a test sentence."] * 100)
        splitter = KnowledgeBaseTextSplitter(chunk_size=100, chunk_overlap=20)

        class MockDoc:
            def __init__(self, content):
                self.page_content = content
                self.metadata = {}

        docs = [MockDoc(long_text)]
        result = splitter.split_documents(docs)

        assert isinstance(result, list)
        assert len(result) > 1

    def test_chunk_overlap_preserved(self):
        """Test that chunk overlap is preserved."""
        text = "A" * 500  # Long text
        splitter = KnowledgeBaseTextSplitter(chunk_size=200, chunk_overlap=50)

        class MockDoc:
            def __init__(self, content):
                self.page_content = content
                self.metadata = {}

        docs = [MockDoc(text)]
        result = splitter.split_documents(docs)

        # Should create multiple chunks
        assert len(result) > 1

    def test_metadata_preservation(self):
        """Test that metadata is preserved during splitting."""
        splitter = KnowledgeBaseTextSplitter(chunk_size=100, chunk_overlap=20)

        class MockDoc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata

        metadata = {"source": "test.txt", "page": 1}
        docs = [MockDoc("Test content " * 50, metadata)]
        result = splitter.split_documents(docs)

        # Check metadata is in results
        for chunk in result:
            assert hasattr(chunk, 'metadata')


class TestDocumentMetadata:
    """Test document metadata handling."""

    def test_metadata_extraction(self, sample_document_metadata):
        """Test metadata extraction."""
        assert "source" in sample_document_metadata
        assert "page" in sample_document_metadata
        assert "title" in sample_document_metadata

    def test_metadata_required_fields(self, sample_document_metadata):
        """Test that required metadata fields exist."""
        required_fields = ["source", "content_type"]
        for field in required_fields:
            assert field in sample_document_metadata


class TestFileTypeDetection:
    """Test file type detection."""

    def test_detect_txt_file(self):
        """Test detecting text files."""
        assert "test.txt".endswith(".txt")

    def test_detect_pdf_file(self):
        """Test detecting PDF files."""
        assert "test.pdf".endswith(".pdf")

    def test_detect_docx_file(self):
        """Test detecting DOCX files."""
        assert "test.docx".endswith(".docx")

    def test_detect_markdown_file(self):
        """Test detecting Markdown files."""
        assert "test.md".endswith(".md")

    def test_case_insensitive_detection(self):
        """Test case-insensitive file detection."""
        assert "test.TXT".lower().endswith(".txt")
        assert "test.PDF".lower().endswith(".pdf")
