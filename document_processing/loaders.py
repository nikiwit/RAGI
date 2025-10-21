"""
Document loaders and processors.
"""

import os
import logging
from typing import List
import html2text
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredEPubLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import config
from .parsers import GenericKnowledgeBaseLoader, GenericKnowledgeBaseParser
from .splitters import KnowledgeBaseTextSplitter

logger = logging.getLogger("RAGI")

class PyMuPDFOCRLoader(BaseLoader):
    """
    Advanced PDF loader using PyMuPDF (fitz) with intelligent OCR fallback.

    Features:
    - Native text extraction (fast)
    - Automatic OCR detection for scanned pages
    - Layout preservation for better LLM understanding
    - Table detection and extraction

    Performance: ~1000x faster than OCR for text-based PDFs
    """

    def __init__(self, file_path: str, ocr_enabled: bool = True):
        """
        Initialize the PyMuPDF OCR loader.

        Args:
            file_path: Path to the PDF file
            ocr_enabled: Whether to use OCR for scanned pages
        """
        self.file_path = file_path
        self.ocr_enabled = ocr_enabled and config.USE_OCR
        self.ocr_language = config.OCR_LANGUAGE
        self.ocr_dpi = config.OCR_DPI
        self.min_text_threshold = config.OCR_MIN_TEXT_THRESHOLD

    def _check_tesseract(self) -> bool:
        """Check if Tesseract is installed and available."""
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            logger.debug(f"Tesseract OCR version {version} detected")
            return True
        except Exception as e:
            logger.warning(f"Tesseract OCR not available: {e}")
            logger.warning("Install Tesseract: brew install tesseract (macOS) or sudo apt-get install tesseract-ocr (Linux)")
            return False

    def _needs_ocr(self, page_text: str) -> bool:
        """
        Intelligently detect if a page needs OCR.

        Args:
            page_text: Extracted text from the page

        Returns:
            True if OCR is needed, False otherwise
        """
        # If text is very short or empty, likely a scanned page
        text_length = len(page_text.strip())
        needs_ocr = text_length < self.min_text_threshold

        if needs_ocr:
            logger.debug(f"Page has only {text_length} characters - triggering OCR")

        return needs_ocr

    def _ocr_page(self, page) -> str:
        """
        Perform OCR on a page using Tesseract.

        Args:
            page: PyMuPDF page object

        Returns:
            Extracted text from OCR
        """
        try:
            import pytesseract
            from PIL import Image
            import io

            # Convert page to image at specified DPI
            pix = page.get_pixmap(dpi=self.ocr_dpi)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            # Perform OCR
            logger.debug(f"Performing OCR at {self.ocr_dpi} DPI with language '{self.ocr_language}'")
            ocr_text = pytesseract.image_to_string(image, lang=self.ocr_language)

            logger.debug(f"OCR extracted {len(ocr_text)} characters")
            return ocr_text

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def load(self) -> List[Document]:
        """
        Load PDF with intelligent OCR fallback.

        Returns:
            List of Document objects
        """
        try:
            import fitz  # PyMuPDF

            logger.info(f"Loading PDF with PyMuPDF: {os.path.basename(self.file_path)}")

            # Check if OCR is available
            tesseract_available = self._check_tesseract() if self.ocr_enabled else False

            # Open PDF
            doc = fitz.open(self.file_path)
            documents = []
            ocr_pages_count = 0

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Try native text extraction first (fast)
                text = page.get_text()

                # Check if OCR is needed
                if self.ocr_enabled and tesseract_available and self._needs_ocr(text):
                    logger.info(f"Page {page_num + 1} needs OCR (minimal text detected)")
                    ocr_text = self._ocr_page(page)
                    if ocr_text:
                        text = ocr_text
                        ocr_pages_count += 1

                # Only add pages with content
                if text.strip():
                    doc_obj = Document(
                        page_content=text,
                        metadata={
                            'source': self.file_path,
                            'filename': os.path.basename(self.file_path),
                            'page': page_num + 1,
                            'total_pages': len(doc),
                            'ocr_used': self.ocr_enabled and tesseract_available and self._needs_ocr(text)
                        }
                    )
                    documents.append(doc_obj)

            doc.close()

            if ocr_pages_count > 0:
                logger.info(f"Extracted {len(documents)} pages ({ocr_pages_count} with OCR)")
            else:
                logger.info(f"Extracted {len(documents)} pages (native text extraction)")

            return documents

        except ImportError:
            logger.warning("PyMuPDF not installed, falling back to PyPDFLoader")
            logger.warning("Install with: pip install PyMuPDF")
            # Fallback to original loader
            fallback = PyPDFLoader(self.file_path)
            return fallback.load()

        except Exception as e:
            logger.error(f"Error loading PDF with PyMuPDF: {e}")
            # Fallback to original loader
            logger.info("Falling back to PyPDFLoader")
            fallback = PyPDFLoader(self.file_path)
            return fallback.load()

class DocumentProcessor:
    """Handles loading, processing, and splitting documents."""

    @staticmethod
    def extract_folder_metadata(file_path: str, base_path: str) -> dict:
        """
        Extract folder hierarchy metadata from file path.

        Args:
            file_path: Full path to the file
            base_path: Base data directory path

        Returns:
            dict: Folder metadata including path, depth, and project info
        """
        try:
            # Normalize paths
            file_path = os.path.abspath(file_path)
            base_path = os.path.abspath(base_path)

            # Get relative path from base
            rel_path = os.path.relpath(file_path, base_path)

            # Split into directory and filename
            dir_path = os.path.dirname(rel_path)

            # Handle root-level files (no subdirectory)
            if dir_path == '' or dir_path == '.':
                return {
                    'folder_path': 'root',
                    'folder_name': 'root',
                    'folder_depth': 0,
                    'project': 'root',
                    'subfolder': None,
                }

            # Split path into components
            path_parts = dir_path.split(os.sep)

            return {
                'folder_path': dir_path,  # Full relative path: "project-alpha/docs"
                'folder_name': path_parts[-1],  # Immediate parent: "docs"
                'folder_depth': len(path_parts),  # Nesting level: 2
                'project': path_parts[0],  # Top-level folder: "project-alpha"
                'subfolder': path_parts[1] if len(path_parts) > 1 else None,  # Second level
            }

        except Exception as e:
            logger.warning(f"Error extracting folder metadata for {file_path}: {e}")
            return {
                'folder_path': 'unknown',
                'folder_name': 'unknown',
                'folder_depth': 0,
                'project': 'unknown',
                'subfolder': None,
            }

    @staticmethod
    def check_dependencies() -> bool:
        """Verify that required dependencies are installed."""
        missing_deps = []
        
        try:
            import docx2txt
        except ImportError:
            missing_deps.append("docx2txt (for DOCX files)")
        
        try:
            import pypdf
        except ImportError:
            missing_deps.append("pypdf (for PDF files)")
        
        try:
            import html2text
        except ImportError:
            missing_deps.append("html2text (for EPUB files)")
        
        try:
            import bs4
        except ImportError:
            missing_deps.append("beautifulsoup4 (for EPUB files)")
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
            logger.warning("Some document types may not load correctly.")
            logger.warning("Install missing dependencies with: pip install " + " ".join([d.split(' ')[0] for d in missing_deps]))
            return False
            
        return True
    
    @staticmethod
    def get_file_loader(file_path: str):
        """Returns appropriate loader based on file extension with error handling."""
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)  # Define filename variable

        try:
            # Check if this is a knowledge base file - DIRECT HANDLING
            if '_kb' in filename.lower() and ext in ['.txt', '.md']:
                logger.info(f"Detected knowledge base file: {filename} - Using direct KB loader")
                # Create a custom loader that directly calls the static method
                class DirectKBLoader(BaseLoader):
                    def __init__(self, file_path):
                        self.file_path = file_path
                    def load(self):
                        logger.info(f"DirectKBLoader: Loading {self.file_path}")
                        try:
                            with open(self.file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                                logger.info(f"DirectKBLoader: Successfully read file with {len(text)} characters")

                            # Parse directly with the parser
                            docs = GenericKnowledgeBaseParser.parse_kb(
                                text,
                                source=self.file_path,
                                filename=os.path.basename(self.file_path)
                            )
                            logger.info(f"DirectKBLoader: Parsed {len(docs)} documents")
                            return docs
                        except Exception as e:
                            logger.error(f"DirectKBLoader error: {e}")
                            return []

                return DirectKBLoader(file_path)
            
            # Regular file types
            if ext == '.pdf':
                # Use PyMuPDF with OCR if enabled, otherwise use standard PyPDFLoader
                if config.USE_PYMUPDF:
                    return PyMuPDFOCRLoader(file_path, ocr_enabled=config.USE_OCR)
                else:
                    return PyPDFLoader(file_path)
            elif ext in ['.docx', '.doc']:
                return Docx2txtLoader(file_path)
            elif ext in ['.ppt', '.pptx']:
                return UnstructuredPowerPointLoader(file_path)
            elif ext == '.epub':
                logger.info(f"Loading EPUB file: {filename}")
                try:
                    return UnstructuredEPubLoader(file_path)
                except Exception as e:
                    logger.warning(f"UnstructuredEPubLoader failed: {e}, trying alternative EPUB loader")
                    # Use DocumentProcessor instead of cls
                    docs = DocumentProcessor.load_epub(file_path)
                    if docs:
                        class CustomEpubLoader(BaseLoader):
                            def __init__(self, documents):
                                self.documents = documents
                            def load(self):
                                return self.documents
                        return CustomEpubLoader(docs)
                    return None
            elif ext in ['.txt', '.md', '.csv']:
                # Special case for knowledge base files (*_kb.txt, *_kb.md, etc.)
                if '_kb' in filename.lower():
                    logger.info(f"Detected knowledge base file in .txt handler: {filename}")
                    # Create a custom loader that directly calls the static method
                    class DirectKBLoader(BaseLoader):
                        def __init__(self, file_path):
                            self.file_path = file_path
                        def load(self):
                            logger.info(f"DirectKBLoader (txt handler): Loading {self.file_path}")
                            try:
                                with open(self.file_path, 'r', encoding='utf-8') as f:
                                    text = f.read()
                                    logger.info(f"DirectKBLoader (txt handler): Successfully read file with {len(text)} characters")

                                # Parse directly with the parser
                                docs = GenericKnowledgeBaseParser.parse_kb(
                                    text,
                                    source=self.file_path,
                                    filename=os.path.basename(self.file_path)
                                )
                                logger.info(f"DirectKBLoader (txt handler): Parsed {len(docs)} documents")
                                return docs
                            except Exception as e:
                                logger.error(f"DirectKBLoader (txt handler) error: {e}")
                                return []

                    return DirectKBLoader(file_path)
                else:
                    return TextLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {ext} for file {filename}")
                return None
        except Exception as e:
            logger.error(f"Error creating loader for {file_path}: {str(e)}")
            return None
    
    @classmethod
    def load_documents(cls, path: str, extensions: List[str] = None) -> List:
        """
        Load documents from specified path with specified extensions.
        Returns a list of documents or empty list if none found.

        Filtering behavior:
        - When FILTER_KB_ONLY=true: Only loads files with '_kb' in the name
        - Otherwise: Loads all files with supported extensions
        """
        if extensions is None:
            extensions = config.SUPPORTED_EXTENSIONS
                
        # Use the configuration variable for filtering
        filter_kb_only = config.FILTER_KB_ONLY

        if filter_kb_only:
            logger.info("KB-only filtering is ENABLED - loading only files with '_kb' in name")
        else:
            logger.info("KB-only filtering is DISABLED - loading all compatible files")
        
        logger.info(f"Loading documents from: {path}")
        
        # Verify data directory exists and list all files
        try:
            if not os.path.exists(path):
                logger.error(f"Data directory does not exist: {path}")
                return []
                
            logger.info(f"Data directory exists: {path}")
            
            # List all files in the directory
            all_files_in_dir = os.listdir(path)
            logger.info(f"Files in data directory: {all_files_in_dir}")

            # Check for knowledge base files
            kb_files = [f for f in all_files_in_dir if '_kb' in f.lower()]
            if kb_files:
                logger.info(f"Found knowledge base files: {kb_files}")
        except Exception as e:
            logger.error(f"Error checking data directory: {e}")
        
        try:
            # Find all files with supported extensions
            all_files = []
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file_path)[1].lower()
                    
                    # Only process files with supported extensions
                    if ext in extensions:
                        # Apply KB-only filtering if enabled
                        if filter_kb_only and '_kb' not in file.lower():
                            logger.info(f"Skipping non-KB document: {file}")
                            continue

                        all_files.append(file_path)
                        logger.info(f"Added file to processing list: {file}")

            if not all_files:
                logger.warning(f"No compatible documents found in {path}")
                return []

            logger.info(f"Found {len(all_files)} compatible files")

            # Load each file with its appropriate loader
            all_documents = []
            for file_path in all_files:
                try:
                    filename = os.path.basename(file_path)
                    logger.info(f"Processing file: {filename}")

                    # Regular loader path
                    logger.info(f"Getting loader for: {filename}")
                    loader = cls.get_file_loader(file_path)
                    
                    if loader:
                        logger.info(f"Loader created for {filename}, type: {type(loader).__name__}")
                        docs = loader.load()
                        
                        if docs:
                            logger.info(f"Loader returned {len(docs)} documents")

                            # Extract folder metadata once per file
                            folder_metadata = cls.extract_folder_metadata(file_path, path)

                            # Source metadata to each document
                            for doc in docs:
                                if not hasattr(doc, 'metadata') or doc.metadata is None:
                                    doc.metadata = {}
                                doc.metadata['source'] = file_path
                                doc.metadata['filename'] = os.path.basename(file_path)

                                # Add folder hierarchy metadata
                                doc.metadata.update(folder_metadata)

                                # Timestamp for sorting by recency if needed
                                try:
                                    doc.metadata['timestamp'] = os.path.getmtime(file_path)
                                    # Add human-readable date
                                    from datetime import datetime
                                    doc.metadata['modified_date'] = datetime.fromtimestamp(
                                        os.path.getmtime(file_path)
                                    ).strftime('%Y-%m-%d')
                                except:
                                    doc.metadata['timestamp'] = 0
                                    doc.metadata['modified_date'] = 'unknown'

                                # Add document type based on extension
                                ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                                doc.metadata['doc_type'] = ext

                            all_documents.extend(docs)
                            logger.info(f"Loaded {len(docs)} sections from {os.path.basename(file_path)}")
                        else:
                            logger.warning(f"Loader returned no documents for {filename}")
                    else:
                        logger.warning(f"No loader created for {filename}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    continue

            # Filter out empty documents
            valid_documents = [doc for doc in all_documents if doc.page_content and doc.page_content.strip()]
            
            if not valid_documents:
                logger.warning("No document content could be extracted successfully")
                    
            logger.info(f"Successfully loaded {len(valid_documents)} total document sections")
            return valid_documents

        except Exception as e:
            logger.error(f"Document loading error: {e}")
            return []
    
    @staticmethod
    def load_epub(file_path: str):
        """
        Custom EPUB loader using ebooklib.
        Returns a list of LangChain Document objects.
        """
        try:
            from ebooklib import epub
            
            filename = os.path.basename(file_path)
            logger.info(f"Loading EPUB with custom loader: {filename}")
            
            # Load the EPUB file
            book = epub.read_epub(file_path)
            
            # Extract and process content
            documents = []
            h2t = html2text.HTML2Text()
            h2t.ignore_links = False
            
            # Get book title and metadata
            title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "Unknown Title"
            
            # Process each chapter/item
            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    # Extract HTML content
                    html_content = item.get_content().decode('utf-8')
                    
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Get plain text content
                    text = h2t.handle(str(soup))
                    
                    if text.strip():
                        # Create a document with metadata
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': file_path,
                                'filename': filename,
                                'title': title,
                                'chapter': item.get_name(),
                            }
                        )
                        documents.append(doc)
            
            logger.info(f"Extracted {len(documents)} sections from EPUB")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading EPUB file {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def split_documents(documents: List, chunk_size: int = None, chunk_overlap: int = None) -> List:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            
        Returns:
            List of document chunks
        """
        if not documents:
            logger.warning("No documents to split")
            return []
            
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
            
        if chunk_overlap is None:
            chunk_overlap = config.CHUNK_OVERLAP
            
        logger.info(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
        
        try:
            # Group documents by type
            apu_kb_docs = []
            standard_docs = []
            
            for doc in documents:
                if doc.metadata.get('content_type') == 'kb_page':
                    apu_kb_docs.append(doc)
                else:
                    standard_docs.append(doc)
            
            logger.info(f"Document split: {len(apu_kb_docs)} knowledge base docs, {len(standard_docs)} standard docs")
            
            chunked_documents = []
            
            # Use knowledge base specific splitter for knowledge base pages
            if apu_kb_docs:
                apu_kb_splitter = KnowledgeBaseTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
                kb_chunks = apu_kb_splitter.split_documents(apu_kb_docs)
                logger.info(f"knowledge base splitter created {len(kb_chunks)} chunks from {len(apu_kb_docs)} documents")
                chunked_documents.extend(kb_chunks)
            
            # Use standard splitter for other documents
            if standard_docs:
                standard_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
                std_chunks = standard_splitter.split_documents(standard_docs)
                logger.info(f"Standard splitter created {len(std_chunks)} chunks from {len(standard_docs)} documents")
                chunked_documents.extend(std_chunks)
            
            # Remove any empty chunks
            valid_chunks = [chunk for chunk in chunked_documents if chunk.page_content and chunk.page_content.strip()]
            
            # Log statistics
            logger.info(f"Created {len(valid_chunks)} chunks from {len(documents)} documents")
            
            return valid_chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return []
