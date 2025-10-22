# RAGI - Retrieval-Augmented Generation Interface

<div align="center">

**A powerful, local-first RAG system for intelligent document question-answering**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#features) •
[Quick Start](#quick-start) •
[Installation](#installation) •
[Usage](#usage) •
[Architecture](#architecture)

</div>

---

## What is RAGI?

RAGI (Retrieval-Augmented Generation Intelligence) is a streamlined, general-purpose RAG system that helps you interact with your document collection using natural language. It combines semantic search with large language models to provide accurate, context-aware answers.

**Perfect for:**
- Personal knowledge bases
- Technical documentation
- Research paper collections
- Corporate document repositories
- Educational content libraries

## Features

### Core Capabilities
- **Hybrid Search** - Combines semantic and keyword search for better results
- **Multi-Format Support** - PDF, DOCX, TXT, MD, PPTX, EPUB
- **OCR Support** - Extract text from scanned PDFs and images
- **Local-First** - Runs entirely on your machine with Ollama
- **Fast Retrieval** - ChromaDB vector store with optimized indexing
- **Session Management** - Multiple conversation sessions with memory
- **Context-Aware** - Remembers conversation history
- **Folder Organization** - Organize documents in folders and filter searches by project/folder

### Technical Features
- **Semantic Search** - BGE embeddings for understanding context
- **Reranking** - Cross-encoder reranking for improved relevance
- **Intelligent OCR** - PyMuPDF + Tesseract with smart detection (only runs when needed)
- **Streaming Responses** - Real-time token-by-token generation
- **GPU Acceleration** - CUDA and Apple Silicon (MPS) support
- **Metadata Filtering** - Pre-filter by folder before vector search (10-100x faster)
- **Flexible Configuration** - Environment-based settings
- **REST API** - Flask API for integration

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/nikiwit/RAGI.git
cd RAGI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 4. Install Tesseract OCR (for scanned PDF support)
brew install tesseract  # macOS
# sudo apt-get install tesseract-ocr  # Linux
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# 5. Install and configure Ollama
# Visit https://ollama.com for installation
ollama pull qwen2.5:3b-instruct

# 6. Add your documents
# Option 1: Place documents directly in data/
cp /path/to/your/documents/*.pdf data/

# Option 2: Organize in folders (recommended)
mkdir -p data/project-alpha/technical-docs
cp /path/to/technical/*.pdf data/project-alpha/technical-docs/

# 7. Run RAGI
python main.py
```

That's it! RAGI will automatically index your documents and start the interactive CLI.

## Installation

### Prerequisites
- Python 3.10-3.12 (3.12 recommended for best compatibility)
- 8GB+ RAM recommended
- [Ollama](https://ollama.com) installed and running
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for scanned PDF support (optional but recommended)

**Note:** Python 3.13+ may have compatibility issues with some dependencies. We recommend Python 3.12 for the best experience.

### Step-by-Step

1. **Set up Python environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate

   # Or using conda (recommended)
   conda create -n RAGI python=3.12
   conda activate RAGI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install spaCy model**
   ```bash
   python -m spacy download en_core_web_md
   ```

4. **Install Tesseract OCR (for scanned PDF support)**
   ```bash
   # macOS
   brew install tesseract

   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr

   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki

   # Verify installation
   tesseract --version
   ```

5. **Configure Ollama**
   ```bash
   # Start Ollama service
   ollama serve

   # Pull the LLM model (in another terminal)
   ollama pull qwen2.5:3b-instruct  # ~2GB download
   ```

6. **Configure RAGI (optional)**
   ```bash
   cp .env.example .env
   # Edit .env with your preferences
   ```

## Usage 

### Command Line Interface

```bash
python main.py
```

**Available Commands:**
- `help` - Show available commands
- `stats` - Display document and session statistics
- `reindex` - Reindex all documents
- `clear` - Clear conversation memory
- `new session` - Create a new chat session
- `list sessions` - Show all sessions
- `switch session` - Change active session
- `list folders` - Show all available folders with document counts
- `filter folder <name>` - Search only in specified folder/project
- `clear filter` - Remove active folder filter
- `exit` / `quit` - Exit application

### REST API

```bash
python api.py
```

The API will be available at `http://localhost:5000`

**Endpoints:**
- `POST /query` - Submit a query
  ```json
  {
    "query": "What is machine learning?",
    "session_id": "optional-session-id"
  }
  ```

- `GET /stats` - Get system statistics
- `POST /reindex` - Trigger document reindexing

# Using folder commands
> list folders
Folder Structure:
  • project-alpha: 15 documents
  • project-beta: 8 documents

> filter folder technical-docs
Folder filter applied: 'technical-docs' (10 documents)

> What is the system architecture?
[Searches only in technical-docs folder - 10x faster!]

> clear filter
Folder filter cleared
```

**Benefits:**
- Organize documents by project, topic, or department
- 10-100x faster searches when filtering by folder
- More relevant results from specific document sets
- Track document counts per folder

See [FOLDER_FEATURE_GUIDE.md](FOLDER_FEATURE_GUIDE.md) for detailed usage and best practices.

### OCR for Scanned PDFs

RAGI includes intelligent OCR support for extracting text from scanned PDFs and images using PyMuPDF + Tesseract:

**Key Features:**
- **Automatic Detection** - Only runs OCR when needed (pages with minimal text)
- **Best Performance** - PyMuPDF achieves F1 score of 0.973 (best-in-class)
- **Fast Processing** - ~1000x faster for text-based PDFs (smart detection)
- **Layout Preservation** - Maintains document structure for better LLM understanding
- **Multi-language** - Supports 100+ languages via Tesseract

**Supported Documents:**
- Scanned PDFs (fully image-based)
- Mixed PDFs (combination of text and scanned pages)
- PDFs with embedded images containing text
- **Note:** For DOCX/PPTX with images, convert to PDF first for best results

**Configuration:**
```bash
# In .env file
RAGI_USE_OCR=true                    # Enable OCR
RAGI_USE_PYMUPDF=true                # Use PyMuPDF for PDF processing
RAGI_OCR_LANGUAGE=eng                # Language (eng, spa, fra, deu, chi_sim, etc.)
RAGI_OCR_DPI=300                     # Resolution (150-600, higher = better quality)
RAGI_OCR_MIN_TEXT_THRESHOLD=50       # Trigger OCR if page has < 50 characters
```

**Troubleshooting:**
```bash
# Verify Tesseract is installed
tesseract --version

# Test OCR functionality
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"

# Install additional language packs (optional)
brew install tesseract-lang  # macOS
```

**Performance:**
- Text-based PDFs: 0.5-1 second per page (native extraction)
- Scanned PDFs: 2-5 seconds per page (OCR processing)
- Memory: ~100-200MB per page during OCR

## Architecture

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────────┐
│          CLI / REST API                 │
└─────────────────┬───────────────────────┘
                  │
       ┌──────────┴──────────┐
       │                     │
       v                     v
┌─────────────┐      ┌─────────────┐
│   Query     │      │  Document   │
│   Router    │      │  Processor  │
└──────┬──────┘      └──────┬──────┘
       │                    │
       v                    v
┌─────────────┐      ┌─────────────┐
│  Retrieval  │◄────►│   Vector    │
│  Handler    │      │   Store     │
└──────┬──────┘      │  (ChromaDB) │
       │             └─────────────┘
       v
┌─────────────┐
│  Response   │
│  Generator  │◄─── Ollama LLM
└──────┬──────┘
       │
       v
┌─────────────┐
│  Streaming  │
│  Response   │
└─────────────┘
```

### Key Components

- **Document Processing** - Multi-format parsing and chunking
- **Vector Store** - ChromaDB for semantic search
- **Query Router** - Intelligent query classification
- **Retrieval Handler** - Hybrid search with reranking
- **Response Generator** - LLM-powered answer generation
- **Session Manager** - Multi-session conversation memory

## Configuration

RAGI can be configured via environment variables or `.env` file:

```bash
# Paths
RAGI_DATA_PATH=./data
RAGI_VECTOR_PATH=./vector_store

# Models
RAGI_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RAGI_LLM_MODEL=qwen2.5:3b-instruct
RAGI_RERANKER_MODEL=BAAI/bge-reranker-base

# Search
RAGI_SEARCH_TYPE=hybrid  # semantic, keyword, or hybrid
RAGI_RETRIEVER_K=6
RAGI_KEYWORD_RATIO=0.4

# Features
RAGI_USE_FAQ_MATCHING=true
RAGI_CONTEXT_COMPRESSION=true
RAGI_QUERY_EXPANSION=true

# OCR (for scanned PDFs)
RAGI_USE_OCR=true
RAGI_USE_PYMUPDF=true
RAGI_OCR_LANGUAGE=eng
RAGI_OCR_DPI=300

# Performance
RAGI_MAX_THREADS=4
RAGI_MAX_MEMORY=4G
```

See `.env.example` for all available options.