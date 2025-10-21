"""
Configuration settings for the RAGI system with environment-specific support and model management.

This module provides comprehensive configuration management including:
- Environment-specific configuration loading
- Hardware detection and optimization
- Model lifecycle management settings
- Logging and error handling setup
- Path and resource management
"""

import os
import logging
import platform
from datetime import datetime
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

# Disable ChromaDB telemetry globally before any ChromaDB imports
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
os.environ.setdefault('CHROMA_TELEMETRY_DISABLED', '1')

# Suppress ChromaDB telemetry error logs
def suppress_chromadb_telemetry_errors():
    """Suppress annoying ChromaDB telemetry error logs."""
    import logging
    chromadb_telemetry_logger = logging.getLogger("chromadb.telemetry.product.posthog")
    chromadb_telemetry_logger.setLevel(logging.CRITICAL)

# Apply telemetry log suppression immediately
suppress_chromadb_telemetry_errors()

# Always use local environment
ENV = "local"
env_file = ".env"

# Load environment variables from .env file
if os.path.exists(env_file):
    load_dotenv(env_file, override=True)
else:
    load_dotenv(override=True)  # Fallback if no .env exists

# Configure logging with rotation
log_level_name = os.environ.get("RAGI_LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_name.upper(), logging.INFO)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure log rotation settings
LOG_MAX_BYTES = int(os.environ.get("RAGI_LOG_MAX_BYTES", "1073741824"))  # 1GB default
LOG_BACKUP_COUNT = int(os.environ.get("RAGI_LOG_BACKUP_COUNT", "5"))  # Keep 5 backup files
LOG_USE_JSON = os.environ.get("RAGI_LOG_USE_JSON", "False").lower() in ("true", "1", "t")

# Create rotating file handler with compression
rotating_handler = RotatingFileHandler(
    "logs/ragi.log",
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding='utf-8'
)

# Configure JSON formatter for production if enabled
if LOG_USE_JSON:
    import json
    import datetime
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            return json.dumps(log_entry)
    
    rotating_handler.setFormatter(JSONFormatter())
    console_formatter = JSONFormatter()
else:
    # Standard text format for development
    standard_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rotating_handler.setFormatter(standard_formatter)
    console_formatter = standard_formatter

# Configure console handler with appropriate formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)

logging.basicConfig(
    level=log_level,
    handlers=[
        rotating_handler,
        console_handler
    ]
)
logger = logging.getLogger("RAGI")

def setup_nltk_with_fallback():
    """
    Setup NLTK with proper error handling and fallback options.
    
    Attempts to download required NLTK data with graceful fallback
    to basic text processing methods if NLTK is unavailable.
    """
    try:
        import nltk
        # Try to download required NLTK data with error handling
        try:
            nltk.data.find('corpora/wordnet')
            logger.info("NLTK WordNet data already available")
        except LookupError:
            try:
                logger.info("Attempting to download NLTK WordNet data...")
                nltk.download('wordnet', quiet=True, raise_on_error=True)
                logger.info("Successfully downloaded NLTK WordNet data")
            except Exception as e:
                logger.warning(f"Failed to download NLTK WordNet data: {e}")
                logger.info("NLTK will use fallback methods for text processing")
                
        # Try to download other useful NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True, raise_on_error=True)
                logger.debug("Downloaded NLTK punkt tokenizer")
            except Exception as e:
                logger.debug(f"Could not download NLTK punkt tokenizer: {e}")
                
    except ImportError:
        logger.info("NLTK not available - using basic text processing methods")
    except Exception as e:
        logger.warning(f"NLTK setup encountered an error: {e}")
        logger.info("Continuing with fallback text processing methods")

class Config:
    """
    Base configuration settings for the RAGI application with model management.

    This class provides centralized configuration management with support for
    environment-specific overrides, hardware detection, and production-grade
    model lifecycle management for RAGI.
    """
    
    # Class variable to track setup status
    _setup_completed = False
    
    # Environment
    ENV = ENV
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.environ.get("RAGI_DATA_PATH", os.path.join(SCRIPT_DIR, "data"))
    PERSIST_PATH = os.environ.get("RAGI_VECTOR_PATH", os.path.join(SCRIPT_DIR, "vector_store"))
    
    # Embedding and retrieval settings
    EMBEDDING_MODEL_NAME = os.environ.get("RAGI_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    LLM_MODEL_NAME = os.environ.get("RAGI_LLM_MODEL", "qwen2.5:3b-instruct")
    
    # Reranker model settings
    RERANKER_MODEL_NAME = os.environ.get("RAGI_RERANKER_MODEL", "BAAI/bge-reranker-base")
    
    # Chunking settings
    CHUNK_SIZE = int(os.environ.get("RAGI_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.environ.get("RAGI_CHUNK_OVERLAP", "150"))
    
    # Response streaming speed
    STREAM_DELAY = 0.015  # Consistent streaming delay across all responses
    
    # Retrieval settings
    RETRIEVER_K = int(os.environ.get("RAGI_RETRIEVER_K", "6"))
    RETRIEVER_SEARCH_TYPE = os.environ.get("RAGI_SEARCH_TYPE", "hybrid")
    KEYWORD_RATIO = float(os.environ.get("RAGI_KEYWORD_RATIO", "0.4"))
    
    # Query processing settings
    USE_QUERY_EXPANSION = os.environ.get("RAGI_QUERY_EXPANSION", "True").lower() in ("true", "1", "t")
    EXPANSION_FACTOR = int(os.environ.get("RAGI_EXPANSION_FACTOR", "3"))
    
    # Semantic enhancement settings (Phase 4)
    USE_ENHANCED_SEMANTICS = os.environ.get("RAGI_USE_ENHANCED_SEMANTICS", "true").lower() == "true"
    SEMANTIC_MODEL = os.environ.get("RAGI_SEMANTIC_MODEL", "en_core_web_md")
    SEMANTIC_CACHE_SIZE = int(os.environ.get("RAGI_SEMANTIC_CACHE_SIZE", "1000"))
    SEMANTIC_EXPANSION_LIMIT = int(os.environ.get("RAGI_SEMANTIC_EXPANSION_LIMIT", "5"))
    SEMANTIC_ERROR_THRESHOLD = int(os.environ.get("RAGI_SEMANTIC_ERROR_THRESHOLD", "5"))

    # Model management settings (for backward compatibility)
    MODEL_CHECK_INTERVAL_DAYS = int(os.environ.get("RAGI_MODEL_CHECK_INTERVAL_DAYS", "7"))
    MODEL_WARNING_AGE_DAYS = int(os.environ.get("RAGI_MODEL_WARNING_AGE_DAYS", "30"))
    MODEL_CRITICAL_AGE_DAYS = int(os.environ.get("RAGI_MODEL_CRITICAL_AGE_DAYS", "90"))
    MODEL_AUTO_UPDATE_PROMPT = os.environ.get("RAGI_MODEL_AUTO_UPDATE_PROMPT", "false").lower() == "true"
    MODEL_UPDATE_CHECK_ENABLED = os.environ.get("RAGI_MODEL_UPDATE_CHECK_ENABLED", "false").lower() == "true"
    MODEL_REQUIRE_APPROVAL = os.environ.get("RAGI_MODEL_REQUIRE_APPROVAL", "true").lower() == "true"
    MODEL_CACHE_CLEANUP = os.environ.get("RAGI_MODEL_CACHE_CLEANUP", "false").lower() == "true"
    MODEL_BACKUP_ENABLED = os.environ.get("RAGI_MODEL_BACKUP_ENABLED", "false").lower() == "true"
    MODEL_MAX_BACKUPS = int(os.environ.get("RAGI_MODEL_MAX_BACKUPS", "3"))
    MODEL_NOTIFICATION_EMAIL = os.environ.get("RAGI_MODEL_NOTIFICATION_EMAIL", "")

    # Document loading settings
    FILTER_KB_ONLY = os.environ.get("RAGI_FILTER_KB_ONLY", "false").lower() == "true"

    # OCR settings (PyMuPDF + Tesseract)
    USE_OCR = os.environ.get("RAGI_USE_OCR", "true").lower() == "true"
    OCR_LANGUAGE = os.environ.get("RAGI_OCR_LANGUAGE", "eng")
    OCR_DPI = int(os.environ.get("RAGI_OCR_DPI", "300"))
    OCR_MIN_TEXT_THRESHOLD = int(os.environ.get("RAGI_OCR_MIN_TEXT_THRESHOLD", "50"))
    PDF_PRESERVE_LAYOUT = os.environ.get("RAGI_PDF_PRESERVE_LAYOUT", "true").lower() == "true"
    USE_PYMUPDF = os.environ.get("RAGI_USE_PYMUPDF", "true").lower() == "true"

    # Context processing settings
    MAX_CONTEXT_SIZE = int(os.environ.get("RAGI_MAX_CONTEXT_SIZE", "4000"))
    USE_CONTEXT_COMPRESSION = os.environ.get("RAGI_CONTEXT_COMPRESSION", "True").lower() in ("true", "1", "t")
    
    # Confidence and boundary detection settings
    # Lowered from 0.4 to 0.15 for better recall with diverse documents
    # Lower threshold helps prevent false negatives when documents are relevant but not perfect matches
    CONFIDENCE_THRESHOLD = float(os.environ.get("RAGI_CONFIDENCE_THRESHOLD", "0.15"))
    
    # Contact Information
    SUPPORT_PHONE = os.environ.get("RAGI_SUPPORT_PHONE", "+1-xxx-xxx-xxxx")
    SUPPORT_EMAIL = os.environ.get("RAGI_SUPPORT_EMAIL", "support@example.com")
    SUPPORT_LOCATION = os.environ.get("RAGI_SUPPORT_LOCATION", "Support Center")
    
    
    # Session management
    MAX_SESSIONS = 5
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("RAGI_OLLAMA_URL", "http://localhost:11434")
    
    # Resource settings
    MAX_THREADS = int(os.environ.get("RAGI_MAX_THREADS", "4"))
    MAX_MEMORY = os.environ.get("RAGI_MAX_MEMORY", "4G")

    @classmethod
    def has_gpu(cls):
        """
        Detect if GPU is available (CUDA or Apple Silicon MPS).
        
        Returns:
            bool: True if GPU acceleration is available
        """
        try:
            import torch
            logger.debug(f"PyTorch version: {torch.__version__}")
            
            # Check for CUDA first (for compatibility)
            if torch.cuda.is_available():
                logger.debug(f"CUDA available: True, Device count: {torch.cuda.device_count()}")
                return True
            else:
                logger.debug("CUDA not available - checking reasons...")
                # Provide detailed CUDA diagnostics
                if hasattr(torch.cuda, 'is_available'):
                    logger.debug(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
                if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
                    cuda_version = torch.version.cuda
                    logger.debug(f"PyTorch compiled with CUDA: {cuda_version if cuda_version else 'No'}")
                
                # Check if NVIDIA GPU is present but PyTorch doesn't have CUDA support
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and result.stdout.strip():
                        gpu_names = result.stdout.strip().split('\n')
                        logger.warning(f"NVIDIA GPU(s) detected: {gpu_names}")
                        logger.warning("PyTorch with CUDA support may not be installed. Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                    logger.debug(f"Could not check for NVIDIA GPUs: {e}")
            
            # Check for Apple Silicon MPS
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.debug("Apple Silicon MPS available")
                return True
            else:
                if platform.system() == "Darwin":
                    logger.debug("Running on macOS but MPS not available")
                
            return False
            
        except ImportError as e:
            logger.warning(f"PyTorch not available: {e}")
            logger.warning("Install PyTorch with: pip install torch torchvision torchaudio")
            return False
        except Exception as e:
            logger.error(f"Error during GPU detection: {e}")
            return False
    
    @classmethod 
    def get_device_info(cls):
        """
        Get detailed device information for logging.
        
        Returns:
            tuple: (device_type, device_name) for hardware identification
        """
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    return "cuda", f"{device_name} (+{device_count-1} more)"
                return "cuda", device_name
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps", "Apple Silicon MPS"
            else:
                # Try to get CPU info for better diagnostics
                cpu_info = "CPU"
                try:
                    if platform.system() == "Darwin":
                        import subprocess
                        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0:
                            cpu_info = f"CPU ({result.stdout.strip()})"
                    elif platform.system() == "Windows":
                        import subprocess
                        result = subprocess.run(['wmic', 'cpu', 'get', 'name', '/format:value'], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0:
                            for line in result.stdout.split('\n'):
                                if line.startswith('Name='):
                                    cpu_info = f"CPU ({line.split('=', 1)[1].strip()})"
                                    break
                except Exception:
                    pass  # Keep default CPU info
                return "cpu", cpu_info
        except ImportError:
            return "cpu", "CPU (PyTorch not available)"
        except Exception as e:
            logger.debug(f"Error getting device info: {e}")
            return "cpu", "CPU"
    
    # Logging settings
    LOG_LEVEL = log_level_name
    LOG_MAX_BYTES = LOG_MAX_BYTES
    LOG_BACKUP_COUNT = LOG_BACKUP_COUNT
    LOG_USE_JSON = LOG_USE_JSON
    
    # Miscellaneous
    FORCE_REINDEX = os.environ.get("RAGI_FORCE_REINDEX", "False").lower() in ("true", "1", "t")

    # Supported file types
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc', '.md', '.ppt', '.pptx', '.epub']

    # KB specific settings
    KB_ANSWER_CONTEXT_SIZE = int(os.environ.get("RAGI_KB_ANSWER_SIZE", "3"))
    KB_EXACT_MATCH_BOOST = float(os.environ.get("RAGI_KB_EXACT_MATCH_BOOST", "2.0"))

    # Optional content type filtering
    KB_CONTENT_TYPES = os.environ.get("RAGI_KB_CONTENT_TYPES", None)
    if KB_CONTENT_TYPES:
        KB_CONTENT_TYPES = [t.strip() for t in KB_CONTENT_TYPES.split(',')]
    
    @classmethod
    def setup(cls):
        """
        Set up the configuration and ensure directories exist.
        
        Performs one-time initialization including directory creation,
        NLTK setup, hardware detection, and configuration logging.
        """
        # Prevent duplicate setup logging
        if cls._setup_completed:
            logger.debug("Configuration setup already completed, skipping duplicate setup")
            return
            
        # Ensure data directory exists
        os.makedirs(cls.DATA_PATH, exist_ok=True)
        
        # Setup NLTK with proper error handling
        setup_nltk_with_fallback()
        
        # Log environment and configuration
        logger.info(f"Running in {cls.ENV} environment")
        logger.info(f"Data directory: {cls.DATA_PATH}")
        logger.info(f"Vector store directory: {cls.PERSIST_PATH}")
        log_size_gb = cls.LOG_MAX_BYTES // 1073741824
        total_space_gb = log_size_gb * (cls.LOG_BACKUP_COUNT + 1)
        logger.info(f"Log rotation: {log_size_gb}GB max size, {cls.LOG_BACKUP_COUNT} backups (~{total_space_gb}GB total)")
        if cls.LOG_USE_JSON:
            logger.info("JSON logging enabled for structured log analysis")
        logger.info(f"Embedding model: {cls.EMBEDDING_MODEL_NAME}")
        logger.info(f"LLM model: {cls.LLM_MODEL_NAME}")
        logger.info(f"Search type: {cls.RETRIEVER_SEARCH_TYPE}")
        logger.info(f"GPU available: {cls.has_gpu()}")
        
        if cls.RETRIEVER_SEARCH_TYPE == "hybrid":
            logger.info(f"Keyword ratio: {cls.KEYWORD_RATIO}")
        if cls.USE_QUERY_EXPANSION:
            logger.info(f"Query expansion enabled with factor: {cls.EXPANSION_FACTOR}")

        if cls.USE_CONTEXT_COMPRESSION:
            logger.info(f"Context compression enabled")

        # Log OCR status
        if cls.USE_OCR:
            logger.info(f"OCR is ENABLED (Language: {cls.OCR_LANGUAGE}, DPI: {cls.OCR_DPI})")
            if cls.USE_PYMUPDF:
                logger.info("Using PyMuPDF for PDF processing with intelligent OCR fallback")
        else:
            logger.info("OCR is DISABLED")

        # Mark setup as completed
        cls._setup_completed = True
        logger.debug("Configuration setup completed successfully")

# Use single unified configuration
ConfigClass = Config
config = ConfigClass