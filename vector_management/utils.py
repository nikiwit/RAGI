"""
Utility functions and context managers for vector store management.

This module provides utility functions including proper context managers, 
async operations, and pathlib usage.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import torch
from langchain_core.documents import Document

from .models import ModelCacheException

logger = logging.getLogger("RAGI")


@contextmanager
def safe_file_operation(
    filepath: Path,
    mode: str = 'r',
    encoding: str = 'utf-8'
) -> Generator[Any, None, None]:
    """
    Context manager for safe file operations with proper error handling.

    Args:
        filepath: Path to the file
        mode: File open mode
        encoding: File encoding

    Yields:
        File handle

    Raises:
        ModelCacheException: If file operations fail
    """
    try:
        with open(filepath, mode, encoding=encoding) as file:
            yield file
    except (FileNotFoundError, PermissionError) as e:
        raise ModelCacheException(f"File operation failed for {filepath}: {e}") from e
    except (OSError, IOError) as e:
        raise ModelCacheException(f"IO error for {filepath}: {e}") from e


@contextmanager
def cleanup_on_error(cleanup_func: callable, *args) -> Generator[None, None, None]:
    """
    Context manager that runs cleanup function if an error occurs.
    
    Args:
        cleanup_func: Function to call for cleanup
        *args: Arguments to pass to cleanup function
    """
    try:
        yield
    except Exception:
        try:
            cleanup_func(*args)
        except Exception as cleanup_error:
            logger.warning(f"Cleanup failed: {cleanup_error}")
        raise


def ensure_directory(path: Path, permissions: int = 0o755) -> Path:
    """
    Ensure directory exists with proper permissions.
    
    Args:
        path: Directory path
        permissions: Directory permissions (Unix only)
        
    Returns:
        Path to the created directory
        
    Raises:
        ModelCacheException: If directory creation fails
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        
        # Set permissions on Unix-like systems
        if sys.platform != 'win32':
            path.chmod(permissions)
            
        return path
    except (PermissionError, OSError) as e:
        raise ModelCacheException(f"Failed to create directory {path}: {e}") from e


def safe_remove_directory(path: Path, max_attempts: int = 3) -> bool:
    """
    Safely remove directory with multiple attempts and permission fixing.
    
    Args:
        path: Directory to remove
        max_attempts: Maximum number of removal attempts
        
    Returns:
        True if successfully removed, False otherwise
    """
    if not path.exists():
        return True
    
    for attempt in range(max_attempts):
        try:
            # Fix permissions before removal
            _fix_directory_permissions(path)
            
            # Attempt removal
            shutil.rmtree(path)
            logger.info(f"Successfully removed directory: {path}")
            return True
            
        except (PermissionError, OSError) as e:
            if attempt == max_attempts - 1:
                logger.error(f"Failed to remove directory {path} after {max_attempts} attempts: {e}")
                return False
            
            logger.warning(f"Attempt {attempt + 1} failed to remove {path}: {e}, retrying...")
            time.sleep(1.0)
    
    return False


def _fix_directory_permissions(path: Path) -> None:
    """
    Fix permissions recursively for directory removal.
    
    Args:
        path: Directory to fix permissions for
    """
    try:
        if sys.platform == 'win32':
            # Windows permission fix
            for item in path.rglob('*'):
                try:
                    if item.is_file():
                        item.chmod(0o777)
                    elif item.is_dir():
                        item.chmod(0o777)
                except (PermissionError, OSError):
                    continue
        else:
            # Unix-like permission fix
            for item in path.rglob('*'):
                try:
                    if item.is_file():
                        item.chmod(0o644)
                    elif item.is_dir():
                        item.chmod(0o755)
                except (PermissionError, OSError):
                    continue
    except Exception as e:
        logger.warning(f"Error fixing permissions for {path}: {e}")


def get_optimal_device() -> str:
    """
    Determine the best available device for embeddings.
    
    Returns:
        Device identifier ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU for embeddings")
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Using Apple Silicon MPS for embeddings")
        return 'mps'
    else:
        logger.info("Using CPU for embeddings")
        return 'cpu'


def sanitize_metadata_for_chroma(documents: list[Document]) -> list[Document]:
    """
    Sanitize document metadata to ensure ChromaDB compatibility.
    
    ChromaDB requires metadata to be simple key-value pairs with
    string, int, float, or bool values.
    
    Args:
        documents: List of documents to process
        
    Returns:
        Documents with sanitized metadata
    """
    sanitized_docs = []
    
    for doc in documents:
        metadata = doc.metadata.copy() if doc.metadata else {}
        
        # Process each metadata field
        sanitized_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized_metadata[key] = value
            elif isinstance(value, list) and value:
                # Convert non-empty lists to JSON strings
                sanitized_metadata[key] = json.dumps(value)
            elif value is not None:
                # Convert other non-None values to strings
                sanitized_metadata[key] = str(value)
            # Skip None values and empty lists
        
        sanitized_doc = Document(
            page_content=doc.page_content,
            metadata=sanitized_metadata
        )
        sanitized_docs.append(sanitized_doc)
    
    return sanitized_docs


def validate_model_path(path: Path) -> bool:
    """
    Validate that a model path contains essential files.
    
    Args:
        path: Path to the model directory
        
    Returns:
        True if model appears valid
    """
    if not path.exists() or not path.is_dir():
        return False
    
    essential_files = ['config.json']
    model_files = ['pytorch_model.bin', 'model.safetensors', 'model.onnx']
    
    # Check for essential configuration
    has_config = any((path / f).exists() for f in essential_files)
    if not has_config:
        # Check in snapshots directory (HF cache structure)
        snapshots_dir = path / 'snapshots'
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    if any((snapshot / f).exists() for f in essential_files):
                        has_config = True
                        break
    
    # Check for model weights
    has_model = any((path / f).exists() for f in model_files)
    if not has_model:
        # Check in snapshots directory
        snapshots_dir = path / 'snapshots'
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    if any((snapshot / f).exists() for f in model_files):
                        has_model = True
                        break
    
    return has_config and has_model


def find_model_cache_paths(model_name: str, cache_base: Path) -> list[Path]:
    """
    Find all possible cache paths for a model.
    
    Args:
        model_name: Name of the model
        cache_base: Base cache directory
        
    Returns:
        List of possible cache paths
    """
    model_variants = [
        f"models--{model_name.replace('/', '--')}",  # HF format
        model_name.replace('/', '_'),  # Old format
        f"sentence-transformers_{model_name.replace('/', '_')}",  # ST format
    ]
    
    possible_paths = []
    cache_dirs = [
        cache_base / "hub",
        cache_base / "sentence_transformers", 
        cache_base
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            for variant in model_variants:
                model_path = cache_dir / variant
                if model_path.exists():
                    possible_paths.append(model_path)
    
    return possible_paths


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, logger: logging.Logger = logger):
        self.operation_name = operation_name
        self.logger = logger
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.debug(f"Completed {self.operation_name} in {duration:.2f}s")
        
        if exc_type:
            self.logger.error(f"Error during {self.operation_name}: {exc_val}")
        
        return False  # Don't suppress exceptions