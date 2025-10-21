"""
Service classes for vector store management.

Simplified version focused on core embedding functionality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Protocol

from langchain_huggingface import HuggingFaceEmbeddings

from .models import (
    EmbeddingConfig,
    VectorStoreStats,
)
from .utils import (
    PerformanceTimer,
    ensure_directory,
    find_model_cache_paths,
    get_optimal_device,
    safe_remove_directory,
    validate_model_path,
)

logger = logging.getLogger("RAGI")


class ConfigProvider(Protocol):
    """Protocol for configuration providers."""

    @property
    def embedding_model_name(self) -> str: ...

    @property
    def persist_path(self) -> str: ...

    @property
    def chunk_size(self) -> int: ...

    @property
    def chunk_overlap(self) -> int: ...

    @property
    def env(self) -> str: ...


class ModelCacheService:
    """Service for managing model caching operations."""

    def __init__(self, cache_base_path: Path):
        self.cache_base = ensure_directory(cache_base_path)

    def setup_cache_environment(self) -> None:
        """Setup HuggingFace cache environment variables."""
        import os

        # Set HuggingFace environment variables
        os.environ['HF_HOME'] = str(self.cache_base)
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(self.cache_base / "sentence_transformers")

        # Performance optimizations
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

        # Remove deprecated variables
        if 'TRANSFORMERS_CACHE' in os.environ:
            del os.environ['TRANSFORMERS_CACHE']

        logger.info(f"Model cache directory configured: {self.cache_base}")

    def cleanup_corrupted_files(self) -> int:
        """
        Clean up corrupted model cache files.

        Returns:
            Number of files cleaned up
        """
        cleanup_count = 0

        # Clean incomplete downloads
        for incomplete_file in self.cache_base.rglob('*.incomplete'):
            try:
                incomplete_file.unlink()
                cleanup_count += 1
                logger.debug(f"Removed incomplete file: {incomplete_file}")
            except OSError as e:
                logger.warning(f"Could not remove incomplete file {incomplete_file}: {e}")

        # Clean lock files
        for lock_file in self.cache_base.rglob('*.lock'):
            try:
                lock_file.unlink()
                cleanup_count += 1
                logger.debug(f"Removed lock file: {lock_file}")
            except OSError as e:
                logger.warning(f"Could not remove lock file {lock_file}: {e}")

        # Clean temporary files
        for tmp_file in self.cache_base.rglob('*.tmp'):
            try:
                tmp_file.unlink()
                cleanup_count += 1
                logger.debug(f"Removed temporary file: {tmp_file}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {tmp_file}: {e}")

        if cleanup_count > 0:
            logger.info(f"Model cache cleanup completed: removed {cleanup_count} corrupted files")

        return cleanup_count

    def is_model_cached(self, model_name: str) -> bool:
        """
        Check if model is cached and valid.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is properly cached
        """
        cache_paths = find_model_cache_paths(model_name, self.cache_base)

        for path in cache_paths:
            if validate_model_path(path):
                logger.info(f"Model {model_name} found in cache at {path}")
                return True

        logger.debug(f"Model {model_name} not found in cache")
        return False

    def clear_model_cache(self, model_name: str) -> bool:
        """
        Clear cached model files.

        Args:
            model_name: Name of the model to clear

        Returns:
            True if successfully cleared
        """
        cache_paths = find_model_cache_paths(model_name, self.cache_base)
        removed_any = False

        for path in cache_paths:
            if safe_remove_directory(path):
                logger.info(f"Removed cached model at: {path}")
                removed_any = True

        if removed_any:
            logger.info(f"Successfully cleared cache for model: {model_name}")
        else:
            logger.warning(f"No cached files found for model: {model_name}")

        return removed_any


class EmbeddingModelService:
    """Service for creating and managing embedding models."""

    def __init__(self, cache_service: ModelCacheService, config_provider: ConfigProvider):
        self.cache_service = cache_service
        self.config_provider = config_provider
        self._cached_embeddings: Optional[HuggingFaceEmbeddings] = None
        self._cached_model_name: Optional[str] = None

    def create_embeddings(self, model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
        """
        Create embedding model with configuration.

        Args:
            model_name: Name of the model (defaults to config value)

        Returns:
            Configured embedding model
        """
        if model_name is None:
            model_name = self.config_provider.embedding_model_name

        # Return cached embeddings if available
        if (self._cached_embeddings is not None and
            self._cached_model_name == model_name):
            logger.info(f"Using cached embeddings for model: {model_name}")
            return self._cached_embeddings

        # Setup cache environment
        self.cache_service.setup_cache_environment()

        # Get optimal device
        device = get_optimal_device()

        # Create embedding configuration
        config = EmbeddingConfig.for_model(model_name, device)

        # Create embeddings
        with PerformanceTimer(f"Creating embeddings for {model_name}"):
            embeddings = HuggingFaceEmbeddings(
                model_name=config.model_name,
                model_kwargs={
                    'device': config.device,
                    'trust_remote_code': config.trust_remote_code,
                },
                encode_kwargs={
                    'normalize_embeddings': config.normalize_embeddings,
                    'batch_size': config.batch_size,
                    'use_fp16': config.use_fp16,
                }
            )

        # Cache for future use
        self._cached_embeddings = embeddings
        self._cached_model_name = model_name

        logger.info(f"Successfully created embeddings for model: {model_name}")
        return embeddings


class VectorStoreHealthService:
    """Service for vector store health checking and diagnostics."""

    @staticmethod
    def check_health(vector_store: Optional[Any]) -> bool:
        """
        Perform basic health check on vector store.

        Args:
            vector_store: Vector store to check

        Returns:
            True if healthy, False otherwise
        """
        if not vector_store:
            logger.warning("No vector store provided for health check")
            return False

        try:
            # Check if we can access the collection
            if hasattr(vector_store, '_collection') and vector_store._collection:
                collection = vector_store._collection
                count = collection.count()
                logger.info(f"Vector store health check passed: {count} documents")
                return count > 0
            else:
                logger.warning("Vector store collection not accessible")
                return False
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False

    @staticmethod
    def get_statistics(vector_store: Optional[Any]) -> Optional[VectorStoreStats]:
        """
        Get basic statistics about vector store contents.

        Args:
            vector_store: Vector store to analyze

        Returns:
            Statistics or None if failed
        """
        if not vector_store:
            return None

        try:
            # Get collection info
            if hasattr(vector_store, '_collection') and vector_store._collection:
                collection = vector_store._collection
                count = collection.count()
                collection_name = collection.name if hasattr(collection, 'name') else "unknown"

                return VectorStoreStats(
                    total_documents=count,
                    collection_name=collection_name,
                    embedding_dimension=None
                )
            else:
                logger.warning("Cannot get statistics: collection not accessible")
                return None

        except Exception as e:
            logger.error(f"Error getting vector store statistics: {e}")
            return None


class VectorStoreManagerModern:
    """
    Simplified vector store manager focused on core embedding functionality.
    """

    def __init__(self, config_provider: ConfigProvider, cache_base_path: Optional[Path] = None):
        self.config_provider = config_provider

        # Setup cache path
        if cache_base_path is None:
            cache_base_path = Path(__file__).parent.parent / "model_cache" / "huggingface"

        # Initialize services
        self.cache_service = ModelCacheService(cache_base_path)
        self.embedding_service = EmbeddingModelService(self.cache_service, config_provider)
        self.health_service = VectorStoreHealthService()

    # Compatibility methods for existing code
    def setup_model_cache(self) -> Path:
        """Setup model cache and return path."""
        self.cache_service.setup_cache_environment()
        return self.cache_service.cache_base

    def is_model_cached(self, model_name: str) -> bool:
        """Check if model is cached."""
        return self.cache_service.is_model_cached(model_name)

    def create_embeddings(self, model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
        """Create embedding model."""
        return self.embedding_service.create_embeddings(model_name)

    def check_vector_store_health(self, vector_store: Any) -> bool:
        """Check vector store health."""
        return self.health_service.check_health(vector_store)

    def print_document_statistics(self, vector_store: Any) -> None:
        """Print document statistics."""
        stats = self.health_service.get_statistics(vector_store)
        if stats:
            print(f"Total documents: {stats.total_documents}")
            print(f"Collection: {stats.collection_name}")
        else:
            print("Error retrieving document statistics.")

    def cleanup_corrupted_cache_files(self) -> int:
        """Clean up corrupted cache files."""
        return self.cache_service.cleanup_corrupted_files()
