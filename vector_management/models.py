"""
Simplified model types for RAGI vector management.
Contains only essential data classes, without complex update checking.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


# Essential data classes
@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True
    trust_remote_code: bool = False
    use_fp16: bool = False

    @classmethod
    def for_model(cls, model_name: str, device: str = "cpu") -> "EmbeddingConfig":
        """
        Create configuration for a specific model.

        Args:
            model_name: Name of the model
            device: Device to use (cpu, cuda, mps)

        Returns:
            EmbeddingConfig instance
        """
        # Use FP16 for GPU acceleration
        use_fp16 = device in ("cuda", "mps")

        return cls(
            model_name=model_name,
            device=device,
            batch_size=32,
            normalize_embeddings=True,
            trust_remote_code=False,
            use_fp16=use_fp16
        )


@dataclass
class VectorStoreStats:
    """Basic statistics for vector store."""
    total_documents: int
    collection_name: str
    embedding_dimension: Optional[int] = None


# Simplified exceptions
class ModelCacheException(Exception):
    """Exception for model cache operations."""
    pass


class VectorStoreException(Exception):
    """Exception for vector store operations."""
    pass
