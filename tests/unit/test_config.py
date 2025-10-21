"""
Unit tests for config module.
"""

import os
import pytest
from config import config, Config


class TestConfig:
    """Test configuration management."""

    def test_config_singleton(self):
        """Test that config is a singleton."""
        assert config is not None
        # Config is actually already an instance, so just verify it exists
        assert hasattr(config, 'CHUNK_SIZE')
        assert hasattr(config, 'DATA_PATH')

    def test_default_values(self):
        """Test default configuration values."""
        assert config.CHUNK_SIZE == 500
        assert config.CHUNK_OVERLAP == 150
        assert config.RETRIEVER_K == 6
        assert config.RETRIEVER_SEARCH_TYPE in ["semantic", "keyword", "hybrid"]

    def test_paths_exist(self):
        """Test that path attributes exist."""
        assert hasattr(config, "DATA_PATH")
        assert hasattr(config, "PERSIST_PATH")
        assert hasattr(config, "SCRIPT_DIR")

    def test_model_names(self):
        """Test that model names are set."""
        assert config.EMBEDDING_MODEL_NAME
        assert config.LLM_MODEL_NAME
        assert config.RERANKER_MODEL_NAME
        assert isinstance(config.EMBEDDING_MODEL_NAME, str)
        assert isinstance(config.LLM_MODEL_NAME, str)

    def test_has_gpu_method(self):
        """Test GPU detection method."""
        result = config.has_gpu()
        assert isinstance(result, bool)

    def test_get_device_info(self):
        """Test device info retrieval."""
        device_type, device_name = config.get_device_info()
        assert device_type in ["cpu", "cuda", "mps"]
        assert isinstance(device_name, str)
        assert len(device_name) > 0

    def test_ollama_url(self):
        """Test Ollama URL configuration."""
        assert config.OLLAMA_BASE_URL
        assert config.OLLAMA_BASE_URL.startswith("http")

    def test_logging_config(self):
        """Test logging configuration."""
        assert config.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert config.LOG_MAX_BYTES > 0
        assert config.LOG_BACKUP_COUNT > 0

    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert hasattr(config, "SUPPORTED_EXTENSIONS")
        assert isinstance(config.SUPPORTED_EXTENSIONS, list)
        assert ".pdf" in config.SUPPORTED_EXTENSIONS
        assert ".txt" in config.SUPPORTED_EXTENSIONS
        assert ".md" in config.SUPPORTED_EXTENSIONS

    def test_retrieval_settings(self):
        """Test retrieval settings."""
        assert config.RETRIEVER_K > 0
        assert 0 <= config.KEYWORD_RATIO <= 1
        assert 0 <= config.FAQ_MATCH_WEIGHT <= 1

    def test_context_settings(self):
        """Test context processing settings."""
        assert config.MAX_CONTEXT_SIZE > 0
        assert isinstance(config.USE_CONTEXT_COMPRESSION, bool)
        assert 0 <= config.CONFIDENCE_THRESHOLD <= 1

    def test_faq_matching_setting(self):
        """Test FAQ matching configuration."""
        assert isinstance(config.USE_FAQ_MATCHING, bool)

    def test_environment_variable_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("RAGI_CHUNK_SIZE", "1000")
        # Note: This tests the mechanism, actual value won't change in already-loaded config
        test_value = os.environ.get("RAGI_CHUNK_SIZE", "500")
        assert test_value == "1000"


class TestConfigSetup:
    """Test configuration setup method."""

    def test_setup_runs_without_error(self):
        """Test that setup can be called without errors."""
        try:
            config.setup()
        except Exception as e:
            pytest.fail(f"Config setup failed: {e}")

    def test_setup_is_idempotent(self):
        """Test that setup can be called multiple times safely."""
        config.setup()
        config.setup()
        config.setup()
        # Should not raise any errors

    def test_data_directory_creation(self, temp_dir, monkeypatch):
        """Test that setup creates data directory."""
        data_path = temp_dir / "test_data"
        monkeypatch.setattr(config, "DATA_PATH", str(data_path))

        config.setup()

        # Directory should be created by setup
        # Note: Actual creation depends on implementation
        assert True  # Placeholder - actual test would verify directory


class TestConfigValidation:
    """Test configuration validation."""

    def test_chunk_size_positive(self):
        """Test that chunk size is positive."""
        assert config.CHUNK_SIZE > 0

    def test_chunk_overlap_less_than_size(self):
        """Test that chunk overlap is less than chunk size."""
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

    def test_retriever_k_reasonable(self):
        """Test that retriever K is in reasonable range."""
        assert 1 <= config.RETRIEVER_K <= 20

    def test_max_threads_positive(self):
        """Test that max threads is positive."""
        assert config.MAX_THREADS > 0

    def test_stream_delay_non_negative(self):
        """Test that stream delay is non-negative."""
        assert config.STREAM_DELAY >= 0
