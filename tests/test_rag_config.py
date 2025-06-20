"""
Tests for RAG configuration system.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config.rag_config import (
    RAGConfiguration,
    RAGConfigManager,
    ChunkingConfig,
    EmbeddingConfig,
    SearchConfig,
    RAGStrategy,
    ChunkingStrategy,
    SearchStrategy,
    get_rag_config_manager,
    get_rag_config
)
from src.config.feature_flags import (
    FeatureFlagManager,
    FeatureFlag,
    get_feature_flags,
    is_feature_enabled
)
from src.config.settings import settings


class TestRAGConfiguration:
    """Test RAG configuration models."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = RAGConfiguration()
        
        assert config.strategy == RAGStrategy.BASIC
        assert config.enable_contextual_retrieval is False
        assert config.enable_parent_retrieval is False
        assert config.enable_hybrid_search is False
        assert config.enable_quality_filtering is True
        
        # Test nested configs
        assert config.chunking.strategy == ChunkingStrategy.BASIC
        assert config.chunking.chunk_size == 1000
        assert config.chunking.chunk_overlap == 200
        
        assert config.embedding.primary_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding.dimensions == 384
        
        assert config.search.primary_strategy == SearchStrategy.SIMILARITY
        assert config.search.similarity_threshold == 0.7
    
    def test_chunking_config_validation(self):
        """Test chunking configuration validation."""
        # Valid configuration
        config = ChunkingConfig(chunk_size=1000, chunk_overlap=200)
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        
        # Invalid configuration - overlap >= chunk_size
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            ChunkingConfig(chunk_size=1000, chunk_overlap=1000)
    
    def test_search_config_validation(self):
        """Test search configuration validation."""
        # Valid configuration
        config = SearchConfig(vector_weight=0.7, text_weight=0.3)
        assert config.vector_weight == 0.7
        assert config.text_weight == 0.3
        
        # Invalid configuration - weights don't sum to 1
        with pytest.raises(ValueError, match="Vector and text weights must sum to 1.0"):
            SearchConfig(vector_weight=0.5, text_weight=0.3)


class TestRAGConfigManager:
    """Test RAG configuration manager."""
    
    def test_load_default_config(self):
        """Test loading default configuration when no file exists."""
        manager = RAGConfigManager()
        config = manager.load_config()
        
        assert isinstance(config, RAGConfiguration)
        assert config.strategy == RAGStrategy.BASIC
    
    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
strategy: contextual
enable_contextual_retrieval: true
enable_parent_retrieval: false

chunking:
  strategy: contextual
  chunk_size: 1500
  chunk_overlap: 300
  add_document_context: true

embedding:
  primary_model: "sentence-transformers/all-mpnet-base-v2"
  dimensions: 768

search:
  primary_strategy: hybrid
  similarity_threshold: 0.8
  vector_weight: 0.6
  text_weight: 0.4
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            try:
                manager = RAGConfigManager(f.name)
                config = manager.load_config()
                
                assert config.strategy == RAGStrategy.CONTEXTUAL
                assert config.enable_contextual_retrieval is True
                assert config.enable_parent_retrieval is False
                
                assert config.chunking.strategy == ChunkingStrategy.CONTEXTUAL
                assert config.chunking.chunk_size == 1500
                assert config.chunking.chunk_overlap == 300
                assert config.chunking.add_document_context is True
                
                assert config.embedding.primary_model == "sentence-transformers/all-mpnet-base-v2"
                assert config.embedding.dimensions == 768
                
                assert config.search.primary_strategy == SearchStrategy.HYBRID
                assert config.search.similarity_threshold == 0.8
                assert config.search.vector_weight == 0.6
                assert config.search.text_weight == 0.4
                
            finally:
                os.unlink(f.name)
    
    def test_save_config(self):
        """Test saving configuration to YAML file."""
        config = RAGConfiguration(
            strategy=RAGStrategy.CONTEXTUAL,
            enable_contextual_retrieval=True
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                manager = RAGConfigManager()
                manager.save_config(config, f.name)
                
                # Load it back and verify
                manager2 = RAGConfigManager(f.name)
                loaded_config = manager2.load_config()
                
                assert loaded_config.strategy == RAGStrategy.CONTEXTUAL
                assert loaded_config.enable_contextual_retrieval is True
                
            finally:
                os.unlink(f.name)
    
    def test_validate_config(self):
        """Test configuration validation."""
        manager = RAGConfigManager()
        
        # Valid configuration
        valid_config = RAGConfiguration(
            strategy=RAGStrategy.CONTEXTUAL,
            enable_contextual_retrieval=True
        )
        assert manager.validate_config(valid_config) is True
        
        # Invalid configuration - strategy requires feature flag
        invalid_config = RAGConfiguration(
            strategy=RAGStrategy.CONTEXTUAL,
            enable_contextual_retrieval=False  # Contradictory
        )
        assert manager.validate_config(invalid_config) is False


class TestFeatureFlagManager:
    """Test feature flag manager."""
    
    def test_default_flags(self):
        """Test default feature flag values."""
        manager = FeatureFlagManager()
        
        # All advanced features should be disabled by default
        assert manager.is_enabled(FeatureFlag.ADVANCED_RAG) is False
        assert manager.is_enabled(FeatureFlag.CONTEXTUAL_CHUNKING) is False
        assert manager.is_enabled(FeatureFlag.PARENT_RETRIEVAL) is False
        assert manager.is_enabled(FeatureFlag.HYBRID_SEARCH) is False
        assert manager.is_enabled(FeatureFlag.QUALITY_FILTERING) is True  # Default enabled
    
    @patch('src.config.feature_flags.settings')
    def test_flags_from_settings(self, mock_settings):
        """Test loading flags from settings."""
        mock_settings.ENABLE_ADVANCED_RAG = True
        mock_settings.ENABLE_CONTEXTUAL_CHUNKING = True
        mock_settings.ENABLE_PARENT_RETRIEVAL = False
        mock_settings.ENABLE_HYBRID_SEARCH = True
        mock_settings.ENABLE_QUALITY_FILTERING = True
        mock_settings.RAG_CONFIG_PATH = "nonexistent.yaml"
        
        manager = FeatureFlagManager()
        
        assert manager.is_enabled(FeatureFlag.ADVANCED_RAG) is True
        assert manager.is_enabled(FeatureFlag.CONTEXTUAL_CHUNKING) is True
        assert manager.is_enabled(FeatureFlag.PARENT_RETRIEVAL) is False
        assert manager.is_enabled(FeatureFlag.HYBRID_SEARCH) is True
        assert manager.is_enabled(FeatureFlag.QUALITY_FILTERING) is True
    
    def test_get_active_strategy(self):
        """Test getting active RAG strategy."""
        manager = FeatureFlagManager()
        
        # Should return BASIC when advanced RAG is disabled
        assert manager.get_active_rag_strategy() == RAGStrategy.BASIC
    
    def test_chunking_config(self):
        """Test getting chunking configuration."""
        manager = FeatureFlagManager()
        config = manager.get_chunking_config()
        
        assert 'chunk_size' in config
        assert 'chunk_overlap' in config
        assert isinstance(config['chunk_size'], int)
        assert isinstance(config['chunk_overlap'], int)
    
    def test_refresh_flags(self):
        """Test refreshing feature flags cache."""
        manager = FeatureFlagManager()
        
        # Load flags once
        manager._load_flags()
        assert manager._flags_cache is not None
        
        # Refresh should clear cache
        manager.refresh_flags()
        assert manager._flags_cache is None


class TestGlobalFunctions:
    """Test global utility functions."""
    
    def test_get_rag_config_manager(self):
        """Test getting global RAG config manager."""
        manager1 = get_rag_config_manager()
        manager2 = get_rag_config_manager()
        
        # Should return the same instance (singleton)
        assert manager1 is manager2
    
    def test_get_rag_config(self):
        """Test getting RAG configuration."""
        config = get_rag_config()
        assert isinstance(config, RAGConfiguration)
    
    def test_get_feature_flags(self):
        """Test getting global feature flags manager."""
        flags1 = get_feature_flags()
        flags2 = get_feature_flags()
        
        # Should return the same instance (singleton)
        assert flags1 is flags2
    
    def test_is_feature_enabled(self):
        """Test convenience function for checking feature flags."""
        result = is_feature_enabled(FeatureFlag.ADVANCED_RAG)
        assert isinstance(result, bool)


class TestIntegration:
    """Integration tests for the configuration system."""
    
    def test_end_to_end_config_flow(self):
        """Test complete configuration flow."""
        yaml_content = """
strategy: contextual
enable_contextual_retrieval: true

chunking:
  strategy: contextual
  chunk_size: 1200
  add_document_context: true

search:
  primary_strategy: hybrid
  similarity_threshold: 0.75
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            try:
                # Load config (clear cache first)
                import src.config.rag_config as rag_config_module
                rag_config_module._config_manager = None  # Clear cache
                config = get_rag_config(f.name)
                
                # Verify configuration
                assert config.strategy == RAGStrategy.CONTEXTUAL
                assert config.enable_contextual_retrieval is True
                assert config.chunking.chunk_size == 1200
                assert config.chunking.add_document_context is True
                assert config.search.similarity_threshold == 0.75
                
                # Test feature flags integration
                with patch('src.config.settings.settings.RAG_CONFIG_PATH', f.name):
                    flags = get_feature_flags()
                    flags.refresh_flags()  # Force reload
                    
                    # Feature flags should reflect config
                    chunking_config = flags.get_chunking_config()
                    assert chunking_config['chunk_size'] == 1200
                
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__]) 