"""
Feature flag management for Advanced RAG capabilities.
Provides runtime control over experimental features.
"""

from typing import Dict, Any, Optional
from enum import Enum
from src.config.settings import settings
from src.config.rag_config import get_rag_config, RAGStrategy
from src.utils.logger import LoggerMixin


class FeatureFlag(str, Enum):
    """Available feature flags."""
    ADVANCED_RAG = "advanced_rag"
    PARENT_RETRIEVAL = "parent_retrieval"
    HYBRID_SEARCH = "hybrid_search"
    QUALITY_FILTERING = "quality_filtering"


class FeatureFlagManager(LoggerMixin):
    """Manages feature flags for Advanced RAG capabilities."""
    
    def __init__(self):
        """Initialize the feature flag manager."""
        self._flags_cache: Optional[Dict[str, bool]] = None
        self._rag_config = None
    
    def _load_flags(self) -> Dict[str, bool]:
        """Load feature flags from settings and RAG config."""
        if self._flags_cache is not None:
            return self._flags_cache
        
        # Load from environment/settings first
        flags = {
            FeatureFlag.ADVANCED_RAG: settings.ENABLE_ADVANCED_RAG,
            FeatureFlag.PARENT_RETRIEVAL: settings.ENABLE_PARENT_RETRIEVAL,
            FeatureFlag.HYBRID_SEARCH: settings.ENABLE_HYBRID_SEARCH,
            FeatureFlag.QUALITY_FILTERING: settings.ENABLE_QUALITY_FILTERING,
        }
        
        # Override with RAG config if available (only if file exists)
        try:
            from pathlib import Path
            config_path = Path(settings.RAG_CONFIG_PATH)
            if config_path.exists():
                if self._rag_config is None:
                    self._rag_config = get_rag_config(settings.RAG_CONFIG_PATH)
                
                flags.update({
                    FeatureFlag.ADVANCED_RAG: self._rag_config.enable_advanced_rag,
                    FeatureFlag.PARENT_RETRIEVAL: self._rag_config.enable_parent_retrieval,
                    FeatureFlag.HYBRID_SEARCH: self._rag_config.enable_hybrid_search,
                    FeatureFlag.QUALITY_FILTERING: self._rag_config.enable_quality_filtering,
                })
            
        except Exception as e:
            self.logger.debug(f"No RAG config file found, using settings: {e}")
        
        self._flags_cache = flags
        self.logger.info(f"Loaded feature flags: {flags}")
        return flags
    
    def is_enabled(self, flag: FeatureFlag) -> bool:
        """
        Check if a feature flag is enabled.
        
        Args:
            flag: Feature flag to check
            
        Returns:
            True if enabled, False otherwise
        """
        flags = self._load_flags()
        return flags.get(flag, False)
    
    def get_active_rag_strategy(self) -> str:
        """
        Get the currently active RAG strategy based on feature flags.
        
        Returns:
            Active RAG strategy name
        """
        if not self.is_enabled(FeatureFlag.ADVANCED_RAG):
            return RAGStrategy.BASIC
        
        # Try to get from RAG config first
        try:
            if self._rag_config is None:
                self._rag_config = get_rag_config(settings.RAG_CONFIG_PATH)
            return self._rag_config.strategy
        except Exception:
            pass
        
        # Fallback to environment setting
        return settings.RAG_STRATEGY
    
    def should_use_parent_retrieval(self) -> bool:
        """Check if parent retrieval should be used."""
        return (self.is_enabled(FeatureFlag.ADVANCED_RAG) and 
                self.is_enabled(FeatureFlag.PARENT_RETRIEVAL))
    
    def should_use_hybrid_search(self) -> bool:
        """Check if hybrid search should be used."""
        return (self.is_enabled(FeatureFlag.ADVANCED_RAG) and 
                self.is_enabled(FeatureFlag.HYBRID_SEARCH))
    
    def should_use_advanced_search(self) -> bool:
        """Check if advanced search strategies should be used."""
        return self.is_enabled(FeatureFlag.ADVANCED_RAG)
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """
        Get chunking configuration based on active flags.
        
        Returns:
            Dictionary with chunking parameters
        """
        config = {
            'chunk_size': settings.CHUNK_SIZE,
            'chunk_overlap': settings.CHUNK_OVERLAP,
        }
        
        if self.should_use_parent_retrieval():
            config.update({
                'parent_chunk_size': settings.PARENT_CHUNK_SIZE,
                'child_chunk_size': settings.CHILD_CHUNK_SIZE,
                'parent_overlap': settings.PARENT_CHUNK_OVERLAP,
                'child_overlap': settings.CHILD_CHUNK_OVERLAP,
            })
        
        # Try to get from RAG config
        try:
            if self._rag_config is None:
                self._rag_config = get_rag_config(settings.RAG_CONFIG_PATH)
            
            chunking_config = self._rag_config.chunking
            config.update({
                'chunk_size': chunking_config.chunk_size,
                'chunk_overlap': chunking_config.chunk_overlap,
            })
            
            if chunking_config.parent_chunk_size:
                config['parent_chunk_size'] = chunking_config.parent_chunk_size
            if chunking_config.child_chunk_size:
                config['child_chunk_size'] = chunking_config.child_chunk_size
                
        except Exception as e:
            self.logger.debug(f"Using default chunking config: {e}")
        
        return config
    
    def refresh_flags(self) -> None:
        """Refresh feature flags cache."""
        self._flags_cache = None
        self._rag_config = None
        self.logger.info("Feature flags cache refreshed")
    
    def get_all_flags(self) -> Dict[str, bool]:
        """Get all current feature flag states."""
        return self._load_flags().copy()


# Global feature flag manager instance
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flags() -> FeatureFlagManager:
    """Get global feature flag manager."""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
    return _feature_flag_manager


def is_feature_enabled(flag: FeatureFlag) -> bool:
    """Check if a feature is enabled (convenience function)."""
    return get_feature_flags().is_enabled(flag) 