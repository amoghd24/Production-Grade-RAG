"""
Advanced RAG Configuration System
Provides configuration management for RAG strategies and components.
"""

from typing import Dict, Any, List, Optional, Union
import yaml
from pathlib import Path

from src.models.rag import (
    RAGStrategy,
    ChunkingStrategy,
    SearchStrategy,
    ChunkingConfig,
    EmbeddingConfig,
    SearchConfig,
    RAGConfiguration
)
from src.utils.logger import LoggerMixin


class RAGConfigManager(LoggerMixin):
    """Manages RAG configuration loading and validation."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the config manager."""
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[RAGConfiguration] = None
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> RAGConfiguration:
        """
        Load RAG configuration from YAML file or use defaults.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            RAGConfiguration instance
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                self._config = RAGConfiguration(**config_data)
                self.logger.info(f"Loaded RAG config from {self.config_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load config from {self.config_path}: {e}")
                self.logger.info("Using default configuration")
                self._config = RAGConfiguration()
        else:
            self.logger.info("No config file found, using defaults")
            self._config = RAGConfiguration()
        
        return self._config
    
    def get_config(self) -> RAGConfiguration:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def save_config(self, config: RAGConfiguration, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: RAGConfiguration to save
            path: Optional path to save to, uses self.config_path if not provided
        """
        save_path = Path(path) if path else self.config_path
        if not save_path:
            raise ValueError("No save path specified")
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and save (with enum string conversion)
        config_dict = config.model_dump(exclude_none=True, mode='json')
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Saved RAG config to {save_path}")
    
    def validate_config(self, config: RAGConfiguration) -> bool:
        """
        Validate configuration consistency.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Strategy compatibility checks
            if config.strategy == RAGStrategy.CONTEXTUAL and not config.enable_contextual_retrieval:
                self.logger.warning("Contextual strategy requires enable_contextual_retrieval=True")
                return False
            
            if config.strategy == RAGStrategy.PARENT_RETRIEVAL and not config.enable_parent_retrieval:
                self.logger.warning("Parent retrieval strategy requires enable_parent_retrieval=True")
                return False
            
            # Chunking validation
            if config.chunking.strategy == ChunkingStrategy.PARENT_CHILD:
                if not config.chunking.parent_chunk_size or not config.chunking.child_chunk_size:
                    self.logger.warning("Parent-child chunking requires both parent and child chunk sizes")
                    return False
            
            self.logger.info("RAG configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False


# Global config manager instance
_config_manager: Optional[RAGConfigManager] = None


def get_rag_config_manager(config_path: Optional[Union[str, Path]] = None) -> RAGConfigManager:
    """Get global RAG configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = RAGConfigManager(config_path)
    return _config_manager


def get_rag_config(config_path: Optional[Union[str, Path]] = None) -> RAGConfiguration:
    """Get current RAG configuration."""
    manager = get_rag_config_manager(config_path)
    return manager.get_config() 