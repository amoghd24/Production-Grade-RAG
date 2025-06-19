"""
Advanced RAG Configuration System
Provides configurable strategies for chunking, retrieval, and embedding.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import yaml
from pathlib import Path

from src.utils.logger import LoggerMixin


class RAGStrategy(str, Enum):
    """Available RAG strategies."""
    BASIC = "basic"
    CONTEXTUAL = "contextual"
    PARENT_RETRIEVAL = "parent_retrieval"
    HYBRID = "hybrid"


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    BASIC = "basic"
    CONTEXTUAL = "contextual"
    PARENT_CHILD = "parent_child"
    ADAPTIVE = "adaptive"


class SearchStrategy(str, Enum):
    """Available search strategies."""
    SIMILARITY = "similarity"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"
    PARENT_CHILD = "parent_child"


class ChunkingConfig(BaseModel):
    """Configuration for chunking strategies."""
    strategy: ChunkingStrategy = ChunkingStrategy.BASIC
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    
    # Parent-child specific settings
    parent_chunk_size: Optional[int] = Field(default=2000, ge=500, le=8000)
    child_chunk_size: Optional[int] = Field(default=400, ge=100, le=2000)
    parent_overlap: Optional[int] = Field(default=400, ge=0)
    child_overlap: Optional[int] = Field(default=100, ge=0)
    
    # Contextual enhancement settings
    add_document_context: bool = Field(default=False)
    add_section_headers: bool = Field(default=False)
    context_template: Optional[str] = None
    
    @field_validator('chunk_overlap')
    @classmethod
    def overlap_less_than_size(cls, v, info):
        chunk_size = info.data.get('chunk_size', 1000)
        if v >= chunk_size:
            raise ValueError('Chunk overlap must be less than chunk size')
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    primary_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    fallback_models: List[str] = Field(default_factory=list)
    dimensions: int = Field(default=384, ge=128)
    batch_size: int = Field(default=32, ge=1, le=256)
    normalize_embeddings: bool = Field(default=True)


class SearchConfig(BaseModel):
    """Configuration for search strategies."""
    primary_strategy: SearchStrategy = SearchStrategy.SIMILARITY
    fallback_strategies: List[SearchStrategy] = Field(default_factory=list)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=100)
    
    # Hybrid search weights
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    text_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Reranking settings
    enable_reranking: bool = Field(default=False)
    rerank_top_k: int = Field(default=20, ge=5, le=100)
    
    @field_validator('text_weight')
    @classmethod
    def weights_sum_to_one(cls, v, info):
        vector_weight = info.data.get('vector_weight', 0.7)
        if abs((vector_weight + v) - 1.0) > 0.01:
            raise ValueError('Vector and text weights must sum to 1.0')
        return v


class RAGConfiguration(BaseModel):
    """Main RAG configuration container."""
    # Strategy selection
    strategy: RAGStrategy = RAGStrategy.BASIC
    
    # Component configurations
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    
    # Feature flags
    enable_advanced_rag: bool = Field(default=False)
    enable_contextual_retrieval: bool = Field(default=False)
    enable_parent_retrieval: bool = Field(default=False)
    enable_hybrid_search: bool = Field(default=False)
    enable_quality_filtering: bool = Field(default=True)
    
    # Performance settings
    max_concurrent_requests: int = Field(default=10, ge=1, le=100)
    request_timeout: int = Field(default=30, ge=5, le=300)
    cache_enabled: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, ge=60)  # 1 hour default
    
    # Monitoring
    enable_metrics: bool = Field(default=True)
    log_level: str = Field(default="INFO")


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