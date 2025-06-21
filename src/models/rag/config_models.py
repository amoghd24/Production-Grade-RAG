"""
RAG Configuration Models
Provides configurable strategies for chunking, retrieval, and embedding.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


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
    MULTI_STRATEGY = "multi_strategy"


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