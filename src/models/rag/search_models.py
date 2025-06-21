"""
RAG Search Models
Vector search and query models for the RAG system.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class VectorSearchQuery(BaseModel):
    """Search query model for vector storage operations."""
    query_text: str = Field(..., description="Search query text")
    query_vector: Optional[List[float]] = Field(None, description="Query embedding vector")
    search_type: str = Field(default="semantic", description="Type of search: semantic, hybrid, filtered")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters") 