"""
Data models and schemas for the Second Brain AI Assistant.
This module defines all the Pydantic models used throughout the application.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


class DocumentType(str, Enum):
    """Types of documents that can be processed."""
    MARKDOWN = "markdown"
    PDF = "pdf"
    WEB_PAGE = "web_page"
    NOTION_PAGE = "notion_page"
    TEXT = "text"


class ContentSource(str, Enum):
    """Sources where content can be collected from."""
    NOTION = "notion"
    WEB_CRAWL = "web_crawl"
    UPLOAD = "upload"
    API = "api"


class ProcessingStatus(str, Enum):
    """Processing status of documents."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(BaseModel):
    """Main document model."""
    id: Optional[str] = None
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Raw document content")
    source: ContentSource = Field(..., description="Source of the document")
    source_url: Optional[HttpUrl] = Field(None, description="Original URL if applicable")
    document_type: DocumentType = Field(..., description="Type of document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Processing info
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    error_message: Optional[str] = None
    
    # Quality metrics
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Content quality score")
    word_count: Optional[int] = Field(None, ge=0)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentChunk(BaseModel):
    """Represents a chunk of a document for vector storage."""
    id: Optional[str] = None
    document_id: str = Field(..., description="Reference to parent document")
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., ge=0, description="Index of chunk within document")
    
    # Embeddings
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    embedding_model: Optional[str] = Field(None, description="Model used for embedding")
    
    # Chunk metadata
    start_char: Optional[int] = Field(None, ge=0)
    end_char: Optional[int] = Field(None, ge=0)
    word_count: int = Field(..., ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk metadata")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SourceAttribution(BaseModel):
    """Rich source attribution information for search results."""
    title: str = Field(..., description="Document title")
    url: Optional[HttpUrl] = Field(None, description="Source URL if available")
    source_type: ContentSource = Field(..., description="Type of source")
    document_type: DocumentType = Field(..., description="Type of document")
    chunk_type: Optional[str] = Field(None, description="Type of chunk (parent/child)")
    strategies_used: List[str] = Field(default_factory=list, description="Search strategies that found this")
    created_at: Optional[datetime] = Field(None, description="When source was created")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchResult(BaseModel):
    """Enhanced search result model with rich source attribution."""
    id: str = Field(..., description="Unique identifier for the result")
    content: str = Field(..., description="Content of the search result")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    
    # Enhanced source attribution
    source: Optional[SourceAttribution] = Field(None, description="Rich source information")
    
    # Legacy metadata for backward compatibility
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryResponse(BaseModel):
    """Enhanced response model for user queries with source attribution."""
    response: str = Field(..., description="Generated response")
    sources: List[SourceAttribution] = Field(default_factory=list, description="Source attribution")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in response")
    processing_time_ms: Optional[float] = Field(None, description="Processing time")
    search_strategy: Optional[str] = Field(None, description="Search strategy used")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_used: Optional[str] = Field(None, description="AI model used for generation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


 