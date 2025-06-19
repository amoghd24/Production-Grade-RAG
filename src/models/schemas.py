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
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchQuery(BaseModel):
    """Search query model."""
    query: str = Field(..., min_length=1, description="Search query text")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")


class SearchResult(BaseModel):
    """Search result model."""
    id: str = Field(..., description="Unique identifier for the result")
    content: str = Field(..., description="Content of the search result")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChatMessage(BaseModel):
    """Chat message model."""
    id: Optional[str] = None
    content: str = Field(..., min_length=1)
    role: str = Field(..., description="user or assistant")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Context
    sources: List[SearchResult] = Field(default_factory=list, description="Sources used for response")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatSession(BaseModel):
    """Chat session model."""
    id: Optional[str] = None
    title: str = Field(default="New Chat", description="Session title")
    messages: List[ChatMessage] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CrawlJob(BaseModel):
    """Web crawling job model."""
    id: Optional[str] = None
    start_url: HttpUrl = Field(..., description="Starting URL for crawl")
    max_pages: int = Field(default=10, ge=1, le=1000)
    allowed_domains: List[str] = Field(default_factory=list)
    
    # Status
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    pages_crawled: int = Field(default=0, ge=0)
    pages_processed: int = Field(default=0, ge=0)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    crawled_urls: List[str] = Field(default_factory=list)
    failed_urls: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingPipeline(BaseModel):
    """Pipeline processing status model."""
    id: Optional[str] = None
    pipeline_type: str = Field(..., description="Type of pipeline (crawl, process, embed, etc.)")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    
    # Progress tracking
    total_items: int = Field(default=0, ge=0)
    processed_items: int = Field(default=0, ge=0)
    failed_items: int = Field(default=0, ge=0)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Configuration and logs
    config: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 