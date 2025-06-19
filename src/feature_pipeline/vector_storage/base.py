"""
Abstract base classes and interfaces for the vector storage system.
This module defines the contracts that concrete implementations must follow.
Follows Interface Segregation Principle - small, focused interfaces.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, AsyncContextManager
from datetime import datetime
from pydantic import BaseModel, Field

from src.models.schemas import Document, DocumentChunk, SearchResult


# Vector storage specific models
class VectorSearchQuery(BaseModel):
    """Search query model for vector storage operations."""
    query_text: str = Field(..., description="Search query text")
    query_vector: Optional[List[float]] = Field(None, description="Query embedding vector")
    search_type: str = Field(default="semantic", description="Type of search: semantic, hybrid, filtered")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")


class IDatabaseConnection(ABC):
    """
    Abstract interface for database connections.
    Follows Single Responsibility Principle - only manages connections.
    """
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if connection is healthy."""
        pass
    
    @abstractmethod
    def get_client(self) -> Any:
        """Get the underlying database client."""
        pass


class IDocumentRepository(ABC):
    """
    Abstract interface for document CRUD operations.
    Follows Single Responsibility Principle - only handles document persistence.
    """
    
    @abstractmethod
    async def insert_document(self, document: Document) -> str:
        """Insert a single document and return its ID."""
        pass
    
    @abstractmethod
    async def insert_documents(self, documents: List[Document]) -> List[str]:
        """Insert multiple documents and return their IDs."""
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        pass
    
    @abstractmethod
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document with given changes."""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        pass
    
    @abstractmethod
    async def get_documents_by_filter(
        self, 
        filters: Dict[str, Any], 
        limit: int = 100
    ) -> List[Document]:
        """Get documents matching the filter criteria."""
        pass


class IChunkRepository(ABC):
    """
    Abstract interface for document chunk operations.
    Separate from document repository following Interface Segregation.
    """
    
    @abstractmethod
    async def insert_chunk(self, chunk: DocumentChunk) -> str:
        """Insert a single chunk and return its ID."""
        pass
    
    @abstractmethod
    async def insert_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """Insert multiple chunks and return their IDs."""
        pass
    
    @abstractmethod
    async def get_chunks_by_document(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a specific document."""
        pass
    
    @abstractmethod
    async def delete_chunks_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document, return count deleted."""
        pass


class IVectorSearch(ABC):
    """
    Abstract interface for vector search operations.
    Follows Single Responsibility Principle - only handles search.
    """
    
    @abstractmethod
    async def similarity_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform similarity search using vector embeddings."""
        pass
    
    @abstractmethod
    async def similarity_search_with_score(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform similarity search and return relevance scores."""
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform hybrid search combining text and vector search."""
        pass


class IIndexManager(ABC):
    """
    Abstract interface for managing vector search indexes.
    Follows Single Responsibility Principle - only manages indexes.
    """
    
    @abstractmethod
    async def create_vector_index(
        self,
        collection_name: str,
        index_name: str,
        vector_field: str,
        dimensions: int,
        similarity_metric: str = "cosine"
    ) -> bool:
        """Create a vector search index on the specified field."""
        pass
    
    @abstractmethod
    async def list_indexes(self, collection_name: str) -> List[Dict[str, Any]]:
        """List all indexes for a collection."""
        pass
    
    @abstractmethod
    async def delete_index(self, collection_name: str, index_name: str) -> bool:
        """Delete a specific index."""
        pass
    
    @abstractmethod
    async def index_exists(self, collection_name: str, index_name: str) -> bool:
        """Check if an index exists."""
        pass


class IVectorStore(ABC):
    """
    High-level interface for vector store operations.
    Follows Facade pattern - provides unified interface to subsystems.
    """
    
    @abstractmethod
    async def store_documents_with_embeddings(
        self, 
        documents: List[Document], 
        chunks: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """Store documents and their chunks with embeddings."""
        pass
    
    @abstractmethod
    async def search(self, query: VectorSearchQuery) -> List[SearchResult]:
        """Perform search based on query type (semantic, hybrid, etc.)."""
        pass
    
    @abstractmethod
    async def delete_document_and_chunks(self, document_id: str) -> bool:
        """Delete a document and all its associated chunks."""
        pass


class VectorStorageException(Exception):
    """Base exception for vector storage operations."""
    pass


class ConnectionError(VectorStorageException):
    """Raised when database connection fails."""
    pass


class IndexError(VectorStorageException):
    """Raised when index operations fail."""
    pass


class SearchError(VectorStorageException):
    """Raised when search operations fail."""
    pass


class RepositoryError(VectorStorageException):
    """Raised when repository operations fail."""
    pass 