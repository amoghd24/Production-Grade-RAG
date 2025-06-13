"""
Vector Storage Module for Second Brain AI Assistant.

This module provides a complete vector storage solution using MongoDB Atlas Vector Search.
Implements clean architecture with SOLID principles and proper separation of concerns.

Key Components:
- MongoVectorStore: Main facade for all vector storage operations
- MongoDB connection management with Motor async driver
- Document and chunk repositories with CRUD operations
- Vector search with multiple strategies (semantic, hybrid, filtered)
- Index management for MongoDB Atlas Vector Search
- LangChain integration for RAG applications

Usage:
    from src.feature_pipeline.vector_storage import create_vector_store
    
    # Create and initialize vector store
    vector_store = await create_vector_store()
    
    # Store documents with embeddings
    await vector_store.store_documents_with_embeddings(documents, chunks)
    
    # Search for similar content
    results = await vector_store.search(search_query)
    
    # Clean up
    await vector_store.close()
"""

from .base import (
    # Abstract interfaces
    IDatabaseConnection,
    IDocumentRepository,
    IChunkRepository,
    IVectorSearch,
    IIndexManager,
    IVectorStore,
    
    # Exceptions
    VectorStorageException,
    ConnectionError,
    IndexError,
    SearchError,
    RepositoryError
)

from .mongodb_client import (
    MongoDBConnection,
    MongoDBConnectionFactory,
    get_mongodb_connection,
    close_mongodb_connection
)

from .document_repository import (
    MongoDocumentRepository,
    MongoChunkRepository,
    create_document_repository,
    create_chunk_repository
)

from .vector_search import (
    MongoVectorSearch,
    SearchStrategy,
    SimilaritySearchStrategy,
    HybridSearchStrategy,
    FilteredSearchStrategy,
    create_vector_search
)

from .index_manager import (
    MongoIndexManager,
    create_index_manager
)

from .vector_store import (
    MongoVectorStore,
    create_vector_store,
    create_vector_store_from_connection
)

# Public API - Main entry points
__all__ = [
    # Main facade
    "MongoVectorStore",
    "create_vector_store",
    "create_vector_store_from_connection",
    
    # Connection management
    "MongoDBConnection",
    "MongoDBConnectionFactory",
    
    # Core interfaces (for testing and extension)
    "IVectorStore",
    "IDatabaseConnection",
    "IDocumentRepository",
    "IChunkRepository",
    "IVectorSearch",
    "IIndexManager",
    
    # Exceptions
    "VectorStorageException",
    "ConnectionError",
    "SearchError",
    "RepositoryError",
    "IndexError",
    
    # Advanced usage
    "MongoDocumentRepository",
    "MongoChunkRepository",
    "MongoVectorSearch",
    "MongoIndexManager",
    
    # Search strategies
    "SearchStrategy",
    "SimilaritySearchStrategy",
    "HybridSearchStrategy",
    "FilteredSearchStrategy",
]

# Version information
__version__ = "1.0.0"
__author__ = "Second Brain AI Assistant"
__description__ = "Vector storage system with MongoDB Atlas Vector Search"
