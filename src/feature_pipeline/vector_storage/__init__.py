"""
Vector Storage Module for Second Brain AI Assistant.
MongoDB Atlas Vector Search implementation with clean architecture.
"""

from .base import (
    IDatabaseConnection,
    IDocumentRepository,
    IChunkRepository,
    IVectorSearch,
    IIndexManager,
    IVectorStore,
    VectorStorageException,
    ConnectionError,
    IndexError,
    SearchError,
    RepositoryError
)

from .mongodb_client import (
    MongoDBConnection,
    get_mongodb_connection,
    close_mongodb_connection
)

from .vector_store import (
    MongoVectorStore,
    create_vector_store
)

# Main entry points
__all__ = [
    "MongoVectorStore",
    "create_vector_store",
    "get_mongodb_connection",
    "VectorStorageException",
    "ConnectionError",
    "SearchError",
    "RepositoryError",
    "IndexError"
]
