"""
Main vector store service that orchestrates all vector storage components.
Implements Facade pattern to provide a unified interface to the vector storage subsystem.
Follows Dependency Inversion Principle - depends on abstractions, not concretions.
"""

import asyncio
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

from src.models.schemas import Document, DocumentChunk, SearchResult
from src.utils.logger import LoggerMixin
from .base import (
    IVectorStore, IDatabaseConnection, IDocumentRepository, 
    IChunkRepository, IVectorSearch, IIndexManager,
    VectorStorageException, VectorSearchQuery
)
from .mongodb_client import MongoDBConnection, MongoDBConnectionFactory
from .document_repository import create_document_repository, create_chunk_repository
from .vector_search import create_vector_search
from .index_manager import create_index_manager


class MongoVectorStore(IVectorStore, LoggerMixin):
    """
    Main vector store service implementing the Facade pattern.
    Orchestrates all vector storage operations through clean interfaces.
    """
    
    def __init__(
        self,
        connection: IDatabaseConnection,
        document_repository: Optional[IDocumentRepository] = None,
        chunk_repository: Optional[IChunkRepository] = None,
        vector_search: Optional[IVectorSearch] = None,
        index_manager: Optional[IIndexManager] = None
    ):
        """
        Initialize vector store with dependency injection.
        
        Args:
            connection: Database connection
            document_repository: Document CRUD operations
            chunk_repository: Chunk CRUD operations  
            vector_search: Vector search operations
            index_manager: Index management operations
        """
        self.connection = connection
        
        # Dependency injection with factory fallbacks
        if isinstance(connection, MongoDBConnection):
            self.document_repo = document_repository or create_document_repository(connection)
            self.chunk_repo = chunk_repository or create_chunk_repository(connection)
            self.vector_search_service = vector_search or create_vector_search(connection)
            self.index_manager = index_manager or create_index_manager(connection)
        else:
            # For other database types, require explicit dependency injection
            if not all([document_repository, chunk_repository, vector_search, index_manager]):
                raise ValueError("All dependencies must be provided for non-MongoDB connections")
            
            self.document_repo = document_repository
            self.chunk_repo = chunk_repository
            self.vector_search_service = vector_search
            self.index_manager = index_manager
        
        self._initialized = False
    
    async def initialize(self, setup_indexes: bool = True) -> bool:
        """
        Initialize the vector store system.
        
        Args:
            setup_indexes: Whether to create default indexes
            
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing vector store system...")
            
            # Ensure database connection
            if not self.connection.is_connected:
                await self.connection.connect()
            
            # Test connection health
            if not await self.connection.health_check():
                raise VectorStorageException("Database connection health check failed")
            
            # Setup indexes if requested
            if setup_indexes:
                self.logger.info("Setting up default indexes...")
                index_results = await self.index_manager.setup_default_indexes()
                self.logger.info(f"Index setup results: {index_results}")
            
            self._initialized = True
            self.logger.info("Vector store system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {str(e)}")
            raise VectorStorageException(f"Vector store initialization failed: {str(e)}") from e
    
    async def store_documents_with_embeddings(
        self, 
        documents: List[Document], 
        chunks: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        Store documents and their chunks with embeddings.
        Implements retry logic for failed operations.
        
        Args:
            documents: List of documents to store
            chunks: List of document chunks with embeddings
            
        Returns:
            Dictionary with storage results
            
        Raises:
            VectorStorageException: If storage fails after retries
        """
        if not self._initialized:
            raise VectorStorageException("Vector store not initialized. Call initialize() first.")
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Storing {len(documents)} documents and {len(chunks)} chunks (attempt {attempt + 1}/{max_retries})")
                
                # Group chunks by document for validation
                chunks_by_doc = {}
                for chunk in chunks:
                    if chunk.document_id not in chunks_by_doc:
                        chunks_by_doc[chunk.document_id] = []
                    chunks_by_doc[chunk.document_id].append(chunk)
                
                # Store documents first
                document_ids = await self.document_repo.insert_documents(documents)
                
                # Update chunk document_ids if documents were assigned new IDs
                if len(document_ids) == len(documents):
                    for i, (doc, doc_id) in enumerate(zip(documents, document_ids)):
                        if doc.id != doc_id:  # Document got a new ID
                            # Update chunks that reference this document
                            old_doc_id = doc.id
                            if old_doc_id in chunks_by_doc:
                                for chunk in chunks_by_doc[old_doc_id]:
                                    chunk.document_id = doc_id
                
                # Store chunks with embeddings
                chunk_ids = await self.chunk_repo.insert_chunks(chunks)
                
                # Update document processing status
                for doc_id in document_ids:
                    await self.document_repo.update_document(
                        doc_id, 
                        {"processing_status": "completed"}
                    )
                
                result = {
                    "documents_stored": len(document_ids),
                    "chunks_stored": len(chunk_ids),
                    "document_ids": document_ids,
                    "chunk_ids": chunk_ids,
                    "timestamp": datetime.utcnow()
                }
                
                self.logger.info(f"Successfully stored documents and chunks: {result}")
                return result
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise VectorStorageException(f"Document storage failed after {max_retries} attempts: {str(e)}") from e
    
    async def search(self, query: VectorSearchQuery) -> List[SearchResult]:
        """
        Perform search based on query type.
        
        Args:
            query: Search query with parameters
            
        Returns:
            List of search results
            
        Raises:
            VectorStorageException: If search fails
        """
        if not self._initialized:
            raise VectorStorageException("Vector store not initialized. Call initialize() first.")
        
        try:
            self.logger.info(f"Executing search query: {query.query_text[:50]}...")
            
            # Prepare filters
            filters = {}
            if query.filters:
                filters.update(query.filters)
            
            # Execute search based on type
            if query.search_type == "semantic" and query.query_vector:
                results = await self.vector_search_service.similarity_search(
                    query_vector=query.query_vector,
                    limit=query.limit,
                    filters=filters
                )
            
            elif query.search_type == "hybrid" and query.query_vector and query.query_text:
                results = await self.vector_search_service.hybrid_search(
                    query_text=query.query_text,
                    query_vector=query.query_vector,
                    limit=query.limit,
                    filters=filters
                )
            
            elif query.search_type == "filtered" and query.query_vector:
                results = await self.vector_search_service.similarity_search_with_score(
                    query_vector=query.query_vector,
                    limit=query.limit,
                    score_threshold=query.min_score,
                    filters=filters
                )
            
            else:
                raise VectorStorageException(
                    f"Invalid search configuration. Type: {query.search_type}, "
                    f"Has vector: {query.query_vector is not None}, "
                    f"Has text: {query.query_text is not None}"
                )
            
            self.logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise VectorStorageException(f"Search operation failed: {str(e)}") from e
    
    async def delete_document_and_chunks(self, document_id: str) -> bool:
        """
        Delete a document and all its associated chunks.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if deletion successful
            
        Raises:
            VectorStorageException: If deletion fails
        """
        if not self._initialized:
            raise VectorStorageException("Vector store not initialized. Call initialize() first.")
        
        try:
            self.logger.info(f"Deleting document and chunks: {document_id}")
            
            # Delete chunks first (referential integrity)
            chunks_deleted = await self.chunk_repo.delete_chunks_by_document(document_id)
            
            # Delete the document
            document_deleted = await self.document_repo.delete_document(document_id)
            
            if document_deleted:
                self.logger.info(f"Deleted document {document_id} and {chunks_deleted} chunks")
                return True
            else:
                self.logger.warning(f"Document {document_id} not found for deletion")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {str(e)}")
            raise VectorStorageException(f"Document deletion failed: {str(e)}") from e
    
    async def get_document_with_chunks(self, document_id: str) -> Optional[Tuple[Document, List[DocumentChunk]]]:
        """
        Get a document with all its chunks.
        
        Args:
            document_id: Document ID
            
        Returns:
            Tuple of (Document, List[DocumentChunk]) or None if not found
        """
        try:
            # Get document
            document = await self.document_repo.get_document(document_id)
            if not document:
                return None
            
            # Get chunks
            chunks = await self.chunk_repo.get_chunks_by_document(document_id)
            
            return document, chunks
            
        except Exception as e:
            self.logger.error(f"Failed to get document with chunks {document_id}: {str(e)}")
            raise VectorStorageException(f"Document retrieval failed: {str(e)}") from e
    
    async def get_similar_documents(
        self, 
        document_id: str, 
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Find documents similar to the given document.
        
        Args:
            document_id: Reference document ID
            limit: Maximum results to return
            
        Returns:
            List of similar documents
        """
        try:
            # Get chunks for the reference document
            chunks = await self.chunk_repo.get_chunks_by_document(document_id)
            
            if not chunks:
                return []
            
            # Use the first chunk's embedding for similarity search
            reference_chunk = chunks[0]
            if not reference_chunk.embedding:
                return []
            
            # Search for similar chunks from different documents
            return await self.vector_search_service.similarity_search(
                query_vector=reference_chunk.embedding,
                limit=limit,
                filters={"document_id": {"$ne": document_id}}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to find similar documents: {str(e)}")
            raise VectorStorageException(f"Similar documents search failed: {str(e)}") from e
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            if isinstance(self.connection, MongoDBConnection):
                database = self.connection.get_database()
                
                # Get collection stats
                docs_collection = database["documents"]
                chunks_collection = database["chunks"]
                
                docs_count = await docs_collection.count_documents({})
                chunks_count = await chunks_collection.count_documents({})
                
                # Get index information
                docs_indexes = await self.index_manager.list_indexes("documents")
                chunks_indexes = await self.index_manager.list_indexes("chunks")
                
                return {
                    "documents_count": docs_count,
                    "chunks_count": chunks_count,
                    "documents_indexes": len(docs_indexes),
                    "chunks_indexes": len(chunks_indexes),
                    "connection_healthy": await self.connection.health_check(),
                    "timestamp": datetime.utcnow()
                }
            else:
                return {"error": "Statistics not available for this connection type"}
                
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {str(e)}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the vector store and cleanup resources."""
        try:
            if self.connection and self.connection.is_connected:
                await self.connection.disconnect()
            self._initialized = False
            self.logger.info("Vector store closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing vector store: {str(e)}")


# Factory functions for easy initialization
async def create_vector_store(
    connection_string: Optional[str] = None,
    database_name: Optional[str] = None,
    initialize: bool = True,
    setup_indexes: bool = True
) -> MongoVectorStore:
    """
    Create and initialize a vector store instance.
    
    Args:
        connection_string: MongoDB connection string
        database_name: Database name
        initialize: Whether to initialize the store
        setup_indexes: Whether to setup default indexes
        
    Returns:
        Initialized MongoVectorStore instance
    """
    connection = MongoDBConnectionFactory.create_connection(
        connection_string=connection_string,
        database_name=database_name
    )
    
    vector_store = MongoVectorStore(connection)
    
    if initialize:
        await vector_store.initialize(setup_indexes=setup_indexes)
    
    return vector_store


async def create_vector_store_from_connection(
    connection: MongoDBConnection,
    initialize: bool = True,
    setup_indexes: bool = True
) -> MongoVectorStore:
    """
    Create vector store from existing connection.
    
    Args:
        connection: Existing MongoDB connection
        initialize: Whether to initialize the store
        setup_indexes: Whether to setup default indexes
        
    Returns:
        MongoVectorStore instance
    """
    vector_store = MongoVectorStore(connection)
    
    if initialize:
        await vector_store.initialize(setup_indexes=setup_indexes)
    
    return vector_store 