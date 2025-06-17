"""
Repository implementations for document and chunk persistence.
Implements Repository pattern with proper error handling and validation.
Follows Single Responsibility and Dependency Inversion principles.
"""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from bson import ObjectId
from pymongo.errors import DuplicateKeyError, PyMongoError

from src.models.schemas import Document, DocumentChunk, ProcessingStatus
from src.utils.logger import LoggerMixin
from .base import IDocumentRepository, IChunkRepository, RepositoryError
from .mongodb_client import MongoDBConnection


class MongoDocumentRepository(IDocumentRepository, LoggerMixin):
    """
    MongoDB implementation of document repository.
    Handles CRUD operations for Document objects.
    """
    
    def __init__(self, connection: MongoDBConnection, collection_name: str = "documents"):
        """
        Initialize document repository.
        
        Args:
            connection: MongoDB connection instance
            collection_name: Name of the documents collection
        """
        self.connection = connection
        self.collection_name = collection_name
    
    def _get_collection(self):
        """Get the documents collection."""
        return self.connection.get_collection(self.collection_name)
    
    def _document_to_dict(self, document: Document) -> Dict[str, Any]:
        """
        Convert Document model to MongoDB document.
        
        Args:
            document: Document instance
            
        Returns:
            Dictionary representation for MongoDB
        """
        doc_dict = document.model_dump(exclude_none=True)
        
        # Handle datetime serialization
        if doc_dict.get('created_at'):
            doc_dict['created_at'] = document.created_at
        if doc_dict.get('updated_at'):
            doc_dict['updated_at'] = document.updated_at
        
        # Convert enums to strings
        doc_dict['source'] = document.source.value
        doc_dict['document_type'] = document.document_type.value
        doc_dict['processing_status'] = document.processing_status.value
        
        # Handle URL conversion
        if document.source_url:
            doc_dict['source_url'] = str(document.source_url)
        
        return doc_dict
    
    def _dict_to_document(self, doc_dict: Dict[str, Any]) -> Document:
        """
        Convert MongoDB document to Document model.
        
        Args:
            doc_dict: MongoDB document dictionary
            
        Returns:
            Document instance
        """
        # Convert ObjectId to string
        if '_id' in doc_dict:
            doc_dict['id'] = str(doc_dict['_id'])
            del doc_dict['_id']
        
        return Document(**doc_dict)
    
    async def insert_document(self, document: Document) -> str:
        """
        Insert a single document.
        
        Args:
            document: Document to insert
            
        Returns:
            Document ID
            
        Raises:
            RepositoryError: If insertion fails
        """
        try:
            collection = self._get_collection()
            doc_dict = self._document_to_dict(document)
            
            # Remove id if present (MongoDB will generate _id)
            doc_dict.pop('id', None)
            
            result = await collection.insert_one(doc_dict)
            document_id = str(result.inserted_id)
            
            self.logger.info(f"Inserted document: {document_id}")
            return document_id
            
        except PyMongoError as e:
            self.logger.error(f"Failed to insert document: {str(e)}")
            raise RepositoryError(f"Document insertion failed: {str(e)}") from e
    
    async def insert_documents(self, documents: List[Document]) -> List[str]:
        """
        Insert multiple documents.
        
        Args:
            documents: List of documents to insert
            
        Returns:
            List of document IDs
            
        Raises:
            RepositoryError: If insertion fails
        """
        if not documents:
            return []
        
        try:
            collection = self._get_collection()
            doc_dicts = []
            
            for document in documents:
                doc_dict = self._document_to_dict(document)
                doc_dict.pop('id', None)  # Remove id field
                doc_dicts.append(doc_dict)
            
            result = await collection.insert_many(doc_dicts)
            document_ids = [str(oid) for oid in result.inserted_ids]
            
            self.logger.info(f"Inserted {len(document_ids)} documents")
            return document_ids
            
        except PyMongoError as e:
            self.logger.error(f"Failed to insert documents: {str(e)}")
            raise RepositoryError(f"Bulk document insertion failed: {str(e)}") from e
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document instance or None if not found
            
        Raises:
            RepositoryError: If retrieval fails
        """
        try:
            collection = self._get_collection()
            doc_dict = await collection.find_one({"_id": ObjectId(document_id)})
            
            if doc_dict:
                return self._dict_to_document(doc_dict)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get document {document_id}: {str(e)}")
            raise RepositoryError(f"Document retrieval failed: {str(e)}") from e
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a document with given changes.
        
        Args:
            document_id: Document ID
            updates: Dictionary of fields to update
            
        Returns:
            True if document was updated, False if not found
            
        Raises:
            RepositoryError: If update fails
        """
        try:
            collection = self._get_collection()
            
            # Add updated timestamp
            updates['updated_at'] = datetime.utcnow()
            
            result = await collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": updates}
            )
            
            if result.modified_count > 0:
                self.logger.info(f"Updated document: {document_id}")
                return True
            return False
            
        except PyMongoError as e:
            self.logger.error(f"Failed to update document {document_id}: {str(e)}")
            raise RepositoryError(f"Document update failed: {str(e)}") from e
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if document was deleted, False if not found
            
        Raises:
            RepositoryError: If deletion fails
        """
        try:
            collection = self._get_collection()
            result = await collection.delete_one({"_id": ObjectId(document_id)})
            
            if result.deleted_count > 0:
                self.logger.info(f"Deleted document: {document_id}")
                return True
            return False
            
        except PyMongoError as e:
            self.logger.error(f"Failed to delete document {document_id}: {str(e)}")
            raise RepositoryError(f"Document deletion failed: {str(e)}") from e
    
    async def get_documents_by_filter(
        self, 
        filters: Dict[str, Any], 
        limit: int = 100
    ) -> List[Document]:
        """
        Get documents matching filter criteria.
        
        Args:
            filters: MongoDB filter criteria
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
            
        Raises:
            RepositoryError: If query fails
        """
        try:
            collection = self._get_collection()
            cursor = collection.find(filters).limit(limit)
            
            documents = []
            async for doc_dict in cursor:
                documents.append(self._dict_to_document(doc_dict))
            
            self.logger.info(f"Found {len(documents)} documents matching filters")
            return documents
            
        except PyMongoError as e:
            self.logger.error(f"Failed to query documents: {str(e)}")
            raise RepositoryError(f"Document query failed: {str(e)}") from e


class MongoChunkRepository(IChunkRepository, LoggerMixin):
    """
    MongoDB implementation of chunk repository.
    Handles CRUD operations for DocumentChunk objects.
    """
    
    def __init__(self, connection: MongoDBConnection, collection_name: str = "chunks"):
        """
        Initialize chunk repository.
        
        Args:
            connection: MongoDB connection instance
            collection_name: Name of the chunks collection
        """
        self.connection = connection
        self.collection_name = collection_name
    
    def _get_collection(self):
        """Get the chunks collection."""
        return self.connection.get_collection(self.collection_name)
    
    def _chunk_to_dict(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """
        Convert DocumentChunk model to MongoDB document.
        
        Args:
            chunk: DocumentChunk instance
            
        Returns:
            Dictionary representation for MongoDB
        """
        chunk_dict = chunk.model_dump(exclude_none=True)
        
        # Handle datetime serialization
        if chunk_dict.get('created_at'):
            chunk_dict['created_at'] = chunk.created_at
        
        return chunk_dict
    
    def _dict_to_chunk(self, chunk_dict: Dict[str, Any]) -> DocumentChunk:
        """
        Convert MongoDB document to DocumentChunk model.
        
        Args:
            chunk_dict: MongoDB document dictionary
            
        Returns:
            DocumentChunk instance
        """
        # Convert ObjectId to string
        if '_id' in chunk_dict:
            chunk_dict['id'] = str(chunk_dict['_id'])
            del chunk_dict['_id']
        
        return DocumentChunk(**chunk_dict)
    
    async def insert_chunk(self, chunk: DocumentChunk) -> str:
        """
        Insert a single chunk.
        
        Args:
            chunk: DocumentChunk to insert
            
        Returns:
            Chunk ID
            
        Raises:
            RepositoryError: If insertion fails
        """
        try:
            collection = self._get_collection()
            chunk_dict = self._chunk_to_dict(chunk)
            
            # Remove id if present
            chunk_dict.pop('id', None)
            
            result = await collection.insert_one(chunk_dict)
            chunk_id = str(result.inserted_id)
            
            self.logger.debug(f"Inserted chunk: {chunk_id}")
            return chunk_id
            
        except PyMongoError as e:
            self.logger.error(f"Failed to insert chunk: {str(e)}")
            raise RepositoryError(f"Chunk insertion failed: {str(e)}") from e
    
    async def insert_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Insert multiple chunks in batches to handle large insertions.
        
        Args:
            chunks: List of chunks to insert
            
        Returns:
            List of chunk IDs
            
        Raises:
            RepositoryError: If insertion fails
        """
        if not chunks:
            return []
        
        try:
            collection = self._get_collection()
            chunk_dicts = []
            batch_size = 500  # Process in batches of 500
            all_chunk_ids = []
            
            for chunk in chunks:
                chunk_dict = self._chunk_to_dict(chunk)
                chunk_dict.pop('id', None)
                chunk_dicts.append(chunk_dict)
                
                # Process in batches
                if len(chunk_dicts) >= batch_size:
                    result = await collection.insert_many(chunk_dicts)
                    batch_ids = [str(oid) for oid in result.inserted_ids]
                    all_chunk_ids.extend(batch_ids)
                    chunk_dicts = []  # Clear the batch
                    self.logger.info(f"Inserted batch of {len(batch_ids)} chunks")
            
            # Insert any remaining chunks
            if chunk_dicts:
                result = await collection.insert_many(chunk_dicts)
                batch_ids = [str(oid) for oid in result.inserted_ids]
                all_chunk_ids.extend(batch_ids)
                self.logger.info(f"Inserted final batch of {len(batch_ids)} chunks")
            
            self.logger.info(f"Total chunks inserted: {len(all_chunk_ids)}")
            return all_chunk_ids
            
        except PyMongoError as e:
            self.logger.error(f"Failed to insert chunks: {str(e)}")
            raise RepositoryError(f"Bulk chunk insertion failed: {str(e)}") from e
    
    async def get_chunks_by_document(self, document_id: str) -> List[DocumentChunk]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunks for the document
            
        Raises:
            RepositoryError: If query fails
        """
        try:
            collection = self._get_collection()
            cursor = collection.find({"document_id": document_id}).sort("chunk_index", 1)
            
            chunks = []
            async for chunk_dict in cursor:
                chunks.append(self._dict_to_chunk(chunk_dict))
            
            self.logger.debug(f"Found {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except PyMongoError as e:
            self.logger.error(f"Failed to get chunks for document {document_id}: {str(e)}")
            raise RepositoryError(f"Chunk query failed: {str(e)}") from e
    
    async def delete_chunks_by_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of chunks deleted
            
        Raises:
            RepositoryError: If deletion fails
        """
        try:
            collection = self._get_collection()
            result = await collection.delete_many({"document_id": document_id})
            
            deleted_count = result.deleted_count
            self.logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count
            
        except PyMongoError as e:
            self.logger.error(f"Failed to delete chunks for document {document_id}: {str(e)}")
            raise RepositoryError(f"Chunk deletion failed: {str(e)}") from e


# Factory functions for dependency injection
def create_document_repository(connection: MongoDBConnection) -> IDocumentRepository:
    """Create a document repository instance."""
    return MongoDocumentRepository(connection)


def create_chunk_repository(connection: MongoDBConnection) -> IChunkRepository:
    """Create a chunk repository instance."""
    return MongoChunkRepository(connection) 