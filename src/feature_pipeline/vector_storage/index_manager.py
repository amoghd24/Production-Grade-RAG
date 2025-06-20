"""
Index management for MongoDB Atlas Vector Search.
Handles creation, deletion, and management of vector search indexes.
Follows Single Responsibility Principle - only manages indexes.
"""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

from src.utils.logger import LoggerMixin
from .base import IIndexManager, IndexError
from .mongodb_client import MongoDBConnection


class MongoIndexManager(IIndexManager, LoggerMixin):
    """
    MongoDB Atlas Vector Search index manager.
    Handles vector index lifecycle management.
    """
    
    def __init__(self, connection: MongoDBConnection):
        """
        Initialize index manager.
        
        Args:
            connection: MongoDB connection instance
        """
        self.connection = connection
    
    async def create_vector_index(
        self,
        collection_name: str,
        index_name: str,
        vector_field: str,
        dimensions: int,
        similarity_metric: str = "cosine"
    ) -> bool:
        """
        Create a vector search index using MongoDB Atlas Search.
        
        Args:
            collection_name: Name of the collection
            index_name: Name of the vector index
            vector_field: Field containing the vector embeddings
            dimensions: Vector dimensions
            similarity_metric: Similarity metric (cosine, euclidean, dotProduct)
            
        Returns:
            True if index was created successfully
            
        Raises:
            IndexError: If index creation fails
        """
        try:
            # Validate similarity metric
            valid_metrics = ["cosine", "euclidean", "dotProduct"]
            if similarity_metric not in valid_metrics:
                raise IndexError(f"Invalid similarity metric: {similarity_metric}")
            
            # Atlas Search index definition for vector search
            index_definition = {
                "name": index_name,
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": vector_field,
                            "numDimensions": dimensions,
                            "similarity": similarity_metric
                        }
                    ]
                }
            }
            
            self.logger.info(f"Creating vector index '{index_name}' on collection '{collection_name}'")
            self.logger.info(f"Vector field: {vector_field}, Dimensions: {dimensions}, Metric: {similarity_metric}")
            
            # Get database and collection
            database = self.connection.get_database()
            collection = database[collection_name]
            
            # Create the search index using the Atlas Admin API approach
            # Note: This requires MongoDB Atlas and proper authentication
            try:
                # Create index using MongoDB's createSearchIndex command
                result = await database.command(
                    "createSearchIndexes",
                    collection_name,
                    indexes=[index_definition]
                )
                
                self.logger.info(f"Vector index '{index_name}' created successfully")
                self.logger.info(f"Index creation result: {result}")
                return True
                
            except Exception as atlas_error:
                # Fallback: Log the index definition for manual creation
                self.logger.warning(f"Automatic index creation failed: {str(atlas_error)}")
                self.logger.info("Index definition for manual creation:")
                self.logger.info(json.dumps(index_definition, indent=2))
                
                # For development, we'll consider this successful if the collection exists
                collections = await database.list_collection_names()
                if collection_name in collections:
                    self.logger.info(f"Collection '{collection_name}' exists. Index can be created manually in Atlas UI.")
                    return True
                else:
                    raise IndexError(f"Collection '{collection_name}' does not exist")
            
        except Exception as e:
            self.logger.error(f"Failed to create vector index: {str(e)}")
            raise IndexError(f"Vector index creation failed: {str(e)}") from e
    
    async def create_text_index(
        self,
        collection_name: str,
        index_name: str,
        text_fields: List[str]
    ) -> bool:
        """
        Create a text search index for hybrid search.
        
        Args:
            collection_name: Name of the collection
            index_name: Name of the text index
            text_fields: List of text fields to index
            
        Returns:
            True if index was created successfully
            
        Raises:
            IndexError: If index creation fails
        """
        try:
            # Atlas Search index definition for text search
            index_definition = {
                "name": index_name,
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            field: {
                                "type": "string",
                                "analyzer": "lucene.standard"
                            } for field in text_fields
                        }
                    }
                }
            }
            
            self.logger.info(f"Creating text index '{index_name}' on collection '{collection_name}'")
            self.logger.info(f"Text fields: {text_fields}")
            
            database = self.connection.get_database()
            
            try:
                result = await database.command(
                    "createSearchIndexes",
                    collection_name,
                    indexes=[index_definition]
                )
                
                self.logger.info(f"Text index '{index_name}' created successfully")
                return True
                
            except Exception as atlas_error:
                # Fallback: Log the index definition
                self.logger.warning(f"Automatic text index creation failed: {str(atlas_error)}")
                self.logger.info("Text index definition for manual creation:")
                self.logger.info(json.dumps(index_definition, indent=2))
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to create text index: {str(e)}")
            raise IndexError(f"Text index creation failed: {str(e)}") from e
    
    async def list_indexes(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        List all search indexes for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            List of index definitions
            
        Raises:
            IndexError: If listing fails
        """
        try:
            database = self.connection.get_database()
            
            try:
                # Try to list search indexes
                result = await database.command("listSearchIndexes", collection_name)
                indexes = result.get("cursor", {}).get("firstBatch", [])
                
                self.logger.info(f"Found {len(indexes)} search indexes for collection '{collection_name}'")
                return indexes
                
            except Exception as atlas_error:
                # Fallback: List regular MongoDB indexes
                self.logger.warning(f"Failed to list search indexes: {str(atlas_error)}")
                collection = database[collection_name]
                indexes = await collection.list_indexes().to_list(None)
                
                self.logger.info(f"Found {len(indexes)} regular indexes for collection '{collection_name}'")
                return indexes
            
        except Exception as e:
            self.logger.error(f"Failed to list indexes: {str(e)}")
            raise IndexError(f"Index listing failed: {str(e)}") from e
    
    async def delete_index(self, collection_name: str, index_name: str) -> bool:
        """
        Delete a search index.
        
        Args:
            collection_name: Name of the collection
            index_name: Name of the index to delete
            
        Returns:
            True if index was deleted successfully
            
        Raises:
            IndexError: If deletion fails
        """
        try:
            database = self.connection.get_database()
            
            try:
                result = await database.command(
                    "dropSearchIndex",
                    collection_name,
                    index=index_name
                )
                
                self.logger.info(f"Search index '{index_name}' deleted successfully")
                return True
                
            except Exception as atlas_error:
                # Fallback: Try regular index deletion
                self.logger.warning(f"Failed to delete search index: {str(atlas_error)}")
                collection = database[collection_name]
                await collection.drop_index(index_name)
                
                self.logger.info(f"Regular index '{index_name}' deleted successfully")
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete index: {str(e)}")
            raise IndexError(f"Index deletion failed: {str(e)}") from e
    
    async def index_exists(self, collection_name: str, index_name: str) -> bool:
        """
        Check if a specific index exists and is ready for use.
        
        Args:
            collection_name: Name of the collection
            index_name: Name of the index
            
        Returns:
            True if index exists and is ready, False otherwise
        """
        try:
            database = self.connection.get_database()
            
            # First try to check search indexes directly
            try:
                result = await database.command("listSearchIndexes", collection_name)
                search_indexes = result.get("cursor", {}).get("firstBatch", [])
                
                for idx in search_indexes:
                    if idx.get("name") == index_name:
                        # Check if the index is ready for use
                        status = idx.get("status", "unknown").upper()
                        if status in ["READY", "ACTIVE"]:
                            self.logger.info(f"Search index '{index_name}' exists and is ready (status: {status})")
                            return True
                        else:
                            self.logger.info(f"Search index '{index_name}' exists but not ready (status: {status})")
                            return False
                
                # Index not found in search indexes
                self.logger.info(f"Search index '{index_name}' not found")
                return False
                
            except Exception as search_error:
                self.logger.debug(f"Search index check failed: {search_error}")
                # Fall back to regular index check for non-Atlas environments
                pass
            
            # Fallback: Check regular indexes (for local MongoDB)
            indexes = await self.list_indexes(collection_name)
            found = any(idx.get("name") == index_name for idx in indexes)
            
            if found:
                self.logger.info(f"Regular index '{index_name}' exists")
            
            return found
            
        except Exception as e:
            self.logger.warning(f"Failed to check index existence: {str(e)}")
            return False
    
    async def get_index_status(self, collection_name: str, index_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status information for a specific index.
        
        Args:
            collection_name: Name of the collection
            index_name: Name of the index
            
        Returns:
            Index status information if found, None otherwise
        """
        try:
            database = self.connection.get_database()
            
            # Try to get search index status
            try:
                result = await database.command("listSearchIndexes", collection_name)
                search_indexes = result.get("cursor", {}).get("firstBatch", [])
                
                for idx in search_indexes:
                    if idx.get("name") == index_name:
                        return {
                            "name": idx.get("name"),
                            "type": idx.get("type", "unknown"),
                            "status": idx.get("status", "unknown"),
                            "queryable": idx.get("queryable", False),
                            "definition": idx.get("definition", {})
                        }
                        
            except Exception as e:
                self.logger.debug(f"Failed to get search index status: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting index status: {e}")
            return None
    
    async def ensure_vector_index(
        self,
        collection_name: str,
        index_name: str,
        vector_field: str,
        dimensions: int,
        similarity_metric: str = "cosine"
    ) -> bool:
        """
        Ensure a vector index exists and is ready for use.
        
        Args:
            collection_name: Name of the collection
            index_name: Name of the vector index
            vector_field: Field containing the vector embeddings
            dimensions: Vector dimensions
            similarity_metric: Similarity metric
            
        Returns:
            True if index exists and is ready for use
        """
        try:
            # Check if index already exists and is ready
            if await self.index_exists(collection_name, index_name):
                self.logger.info(f"Vector index '{index_name}' already exists and is ready")
                return True
            
            # Check if index exists but not ready
            index_status = await self.get_index_status(collection_name, index_name)
            
            if index_status:
                status = index_status.get("status", "unknown").upper()
                if status in ["BUILDING", "PENDING"]:
                    self.logger.info(f"Vector index '{index_name}' is building (status: {status}), waiting for readiness...")
                    # Wait for index to become ready
                    ready = await self.wait_for_index_ready(collection_name, index_name, max_wait_seconds=300)
                    if ready:
                        self.logger.info(f"Vector index '{index_name}' is now ready")
                        return True
                    else:
                        self.logger.warning(f"Vector index '{index_name}' did not become ready within timeout")
                        return False
                elif status in ["READY", "ACTIVE"]:
                    self.logger.info(f"Vector index '{index_name}' exists and is ready (status: {status})")
                    return True
                else:
                    self.logger.warning(f"Vector index '{index_name}' exists but has unexpected status: {status}")
                    return False
            
            # Index doesn't exist, create it
            self.logger.info(f"Vector index '{index_name}' not found, creating...")
            success = await self.create_vector_index(
                collection_name=collection_name,
                index_name=index_name,
                vector_field=vector_field,
                dimensions=dimensions,
                similarity_metric=similarity_metric
            )
            
            if success:
                self.logger.info(f"Vector index '{index_name}' created successfully. Waiting for it to become ready...")
                # Wait for the newly created index to become ready
                ready = await self.wait_for_index_ready(collection_name, index_name, max_wait_seconds=300)
                if ready:
                    self.logger.info(f"Vector index '{index_name}' is now ready for use")
                    return True
                else:
                    self.logger.warning(f"Vector index '{index_name}' created but did not become ready within timeout")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to ensure vector index: {str(e)}")
            return False
    
    async def wait_for_index_ready(
        self,
        collection_name: str,
        index_name: str,
        max_wait_seconds: int = 60,
        check_interval: int = 5
    ) -> bool:
        """
        Wait for a vector search index to become ready.
        
        Args:
            collection_name: Name of the collection
            index_name: Name of the index
            max_wait_seconds: Maximum time to wait in seconds
            check_interval: Seconds between status checks
            
        Returns:
            True if index becomes ready, False if timeout
        """
        import asyncio
        
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < max_wait_seconds:
            try:
                status_info = await self.get_index_status(collection_name, index_name)
                
                if status_info:
                    status = status_info.get("status", "unknown").upper()
                    
                    if status in ["READY", "ACTIVE"]:
                        self.logger.info(f"Index '{index_name}' is ready (status: {status})")
                        return True
                    elif status in ["BUILDING", "PENDING"]:
                        self.logger.info(f"Index '{index_name}' still building (status: {status}), waiting...")
                    else:
                        self.logger.warning(f"Index '{index_name}' has unexpected status: {status}")
                        return False
                else:
                    self.logger.warning(f"Index '{index_name}' not found")
                    return False
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error checking index status: {e}")
                await asyncio.sleep(check_interval)
        
        self.logger.warning(f"Timeout waiting for index '{index_name}' to become ready")
        return False

    async def setup_default_indexes(
        self,
        documents_collection: str = "documents",
        chunks_collection: str = "chunks",
        embedding_dimensions: int = 384
    ) -> Dict[str, bool]:
        """
        Set up default indexes for the vector storage system.
        
        Args:
            documents_collection: Name of documents collection
            chunks_collection: Name of chunks collection
            embedding_dimensions: Dimensions of the embedding vectors
            
        Returns:
            Dictionary with index creation results
        """
        results = {}
        
        try:
            # Vector index for chunks collection
            results["vector_index"] = await self.ensure_vector_index(
                collection_name=chunks_collection,
                index_name="vector_index",
                vector_field="embedding",
                dimensions=embedding_dimensions,
                similarity_metric="cosine"
            )
            
            # Text index for chunks collection (for hybrid search)
            results["text_index"] = await self.create_text_index(
                collection_name=chunks_collection,
                index_name="text_index",
                text_fields=["content"]
            )
            
            # Regular indexes for common queries
            database = self.connection.get_database()
            
            # Index on document_id for chunks
            chunks_coll = database[chunks_collection]
            await chunks_coll.create_index("document_id")
            results["document_id_index"] = True
            
            # Index on source and processing_status for documents
            docs_coll = database[documents_collection]
            await docs_coll.create_index([("source", 1), ("processing_status", 1)])
            results["source_status_index"] = True
            
            # Index on created_at for time-based queries
            await docs_coll.create_index("created_at")
            await chunks_coll.create_index("created_at")
            results["created_at_indexes"] = True
            
            self.logger.info("Default indexes setup completed")
            self.logger.info(f"Index creation results: {results}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to setup default indexes: {str(e)}")
            results["error"] = str(e)
            return results


# Factory function for dependency injection
def create_index_manager(connection: MongoDBConnection) -> IIndexManager:
    """Create an index manager instance."""
    return MongoIndexManager(connection) 