"""
Vector search implementation using MongoDB Atlas Vector Search.
Implements Strategy pattern for different search types and follows clean architecture.
Supports semantic search, hybrid search, and filtered search operations.
"""

import asyncio
from typing import List, Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
from datetime import datetime

from src.models.schemas import SearchResult, DocumentChunk
from src.utils.logger import LoggerMixin
from .base import IVectorSearch, SearchError
from .mongodb_client import MongoDBConnection


class SearchStrategy(ABC):
    """
    Abstract strategy for different types of vector searches.
    Implements Strategy pattern for extensibility.
    """
    
    @abstractmethod
    async def search(
        self,
        collection,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute the search strategy."""
        pass


class SimilaritySearchStrategy(SearchStrategy):
    """Strategy for basic cosine similarity search."""
    
    async def search(
        self,
        collection,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using MongoDB Atlas Vector Search.
        
        Args:
            collection: MongoDB collection
            query_vector: Query embedding vector
            limit: Maximum results to return
            filters: Additional filter criteria
            
        Returns:
            List of search results with scores
        """
        # Build the aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": min(limit * 10, 10000),  # Search candidates
                    "limit": limit
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        # Add filters if provided
        if filters:
            pipeline.append({"$match": filters})
        
        # Project required fields
        pipeline.append({
            "$project": {
                "_id": 1,
                "document_id": 1,
                "content": 1,
                "chunk_index": 1,
                "metadata": 1,
                "created_at": 1,
                "score": 1
            }
        })
        
        return await collection.aggregate(pipeline).to_list(length=limit)


class HybridSearchStrategy(SearchStrategy):
    """Strategy combining vector search with text search."""
    
    async def search(
        self,
        collection,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and text search.
        
        Args:
            collection: MongoDB collection
            query_vector: Query embedding vector
            limit: Maximum results to return
            filters: Additional filter criteria
            query_text: Text query for hybrid search
            
        Returns:
            List of search results with combined scores
        """
        # Vector search pipeline
        vector_pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": min(limit * 5, 5000),
                    "limit": limit * 2  # Get more candidates for reranking
                }
            },
            {
                "$addFields": {
                    "vector_score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        # Text search pipeline (if query_text provided)
        if query_text:
            text_pipeline = [
                {
                    "$search": {
                        "index": "text_index",
                        "text": {
                            "query": query_text,
                            "path": "content"
                        }
                    }
                },
                {
                    "$addFields": {
                        "text_score": {"$meta": "searchScore"}
                    }
                },
                {"$limit": limit * 2}
            ]
            
            # Combine both searches using $unionWith
            vector_pipeline.extend([
                {
                    "$unionWith": {
                        "coll": collection.name,
                        "pipeline": text_pipeline
                    }
                },
                {
                    "$group": {
                        "_id": "$_id",
                        "document_id": {"$first": "$document_id"},
                        "content": {"$first": "$content"},
                        "chunk_index": {"$first": "$chunk_index"},
                        "metadata": {"$first": "$metadata"},
                        "created_at": {"$first": "$created_at"},
                        "vector_score": {"$max": "$vector_score"},
                        "text_score": {"$max": "$text_score"}
                    }
                },
                {
                    "$addFields": {
                        # Combine scores with weights
                        "score": {
                            "$add": [
                                {"$multiply": [{"$ifNull": ["$vector_score", 0]}, 0.7]},
                                {"$multiply": [{"$ifNull": ["$text_score", 0]}, 0.3]}
                            ]
                        }
                    }
                }
            ])
        
        # Add filters
        if filters:
            vector_pipeline.append({"$match": filters})
        
        # Sort by combined score and limit
        vector_pipeline.extend([
            {"$sort": {"score": -1}},
            {"$limit": limit}
        ])
        
        return await collection.aggregate(vector_pipeline).to_list(length=limit)


class FilteredSearchStrategy(SearchStrategy):
    """Strategy for vector search with advanced filtering."""
    
    async def search(
        self,
        collection,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform filtered vector search with score threshold.
        
        Args:
            collection: MongoDB collection
            query_vector: Query embedding vector
            limit: Maximum results to return
            filters: Additional filter criteria
            score_threshold: Minimum similarity score
            
        Returns:
            List of filtered search results
        """
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": min(limit * 10, 10000),
                    "limit": limit * 2  # Get more for filtering
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        # Apply score threshold
        if score_threshold > 0:
            pipeline.append({
                "$match": {"score": {"$gte": score_threshold}}
            })
        
        # Apply additional filters
        if filters:
            pipeline.append({"$match": filters})
        
        # Final projection and limit
        pipeline.extend([
            {
                "$project": {
                    "_id": 1,
                    "document_id": 1,
                    "content": 1,
                    "chunk_index": 1,
                    "metadata": 1,
                    "created_at": 1,
                    "score": 1
                }
            },
            {"$limit": limit}
        ])
        
        return await collection.aggregate(pipeline).to_list(length=limit)


class MongoVectorSearch(IVectorSearch, LoggerMixin):
    """
    MongoDB Atlas Vector Search implementation.
    Uses Strategy pattern for different search types.
    """
    
    def __init__(
        self,
        connection: MongoDBConnection,
        collection_name: str = "chunks",
        default_index: str = "vector_index"
    ):
        """
        Initialize vector search.
        
        Args:
            connection: MongoDB connection instance
            collection_name: Name of the chunks collection
            default_index: Default vector search index name
        """
        self.connection = connection
        self.collection_name = collection_name
        self.default_index = default_index
        
        # Initialize search strategies
        self.strategies = {
            "similarity": SimilaritySearchStrategy(),
            "hybrid": HybridSearchStrategy(),
            "filtered": FilteredSearchStrategy()
        }
    
    async def _check_atlas_features(self) -> bool:
        """
        Check if the MongoDB instance supports Atlas Vector Search.
        
        Returns:
            True if Atlas features are available, False otherwise
        """
        try:
            # Method 1: Check connection string (most reliable)
            conn_str = getattr(self.connection, 'connection_string', '')
            if "mongodb+srv://" in conn_str and "mongodb.net" in conn_str:
                self.logger.info("Atlas connection detected via connection string")
                return True
            
            # Method 2: Check buildInfo for Atlas indicators
            db = self.connection.get_database()
            server_info = await db.command("buildInfo")
            
            # Check for Atlas-specific indicators
            if "atlasVersion" in server_info:
                self.logger.info("Atlas connection detected via atlasVersion")
                return True
            
            # Check for enterprise modules (Atlas indicator)
            modules = server_info.get("modules", [])
            if "enterprise" in modules and "mongodb+srv" in conn_str:
                self.logger.info("Atlas connection detected via enterprise modules")
                return True
            
            # Method 3: Try to test vector search index creation
            try:
                # Try to check if vector search indexes can be listed
                # This is an Atlas-specific command
                await db.command({"listSearchIndexes": "test_collection"})
                self.logger.info("Atlas connection detected via search index capability")
                return True
            except Exception as index_e:
                # listSearchIndexes command fails on local MongoDB
                if "command not found" not in str(index_e).lower():
                    # If it's not "command not found", it might be Atlas
                    self.logger.info("Atlas connection detected via search index response")
                    return True
            
            # If none of the Atlas indicators are present
            version = server_info.get("version", "")
            self.logger.warning(f"Local MongoDB detected (version: {version})")
            self.logger.warning("Atlas Vector Search requires MongoDB Atlas cloud service")
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking Atlas features: {e}")
            return False
    
    async def _debug_vector_search_error(self, error: Exception) -> str:
        """
        Provide detailed debugging information for vector search errors.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Debugging message with solutions
        """
        error_str = str(error).lower()
        
        if "vectorsearch" in error_str or "unknown operator" in error_str:
            return (
                "❌ VECTOR SEARCH ERROR DETECTED\n"
                "🔍 CAUSE: $vectorSearch operator not supported\n"
                "📍 SOLUTION OPTIONS:\n"
                "   1. Switch to MongoDB Atlas (recommended for course)\n"
                "   2. Use local fallback with basic similarity search\n"
                "   3. Set up Atlas cluster: https://www.mongodb.com/atlas\n"
                f"   4. Current connection: {self.connection.connection_string}\n"
                "⚠️  Local MongoDB doesn't support Atlas Vector Search features"
            )
        
        return f"Unexpected vector search error: {error}"
    
    def _get_collection(self):
        """Get the chunks collection."""
        return self.connection.get_collection(self.collection_name)
    
    def _result_to_search_result(self, result_dict: Dict[str, Any]) -> SearchResult:
        """
        Convert MongoDB result to SearchResult model.
        
        Args:
            result_dict: MongoDB aggregation result
            
        Returns:
            SearchResult instance
        """
        return SearchResult(
            id=str(result_dict["_id"]),
            content=result_dict["content"],
            score=result_dict.get("score", 0.0),
            metadata={
                "document_id": result_dict["document_id"],
                "chunk_index": result_dict["chunk_index"],
                "created_at": result_dict.get("created_at"),
                **(result_dict.get("metadata", {}))
            }
        )
    
    async def similarity_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform semantic similarity search.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum results to return
            filters: Additional filter criteria
            
        Returns:
            List of search results
            
        Raises:
            SearchError: If search fails
        """
        try:
            # Check if vector search index is ready before performing search
            if hasattr(self.connection, 'get_database'):
                try:
                    database = self.connection.get_database()
                    result = await database.command("listSearchIndexes", self.collection_name)
                    indexes = result.get("cursor", {}).get("firstBatch", [])
                    
                    vector_index_ready = False
                    for idx in indexes:
                        if idx.get("name") == self.default_index:
                            status = idx.get("status", "unknown").upper()
                            if status in ["READY", "ACTIVE"]:
                                vector_index_ready = True
                                break
                            else:
                                self.logger.warning(f"Vector index '{self.default_index}' not ready (status: {status})")
                    
                    if not vector_index_ready:
                        self.logger.warning(f"Vector index '{self.default_index}' not ready for search")
                        return []  # Return empty results instead of failing
                        
                except Exception as index_check_error:
                    self.logger.debug(f"Could not check index status: {index_check_error}")
                    # Continue with search attempt
            
            collection = self._get_collection()
            strategy = self.strategies["similarity"]
            
            results = await strategy.search(
                collection=collection,
                query_vector=query_vector,
                limit=limit,
                filters=filters
            )
            
            search_results = [
                self._result_to_search_result(result) for result in results
            ]
            
            self.logger.info(f"Similarity search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            debug_msg = await self._debug_vector_search_error(e)
            self.logger.error(f"Similarity search failed: {debug_msg}")
            raise SearchError(f"Similarity search failed: {debug_msg}") from e
    
    async def similarity_search_with_score(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform similarity search with score filtering.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            filters: Additional filter criteria
            
        Returns:
            List of filtered search results
            
        Raises:
            SearchError: If search fails
        """
        try:
            collection = self._get_collection()
            strategy = self.strategies["filtered"]
            
            results = await strategy.search(
                collection=collection,
                query_vector=query_vector,
                limit=limit,
                filters=filters,
                score_threshold=score_threshold
            )
            
            search_results = [
                self._result_to_search_result(result) for result in results
            ]
            
            self.logger.info(
                f"Filtered search returned {len(search_results)} results "
                f"(threshold: {score_threshold})"
            )
            return search_results
            
        except Exception as e:
            self.logger.error(f"Filtered search failed: {str(e)}")
            raise SearchError(f"Filtered search failed: {str(e)}") from e
    
    async def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining text and vector search.
        
        Args:
            query_text: Text query
            query_vector: Query embedding vector
            limit: Maximum results to return
            filters: Additional filter criteria
            
        Returns:
            List of search results with combined scores
            
        Raises:
            SearchError: If search fails
        """
        try:
            collection = self._get_collection()
            strategy = self.strategies["hybrid"]
            
            results = await strategy.search(
                collection=collection,
                query_vector=query_vector,
                limit=limit,
                filters=filters,
                query_text=query_text
            )
            
            search_results = [
                self._result_to_search_result(result) for result in results
            ]
            
            self.logger.info(f"Hybrid search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {str(e)}")
            raise SearchError(f"Hybrid search failed: {str(e)}") from e
    
    async def search_similar_chunks(
        self,
        chunk_id: str,
        limit: int = 10,
        exclude_same_document: bool = True
    ) -> List[SearchResult]:
        """
        Find similar chunks to a given chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            limit: Maximum results to return
            exclude_same_document: Whether to exclude chunks from same document
            
        Returns:
            List of similar chunks
            
        Raises:
            SearchError: If search fails
        """
        try:
            collection = self._get_collection()
            
            # First, get the reference chunk's embedding
            from bson import ObjectId
            ref_chunk = await collection.find_one({"_id": ObjectId(chunk_id)})
            
            if not ref_chunk or "embedding" not in ref_chunk:
                raise SearchError(f"Chunk {chunk_id} not found or has no embedding")
            
            # Prepare filters
            filters = {"_id": {"$ne": ObjectId(chunk_id)}}  # Exclude self
            
            if exclude_same_document:
                filters["document_id"] = {"$ne": ref_chunk["document_id"]}
            
            # Perform similarity search
            return await self.similarity_search(
                query_vector=ref_chunk["embedding"],
                limit=limit,
                filters=filters
            )
            
        except Exception as e:
            self.logger.error(f"Similar chunks search failed: {str(e)}")
            raise SearchError(f"Similar chunks search failed: {str(e)}") from e


# Factory function for dependency injection
def create_vector_search(connection: MongoDBConnection) -> IVectorSearch:
    """Create a vector search instance."""
    return MongoVectorSearch(connection) 