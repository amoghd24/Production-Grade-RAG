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
    
    async def _debug_vector_search_error(self, error: Exception) -> str:
        """
        Provide basic debugging information for vector search errors.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Simple error message
        """
        error_str = str(error).lower()
        
        if "vectorsearch" in error_str or "unknown operator" in error_str:
            return "Vector search operator not supported - ensure MongoDB Atlas is configured"
        
        return str(error)
    
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