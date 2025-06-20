"""
Source Attribution Service for Advanced RAG.
Provides rich source attribution information for search results and responses.
"""

from typing import List, Dict, Optional, Any
from src.models.schemas import (
    SearchResult, SourceAttribution, QueryResponse, Document, 
    ContentSource, DocumentType
)
from src.feature_pipeline.vector_storage import MongoVectorStore
from src.utils.logger import LoggerMixin


class SourceAttributionService(LoggerMixin):
    """Service for enriching search results with source attribution."""
    
    def __init__(self, vector_store: Optional[MongoVectorStore] = None):
        """Initialize the source attribution service."""
        self.vector_store = vector_store
        self._document_cache: Dict[str, Document] = {}
    
    async def enrich_search_results(
        self, 
        search_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Enrich search results with detailed source attribution.
        
        Args:
            search_results: List of basic search results
            
        Returns:
            List of enriched search results with source attribution
        """
        enriched_results = []
        
        for result in search_results:
            try:
                # Get document information for this chunk
                document_id = result.metadata.get('document_id')
                if not document_id:
                    enriched_results.append(result)
                    continue
                
                # Get document details
                document = await self._get_document_info(document_id)
                if not document:
                    enriched_results.append(result)
                    continue
                
                # Create rich source attribution
                source_attribution = self._create_source_attribution(
                    document=document,
                    search_result=result
                )
                
                # Enrich the search result
                result.source = source_attribution
                enriched_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to enrich search result {result.id}: {e}")
                enriched_results.append(result)  # Include original result
        
        self.logger.info(f"Enriched {len(enriched_results)} search results with source attribution")
        return enriched_results
    
    def _create_source_attribution(
        self, 
        document: Document, 
        search_result: SearchResult
    ) -> SourceAttribution:
        """
        Create detailed source attribution for a search result.
        
        Args:
            document: Source document
            search_result: Search result to enrich
            
        Returns:
            Rich source attribution object
        """
        # Extract chunk type from metadata
        chunk_type = search_result.metadata.get('chunk_type')
        
        # Extract strategies used
        strategies_used = search_result.metadata.get('strategies', [])
        if isinstance(strategies_used, str):
            strategies_used = [strategies_used]
        
        return SourceAttribution(
            title=document.title,
            url=document.source_url,
            source_type=document.source,
            document_type=document.document_type,
            chunk_type=chunk_type,
            strategies_used=strategies_used,
            created_at=document.created_at
        )
    
    async def _get_document_info(self, document_id: str) -> Optional[Document]:
        """
        Get document information by ID with caching.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document object if found, None otherwise
        """
        # Check cache first
        if document_id in self._document_cache:
            return self._document_cache[document_id]
        
        # Query database using document repository
        if self.vector_store and hasattr(self.vector_store, 'document_repo'):
            try:
                document = await self.vector_store.document_repo.get_document(document_id)
                
                if document:
                    # Cache for future use
                    self._document_cache[document_id] = document
                    return document
                    
            except Exception as e:
                self.logger.error(f"Error fetching document {document_id}: {e}")
        
        return None
    
    def create_response_with_sources(
        self,
        response_text: str,
        search_results: List[SearchResult],
        **kwargs
    ) -> QueryResponse:
        """
        Create a complete response with source attribution.
        
        Args:
            response_text: Generated response text
            search_results: Search results that contributed to response
            **kwargs: Additional metadata
            
        Returns:
            Complete query response with sources
        """
        # Extract unique sources from search results
        sources = []
        seen_sources = set()
        
        for result in search_results:
            if result.source and result.source.title not in seen_sources:
                sources.append(result.source)
                seen_sources.add(result.source.title)
        
        # Sort sources by relevance (score)
        if search_results:
            score_map = {r.source.title: r.score for r in search_results if r.source}
            sources.sort(key=lambda s: score_map.get(s.title, 0.0), reverse=True)
        
        # Calculate confidence based on source quality and relevance
        confidence_score = self._calculate_confidence(search_results)
        
        return QueryResponse(
            response=response_text,
            sources=sources,
            confidence_score=confidence_score,
            processing_time_ms=kwargs.get('processing_time_ms'),
            search_strategy=kwargs.get('search_strategy'),
            model_used=kwargs.get('model_used'),
            metadata=kwargs.get('metadata', {})
        )
    
    def _calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """
        Calculate confidence score based on search results quality.
        
        Args:
            search_results: List of search results
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not search_results:
            return 0.0
        
        # Base confidence on top result's score
        top_score = max(result.score for result in search_results)
        
        # Adjust based on number of sources
        source_count_factor = min(len(search_results) / 3.0, 1.0)  # Optimal: 3+ sources
        
        # Adjust based on source diversity
        source_types = set()
        for result in search_results:
            if result.source:
                source_types.add(result.source.source_type)
        
        diversity_factor = min(len(source_types) / 2.0, 1.0)  # Optimal: 2+ source types
        
        # Combined confidence
        confidence = top_score * 0.6 + source_count_factor * 0.2 + diversity_factor * 0.2
        
        return min(confidence, 1.0)
    
    def format_sources_for_display(
        self, 
        sources: List[SourceAttribution], 
        max_sources: int = 5
    ) -> str:
        """
        Format sources for user-friendly display.
        
        Args:
            sources: List of source attributions
            max_sources: Maximum number of sources to display
            
        Returns:
            Formatted source string
        """
        if not sources:
            return "No sources available."
        
        display_sources = sources[:max_sources]
        formatted_sources = []
        
        for i, source in enumerate(display_sources, 1):
            # Format source entry
            source_line = f"{i}. **{source.title}**"
            
            # Add URL if available
            if source.url:
                if source.source_type == ContentSource.NOTION:
                    source_line += f" ([Notion Page]({source.url}))"
                elif source.source_type == ContentSource.WEB_CRAWL:
                    source_line += f" ([Web Link]({source.url}))"
                else:
                    source_line += f" ([Source]({source.url}))"
            
            # Add source type info
            type_info = []
            if source.document_type:
                type_info.append(source.document_type.value.replace('_', ' ').title())
            if source.chunk_type:
                type_info.append(f"{source.chunk_type} chunk")
            if source.strategies_used:
                type_info.append(f"via {', '.join(source.strategies_used)}")
            
            if type_info:
                source_line += f" • {' • '.join(type_info)}"
            
            formatted_sources.append(source_line)
        
        # Add "and X more" if there are additional sources
        if len(sources) > max_sources:
            additional_count = len(sources) - max_sources
            formatted_sources.append(f"... and {additional_count} more source{'s' if additional_count > 1 else ''}")
        
        return "\n".join(formatted_sources)
    
    def get_source_statistics(self, sources: List[SourceAttribution]) -> Dict[str, Any]:
        """
        Get statistics about sources for analytics.
        
        Args:
            sources: List of source attributions
            
        Returns:
            Dictionary with source statistics
        """
        if not sources:
            return {}
        
        stats = {
            "total_sources": len(sources),
            "source_types": {},
            "document_types": {},
            "strategies_used": {},
            "has_urls": sum(1 for s in sources if s.url),
            "notion_sources": sum(1 for s in sources if s.source_type == ContentSource.NOTION),
            "web_sources": sum(1 for s in sources if s.source_type == ContentSource.WEB_CRAWL),
        }
        
        # Count by source type
        for source in sources:
            source_type = source.source_type.value
            stats["source_types"][source_type] = stats["source_types"].get(source_type, 0) + 1
            
            doc_type = source.document_type.value
            stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
            
            for strategy in source.strategies_used:
                stats["strategies_used"][strategy] = stats["strategies_used"].get(strategy, 0) + 1
        
        return stats 