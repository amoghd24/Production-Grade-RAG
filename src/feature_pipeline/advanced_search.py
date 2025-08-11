"""
Advanced Search Orchestrator for Advanced RAG.
Coordinates multiple search strategies and fuses results for optimal retrieval.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.models.schemas import DocumentChunk, SearchResult

from src.config.feature_flags import get_feature_flags
from src.utils.logger import LoggerMixin


class SearchStrategy(str, Enum):
    """Available search strategies."""
    SIMILARITY = "similarity"
    PARENT_CHILD = "parent_child" 
    HYBRID = "hybrid"
    MULTI_VECTOR = "multi_vector"


@dataclass
class SearchConfig:
    """Configuration for search strategies."""
    strategy: SearchStrategy
    weight: float = 1.0
    max_results: int = 10
    threshold: float = 0.0
    enabled: bool = True


@dataclass 
class SearchContext:
    """Context information for search execution."""
    query: str
    original_query: str
    query_intent: str
    expanded_queries: List[str]
    metadata: Dict[str, Any]


class AdvancedSearchOrchestrator(LoggerMixin):
    """Orchestrates multiple search strategies and fuses results."""
    
    def __init__(self, vector_store=None):
        """Initialize the advanced search orchestrator."""
        self.feature_flags = get_feature_flags()
        self.vector_store = vector_store
        
        # Default search strategy configurations
        self.search_configs = {
            SearchStrategy.SIMILARITY: SearchConfig(
                strategy=SearchStrategy.SIMILARITY,
                weight=0.6,
                max_results=15,
                threshold=0.7
            ),
            SearchStrategy.PARENT_CHILD: SearchConfig(
                strategy=SearchStrategy.PARENT_CHILD,
                weight=0.4,
                max_results=8,
                threshold=0.65
            ),
        }
    
    async def search(self, query: str, max_results: int = 10, **kwargs) -> List[SearchResult]:
        """
        Execute advanced search using multiple strategies.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of fused and ranked search results
        """
        if not self.feature_flags.should_use_advanced_search() or not self.vector_store:
            # Fallback to empty results if no vector store
            return []
        
        # Prepare search context
        context = self._prepare_search_context(query)
        
        # Execute multiple search strategies
        strategy_results = await self._execute_search_strategies(context, max_results)
        
        # Fuse results from different strategies
        fused_results = self._fuse_results(strategy_results, max_results)
        
        # Apply final reranking if available
        final_results = self._apply_final_ranking(fused_results, context)
        
        self.logger.info(f"Advanced search returned {len(final_results)} results for query: '{query}'")
        return final_results[:max_results]
    
    def _prepare_search_context(self, query: str) -> SearchContext:
        """Prepare search context with basic query processing."""
        # Basic query cleaning - remove extra whitespace
        processed_query = ' '.join(query.split()).strip()
        
        # Simple intent detection based on common patterns
        intent = self._get_simple_query_intent(processed_query)
        
        # Use original query without expansion
        expanded_queries = [processed_query]
        
        context = SearchContext(
            query=processed_query,
            original_query=query,
            query_intent=intent,
            expanded_queries=expanded_queries,
            metadata={
                'timestamp': None,  # Would be set in real implementation
                'user_context': None,
                'session_id': None
            }
        )
        
        self.logger.debug(f"Search context prepared: intent={intent}, query='{processed_query}'")
        return context
    
    def _get_simple_query_intent(self, query: str) -> str:
        """Simple query intent detection without complex expansion logic."""
        query_lower = query.lower()
        
        # Technical queries
        if any(term in query_lower for term in ['api', 'code', 'function', 'class', 'implement']):
            return 'technical'
        
        # How-to queries
        elif any(term in query_lower for term in ['how to', 'how do', 'steps', 'tutorial']):
            return 'how-to'
        
        # Problem-solving queries
        elif any(term in query_lower for term in ['error', 'problem', 'issue', 'fix', 'debug']):
            return 'problem-solving'
        
        # Conceptual queries
        elif any(term in query_lower for term in ['what is', 'explain', 'concept', 'theory']):
            return 'conceptual'
        
        # Default
        return 'general'
    
    async def _execute_search_strategies(self, context: SearchContext, max_results: int) -> Dict[SearchStrategy, List[SearchResult]]:
        """Execute multiple search strategies based on query context."""
        strategy_results = {}
        
        # Determine which strategies to use based on query intent
        active_strategies = self._select_strategies(context)
        
        for strategy in active_strategies:
            if not self._is_strategy_available(strategy):
                continue
            
            try:
                # Execute each strategy (this would call actual search implementations)
                results = await self._execute_single_strategy(strategy, context, max_results)
                strategy_results[strategy] = results
                
                self.logger.debug(f"Strategy {strategy} returned {len(results)} results")
                
            except Exception as e:
                self.logger.warning(f"Strategy {strategy} failed: {e}")
                continue
        
        return strategy_results
    
    def _select_strategies(self, context: SearchContext) -> List[SearchStrategy]:
        """Select appropriate search strategies based on query context."""
        intent = context.query_intent
        strategies = []
        
        # Always include similarity search as baseline
        strategies.append(SearchStrategy.SIMILARITY)
        
        # Add parent-child search for conceptual queries (replacing contextual)
        if intent in ['conceptual', 'general'] and self.feature_flags.should_use_parent_retrieval():
            strategies.append(SearchStrategy.PARENT_CHILD)
        
        # Add parent-child search for technical/detailed queries
        if intent in ['technical', 'how-to'] and self.feature_flags.should_use_parent_retrieval():
            strategies.append(SearchStrategy.PARENT_CHILD)
        
        # Add hybrid search if enabled
        if self.feature_flags.should_use_hybrid_search():
            strategies.append(SearchStrategy.HYBRID)
        
        return strategies
    
    def _is_strategy_available(self, strategy: SearchStrategy) -> bool:
        """Check if a search strategy is available and enabled."""
        config = self.search_configs.get(strategy)
        if not config or not config.enabled:
            return False
        
        # Check feature flags
        if strategy == SearchStrategy.PARENT_CHILD:
            return self.feature_flags.should_use_parent_retrieval()
        elif strategy == SearchStrategy.HYBRID:
            return self.feature_flags.should_use_hybrid_search()
        
        return True
    
    async def _execute_single_strategy(self, strategy: SearchStrategy, context: SearchContext, max_results: int) -> List[SearchResult]:
        """Execute a single search strategy using real vector store."""
        results = []
        config = self.search_configs[strategy]
        
        try:
            # Use real vector store search instead of mocks
            if strategy == SearchStrategy.SIMILARITY:
                results = await self._real_similarity_search(context, config.max_results)
            elif strategy == SearchStrategy.PARENT_CHILD:
                results = await self._real_parent_child_search(context, config.max_results)
        except Exception as e:
            self.logger.error(f"Strategy {strategy} failed: {str(e)}")
            results = []
        
        # Filter by threshold
        filtered_results = [r for r in results if r.score >= config.threshold]
        
        return filtered_results[:max_results]
    
    def _fuse_results(self, strategy_results: Dict[SearchStrategy, List[SearchResult]], max_results: int) -> List[SearchResult]:
        """Fuse results from multiple search strategies using weighted scoring."""
        if not strategy_results:
            return []
        
        # Collect all unique results
        all_results = {}
        
        for strategy, results in strategy_results.items():
            config = self.search_configs[strategy]
            weight = config.weight
            
            for result in results:
                result_id = result.id
                
                if result_id in all_results:
                    # Combine scores using weighted average
                    existing = all_results[result_id]
                    existing_weight = existing.metadata.get('total_weight', 1.0)
                    new_weight = existing_weight + weight
                    
                    # Weighted score combination
                    combined_score = (existing.score * existing_weight + result.score * weight) / new_weight
                    
                    existing.score = combined_score
                    existing.metadata['total_weight'] = new_weight
                    existing.metadata['strategies'] = existing.metadata.get('strategies', []) + [strategy.value]
                else:
                    # New result
                    result.metadata['total_weight'] = weight
                    result.metadata['strategies'] = [strategy.value]
                    all_results[result_id] = result
        
        # Sort by combined score
        fused_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        
        self.logger.debug(f"Fused {len(fused_results)} unique results from {len(strategy_results)} strategies")
        return fused_results[:max_results]
    
    def _apply_final_ranking(self, results: List[SearchResult], context: SearchContext) -> List[SearchResult]:
        """Apply final ranking/reranking to the fused results."""
        # This is where you would apply cross-encoder reranking or other
        # sophisticated ranking algorithms
        
        # For now, just return the results with potential diversity adjustment
        return self._apply_diversity_filter(results, context)
    
    def _apply_diversity_filter(self, results: List[SearchResult], context: SearchContext) -> List[SearchResult]:
        """Apply diversity filtering to avoid too similar results."""
        if len(results) <= 3:
            return results
        
        # Simple diversity filter based on content similarity
        diverse_results = [results[0]]  # Always include top result
        
        for result in results[1:]:
            # Check if result is sufficiently different from already selected ones
            is_diverse = True
            for selected in diverse_results:
                # Simple diversity check (in practice, would use embedding similarity)
                if self._calculate_content_similarity(result.content, selected.content) > 0.85:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity (placeholder implementation)."""
        # Simple Jaccard similarity for demonstration
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _real_similarity_search(self, context: SearchContext, max_results: int) -> List[SearchResult]:
        """Real similarity search using vector store."""
        if not self.vector_store:
            return []
        
        try:
            # Generate embedding for the query
            from sentence_transformers import SentenceTransformer
            from src.config.settings import settings
            
            embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            query_embedding = embedding_model.encode([context.query])[0].tolist()
            
            # Perform similarity search
            results = await self.vector_store.vector_search_service.similarity_search(
                query_vector=query_embedding,
                limit=max_results,
                filters=None
            )
            
            return results
        except Exception as e:
            self.logger.error(f"Real similarity search failed: {str(e)}")
            return []
    
    async def _real_parent_child_search(self, context: SearchContext, max_results: int) -> List[SearchResult]:
        """Real parent-child search with higher score threshold."""
        if not self.vector_store:
            return []
        
        try:
            from sentence_transformers import SentenceTransformer
            from src.config.settings import settings
            
            embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            query_embedding = embedding_model.encode([context.query])[0].tolist()
            
            # Use filtered search with higher threshold for parent-child
            results = await self.vector_store.vector_search_service.similarity_search_with_score(
                query_vector=query_embedding,
                limit=max_results,
                score_threshold=0.75,  # Higher threshold for parent-child
                filters=None
            )
            
            return results
        except Exception as e:
            self.logger.error(f"Real parent-child search failed: {str(e)}")
            return []
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search performance statistics."""
        return {
            "strategies_available": len([s for s in self.search_configs.values() if s.enabled]),
            "advanced_features_enabled": self.feature_flags.should_use_advanced_search(),
            "query_expansion_enabled": False,
            "reranking_enabled": False  # Would be True if reranking model is available
        } 