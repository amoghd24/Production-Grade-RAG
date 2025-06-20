"""
Advanced RAG Engine for Second Brain AI Assistant.
Implements Retrieval Augmented Generation with advanced search strategies,
contextual enhancement, parent-child retrieval, and source attribution.
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from src.inference_pipeline.openai_service import OpenAIService
from src.inference_pipeline.prompt_manager import PromptManager
from src.feature_pipeline.vector_storage import MongoVectorStore, create_vector_store
from src.feature_pipeline.advanced_search import AdvancedSearchOrchestrator
from src.feature_pipeline.source_attribution import SourceAttributionService
from src.models.schemas import SearchResult, QueryResponse, SourceAttribution
from src.config.feature_flags import get_feature_flags
from src.utils.logger import LoggerMixin


class RAGEngine(LoggerMixin):
    """Advanced RAG Engine with integrated search strategies and source attribution."""
    
    def __init__(self):
        """Initialize the advanced RAG engine."""
        self.openai_service = OpenAIService()
        self.prompt_manager = PromptManager()
        self.vector_store = None  # Will be initialized in setup
        
        # Advanced components
        self.search_orchestrator = None  # Will be initialized in setup
        self.source_attribution = None  # Will be initialized in setup
        self.feature_flags = get_feature_flags()
        
        # Configuration
        self.similarity_threshold = 0.7
        self.max_tokens_per_doc = 10000  # Increased for GPT-4
        self.max_total_tokens = 128000   # GPT-4's context window
        
        # Conversation management
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_conversation_turns = 10
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "avg_confidence": 0.0,
            "sources_used": 0,
            "advanced_features_used": 0
        }
    
    async def setup(self):
        """Initialize the vector store and advanced components."""
        self.vector_store = await create_vector_store()
        self.source_attribution = SourceAttributionService(self.vector_store)
        self.search_orchestrator = AdvancedSearchOrchestrator(self.vector_store)
        self.logger.info("Advanced RAG Engine initialized with all components")
    
    async def process_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        include_conversation_context: bool = True,
        similarity_threshold: Optional[float] = None
    ) -> QueryResponse:
        """
        Process a user query using Advanced RAG with full orchestration.
        
        Args:
            query: User's query
            conversation_id: Optional conversation identifier for context
            include_conversation_context: Whether to include conversation history
            similarity_threshold: Minimum similarity score for context documents
            
        Returns:
            QueryResponse with rich source attribution and metadata
        """
        start_time = time.time()
        
        try:
            # Ensure all components are initialized
            if self.vector_store is None or self.source_attribution is None:
                await self.setup()
            
            # Update metrics
            self.metrics["total_queries"] += 1
            
            # Prepare conversation context if needed
            conversation_context = self._get_conversation_context(
                conversation_id, include_conversation_context
            )
            
            # Enhanced query processing with conversation context
            enhanced_query = self._enhance_query_with_context(query, conversation_context)
            
            # Use advanced search orchestrator for retrieval
            search_results = await self._advanced_retrieve_context(
                enhanced_query,
                threshold=similarity_threshold or self.similarity_threshold
            )
            
            # Enrich search results with source attribution
            enriched_results = await self.source_attribution.enrich_search_results(search_results)
            
            # Generate advanced prompt with rich context
            prompt = await self._create_advanced_prompt(
                query, enriched_results, conversation_context
            )
            
            # Generate response with context
            response_data = self.openai_service.generate_response(
                prompt=prompt,
                context=self._format_context_for_generation(enriched_results),
                system_message=self.prompt_manager.get_advanced_system_message()
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Create comprehensive response with source attribution
            query_response = self.source_attribution.create_response_with_sources(
                response_text=response_data["response"],
                search_results=enriched_results,
                processing_time_ms=processing_time,
                search_strategy=self._get_search_strategy_used(),
                model_used=response_data["model"],
                metadata={
                    "conversation_id": conversation_id,
                    "enhanced_query": enhanced_query,
                    "usage": response_data.get("usage", {}),
                    "feature_flags_used": self._get_features_used()
                }
            )
            
            # Update conversation history
            self._update_conversation_history(
                conversation_id, query, query_response, enriched_results
            )
            
            # Update performance metrics
            self._update_metrics(processing_time, query_response, enriched_results)
            
            self.logger.info(
                f"Advanced RAG processed query in {processing_time:.1f}ms with "
                f"{len(enriched_results)} sources (confidence: {query_response.confidence_score:.2f})"
            )
            
            return query_response
            
        except Exception as e:
            self.logger.error(f"Error in advanced RAG processing: {str(e)}")
            # Return error response instead of raising
            return QueryResponse(
                response=f"I encountered an error while processing your query: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                search_strategy="error_fallback",
                metadata={"error": str(e)}
            )
    
    async def _advanced_retrieve_context(
        self,
        query: str,
        threshold: float
    ) -> List[SearchResult]:
        """
        Retrieve context using advanced search orchestrator.
        
        Args:
            query: Enhanced user query
            threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects with rich metadata
        """
        try:
            # Use advanced search orchestrator if features enabled
            if self.feature_flags.should_use_advanced_search() and self.search_orchestrator:
                search_results = await self.search_orchestrator.search(
                    query=query,
                    max_results=10,
                    threshold=threshold
                )
                self.metrics["advanced_features_used"] += 1
            else:
                # Fallback to basic similarity search
                search_results = await self._basic_similarity_search(query, threshold)
            
            # Filter and process results
            filtered_results = self._filter_and_optimize_results(search_results)
            
            self.logger.info(
                f"Advanced retrieval found {len(filtered_results)} relevant sources "
                f"(advanced features: {self.feature_flags.should_use_advanced_search()})"
            )
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error in advanced retrieval: {str(e)}")
            # Fallback to basic search on error
            return await self._basic_similarity_search(query, threshold)
    
    async def _basic_similarity_search(
        self, 
        query: str, 
        threshold: float
    ) -> List[SearchResult]:
        """Fallback basic similarity search."""
        try:
            from sentence_transformers import SentenceTransformer
            from src.config.settings import settings
            
            embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            query_embedding = embedding_model.encode([query])[0].tolist()
            
            results = await self.vector_store.vector_search_service.similarity_search_with_score(
                query_vector=query_embedding,
                limit=10,
                score_threshold=threshold
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Basic similarity search failed: {str(e)}")
            return []
    
    def _filter_and_optimize_results(
        self, 
        search_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Filter and optimize search results for token limits."""
        if not search_results:
            return []
        
        filtered_results = []
        total_tokens = 0
        
        for result in search_results:
            # Calculate approximate tokens
            doc_tokens = len(result.content) // 4
            
            # Skip if would exceed total token limit
            if total_tokens + doc_tokens > self.max_total_tokens:
                self.logger.debug(f"Skipping result due to token limit")
                continue
            
            # Truncate if individual document too long
            if doc_tokens > self.max_tokens_per_doc:
                result.content = result.content[:self.max_tokens_per_doc * 4] + "..."
                doc_tokens = self.max_tokens_per_doc
            
            filtered_results.append(result)
            total_tokens += doc_tokens
        
        return filtered_results
    
    def _get_conversation_context(
        self, 
        conversation_id: Optional[str], 
        include_context: bool
    ) -> Optional[str]:
        """Get relevant conversation context if available."""
        if not include_context or not conversation_id:
            return None
        
        # Find conversation history for this ID
        relevant_history = [
            entry for entry in self.conversation_history[-self.max_conversation_turns:]
            if entry.get("conversation_id") == conversation_id
        ]
        
        if not relevant_history:
            return None
        
        # Format recent conversation turns
        context_parts = []
        for entry in relevant_history[-3:]:  # Last 3 turns max
            context_parts.append(f"Previous Q: {entry['query']}")
            context_parts.append(f"Previous A: {entry['response'][:200]}...")  # Truncate
        
        return "\n".join(context_parts) if context_parts else None
    
    def _enhance_query_with_context(
        self, 
        query: str, 
        conversation_context: Optional[str]
    ) -> str:
        """Enhance query with conversation context if available."""
        if conversation_context:
            return f"Context from previous conversation:\n{conversation_context}\n\nCurrent question: {query}"
        return query
    
    async def _create_advanced_prompt(
        self,
        query: str,
        search_results: List[SearchResult],
        conversation_context: Optional[str]
    ) -> str:
        """Create advanced prompt with rich context and source attribution."""
        if not search_results:
            return self.prompt_manager.create_no_context_prompt(query)
        
        # Group results by source type for better organization
        notion_sources = [r for r in search_results if r.source and r.source.source_type.value == "notion"]
        web_sources = [r for r in search_results if r.source and r.source.source_type.value == "web_crawl"]
        other_sources = [r for r in search_results if r not in notion_sources and r not in web_sources]
        
        # Create structured context
        context_sections = []
        
        if notion_sources:
            context_sections.append(f"=== From Your Notion Workspace ({len(notion_sources)} sources) ===")
            for result in notion_sources:
                title = result.source.title if result.source else "Unknown"
                context_sections.append(f"Source: {title}")
                context_sections.append(f"Content: {result.content}")
                context_sections.append("")
        
        if web_sources:
            context_sections.append(f"=== From Web Sources ({len(web_sources)} sources) ===")
            for result in web_sources:
                title = result.source.title if result.source else "Unknown"
                url = result.source.url if result.source else "No URL"
                context_sections.append(f"Source: {title} ({url})")
                context_sections.append(f"Content: {result.content}")
                context_sections.append("")
        
        if other_sources:
            context_sections.append(f"=== From Other Sources ({len(other_sources)} sources) ===")
            for result in other_sources:
                title = result.source.title if result.source else "Unknown"
                context_sections.append(f"Source: {title}")
                context_sections.append(f"Content: {result.content}")
                context_sections.append("")
        
        structured_context = "\n".join(context_sections)
        
        # Use advanced prompt template
        return self.prompt_manager.create_advanced_rag_prompt(
            query=query,
            structured_context=structured_context,
            conversation_context=conversation_context,
            source_count=len(search_results)
        )
    
    def _format_context_for_generation(
        self, 
        search_results: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        """Format search results for OpenAI service."""
        return [
            {
                "title": result.source.title if result.source else "Unknown",
                "content": result.content,
                "score": result.score,
                "url": str(result.source.url) if result.source and result.source.url else None
            }
            for result in search_results
        ]
    
    def _get_search_strategy_used(self) -> str:
        """Get description of search strategy used."""
        if self.feature_flags.should_use_advanced_search():
            strategies = []
            if self.feature_flags.should_use_contextual_chunking():
                strategies.append("contextual")
            if self.feature_flags.should_use_parent_retrieval():
                strategies.append("parent-child")
            if self.feature_flags.should_use_hybrid_search():
                strategies.append("hybrid")
            
            if strategies:
                return f"multi-strategy ({', '.join(strategies)})"
            return "advanced-similarity"
        return "basic-similarity"
    
    def _get_features_used(self) -> List[str]:
        """Get list of advanced features that were used."""
        features = []
        if self.feature_flags.should_use_advanced_search():
            features.append("advanced_search")
        if self.feature_flags.should_use_contextual_chunking():
            features.append("contextual_chunking")
        if self.feature_flags.should_use_parent_retrieval():
            features.append("parent_retrieval")
        if self.feature_flags.should_use_hybrid_search():
            features.append("hybrid_search")
        return features
    
    def _update_conversation_history(
        self,
        conversation_id: Optional[str],
        query: str,
        response: QueryResponse,
        search_results: List[SearchResult]
    ):
        """Update conversation history for future context."""
        if conversation_id:
            entry = {
                "conversation_id": conversation_id,
                "timestamp": datetime.utcnow(),
                "query": query,
                "response": response.response,
                "source_count": len(search_results),
                "confidence": response.confidence_score
            }
            
            self.conversation_history.append(entry)
            
            # Keep only recent history
            if len(self.conversation_history) > 100:
                self.conversation_history = self.conversation_history[-50:]
    
    def _update_metrics(
        self,
        processing_time: float,
        response: QueryResponse,
        search_results: List[SearchResult]
    ):
        """Update performance metrics."""
        # Update rolling averages
        query_count = self.metrics["total_queries"]
        
        # Response time average
        current_avg_time = self.metrics["avg_response_time"]
        self.metrics["avg_response_time"] = (
            (current_avg_time * (query_count - 1) + processing_time) / query_count
        )
        
        # Confidence average
        current_avg_confidence = self.metrics["avg_confidence"]
        self.metrics["avg_confidence"] = (
            (current_avg_confidence * (query_count - 1) + (response.confidence_score or 0.0)) / query_count
        )
        
        # Source usage
        self.metrics["sources_used"] += len(search_results)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.metrics,
            "avg_sources_per_query": (
                self.metrics["sources_used"] / self.metrics["total_queries"] 
                if self.metrics["total_queries"] > 0 else 0
            ),
            "advanced_features_usage_rate": (
                self.metrics["advanced_features_used"] / self.metrics["total_queries"]
                if self.metrics["total_queries"] > 0 else 0
            ),
            "conversation_history_size": len(self.conversation_history),
            "feature_flags_status": {
                "advanced_search": self.feature_flags.should_use_advanced_search(),
                "contextual": self.feature_flags.should_use_contextual_chunking(),
                "parent_child": self.feature_flags.should_use_parent_retrieval(),
                "hybrid": self.feature_flags.should_use_hybrid_search(),
            }
        }
    
    async def process_conversation_query(
        self,
        query: str,
        conversation_id: str,
        user_id: Optional[str] = None
    ) -> QueryResponse:
        """
        Process a query within a conversation context.
        
        Args:
            query: User's query
            conversation_id: Conversation identifier
            user_id: Optional user identifier for personalization
            
        Returns:
            QueryResponse with conversation context
        """
        return await self.process_query(
            query=query,
            conversation_id=conversation_id,
            include_conversation_context=True
        ) 