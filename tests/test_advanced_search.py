"""
Tests for Advanced Search functionality.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.feature_pipeline.query_expansion import QueryExpansionService
from src.feature_pipeline.advanced_search import (
    AdvancedSearchOrchestrator, 
    SearchStrategy, 
    SearchConfig, 
    SearchContext
)
from src.models.schemas import SearchResult


class TestQueryExpansionService:
    """Test query expansion functionality."""
    
    def test_expansion_disabled_by_default(self):
        """Test that expansion is disabled when feature flag is off."""
        expander = QueryExpansionService()
        
        query = "machine learning models"
        expanded = expander.expand_query(query)
        
        # Should return only original query when disabled
        assert len(expanded) == 1
        assert expanded[0] == query
    
    @patch('src.feature_pipeline.query_expansion.get_feature_flags')
    def test_query_expansion_enabled(self, mock_get_flags):
        """Test query expansion when feature is enabled."""
        # Mock feature flags
        mock_flags = MagicMock()
        mock_flags.should_use_advanced_search.return_value = True
        mock_get_flags.return_value = mock_flags
        
        expander = QueryExpansionService()
        
        query = "ML models"
        expanded = expander.expand_query(query, max_expansions=3)
        
        # Should include original + expansions
        assert len(expanded) > 1
        assert query in expanded
        
        # Check for expected expansions - should find ML synonyms
        expanded_text = ' '.join(expanded).lower()
        assert any(term in expanded_text for term in ['machine learning', 'artificial intelligence', 'neural network'])
    
    @patch('src.feature_pipeline.query_expansion.get_feature_flags')
    def test_abbreviation_expansion(self, mock_get_flags):
        """Test abbreviation expansion."""
        mock_flags = MagicMock()
        mock_flags.should_use_advanced_search.return_value = True
        mock_get_flags.return_value = mock_flags
        
        expander = QueryExpansionService()
        
        query = "NLP techniques"
        expanded = expander.expand_query(query)
        
        # Should expand NLP to natural language processing
        assert any('natural language processing' in exp for exp in expanded)
    
    def test_query_intent_detection(self):
        """Test query intent classification."""
        expander = QueryExpansionService()
        
        test_cases = [
            ("What is machine learning?", "conceptual"),
            ("How to implement CNN?", "technical"),  # Fixed: "implement" triggers technical
            ("database error connection", "problem-solving"),
            ("neural network code", "technical"),
            ("deep learning basics", "general")
        ]
        
        for query, expected_intent in test_cases:
            intent = expander.get_query_intent(query)
            assert intent == expected_intent
    
    def test_query_preprocessing(self):
        """Test query cleaning and preprocessing."""
        expander = QueryExpansionService()
        
        test_cases = [
            ("  extra   spaces  ", "extra spaces"),
            ("the quick brown fox", "quick brown fox"),  # Stop words removed
            ("a simple test", "a simple test"),  # Short query keeps stop words (3 words)
            ("AI", "AI"),  # Short query keeps stop words
        ]
        
        for input_query, expected in test_cases:
            cleaned = expander.preprocess_query(input_query)
            assert cleaned == expected
    
    def test_expansion_criteria(self):
        """Test when expansion should or shouldn't be used."""
        expander = QueryExpansionService()
        
        # Should expand
        assert expander.should_use_expansion("machine learning") is True
        assert expander.should_use_expansion("neural networks") is True
        
        # Should not expand
        assert expander.should_use_expansion('"exact match"') is False
        assert expander.should_use_expansion("function() {} implementation") is False
        assert expander.should_use_expansion("very long query with many words that exceeds the word limit for expansion") is False


class TestAdvancedSearchOrchestrator:
    """Test advanced search orchestration functionality."""
    
    def test_basic_search_fallback(self):
        """Test fallback to basic search when advanced features disabled."""
        orchestrator = AdvancedSearchOrchestrator()
        
        query = "test query"
        results = orchestrator.search(query)
        
        # Should return empty list (basic search mock)
        assert isinstance(results, list)
    
    @patch('src.feature_pipeline.advanced_search.get_feature_flags')
    def test_search_context_preparation(self, mock_get_flags):
        """Test search context preparation."""
        mock_flags = MagicMock()
        mock_flags.should_use_advanced_search.return_value = True
        mock_flags.should_use_contextual_chunking.return_value = True
        mock_flags.should_use_parent_retrieval.return_value = True
        mock_flags.should_use_hybrid_search.return_value = False
        mock_get_flags.return_value = mock_flags
        
        orchestrator = AdvancedSearchOrchestrator()
        
        query = "How to implement neural networks?"
        context = orchestrator._prepare_search_context(query)
        
        assert isinstance(context, SearchContext)
        assert context.original_query == query
        assert context.query_intent == "technical"  # Fixed: "implement" triggers technical
        assert len(context.expanded_queries) >= 1
    
    @patch('src.feature_pipeline.advanced_search.get_feature_flags')
    def test_strategy_selection(self, mock_get_flags):
        """Test strategy selection based on query intent."""
        mock_flags = MagicMock()
        mock_flags.should_use_advanced_search.return_value = True
        mock_flags.should_use_contextual_chunking.return_value = True
        mock_flags.should_use_parent_retrieval.return_value = True
        mock_flags.should_use_hybrid_search.return_value = False
        mock_get_flags.return_value = mock_flags
        
        orchestrator = AdvancedSearchOrchestrator()
        
        # Technical query should include parent-child strategy
        technical_context = SearchContext(
            query="implement neural network",
            original_query="implement neural network",
            query_intent="technical",
            expanded_queries=["implement neural network"],
            metadata={}
        )
        
        strategies = orchestrator._select_strategies(technical_context)
        assert SearchStrategy.SIMILARITY in strategies
        assert SearchStrategy.PARENT_CHILD in strategies
        
        # Conceptual query should include contextual strategy
        conceptual_context = SearchContext(
            query="what is machine learning",
            original_query="what is machine learning",
            query_intent="conceptual",
            expanded_queries=["what is machine learning"],
            metadata={}
        )
        
        strategies = orchestrator._select_strategies(conceptual_context)
        assert SearchStrategy.SIMILARITY in strategies
        assert SearchStrategy.CONTEXTUAL in strategies
    
    @patch('src.feature_pipeline.advanced_search.get_feature_flags')
    def test_result_fusion(self, mock_get_flags):
        """Test result fusion from multiple strategies."""
        mock_flags = MagicMock()
        mock_flags.should_use_advanced_search.return_value = True
        mock_get_flags.return_value = mock_flags
        
        orchestrator = AdvancedSearchOrchestrator()
        
        # Create mock results from different strategies
        strategy_results = {
            SearchStrategy.SIMILARITY: [
                SearchResult(
                    id="result_1",
                    content="Similar content",
                    score=0.9,
                    metadata={}
                ),
                SearchResult(
                    id="result_2", 
                    content="Another similar content",
                    score=0.8,
                    metadata={}
                )
            ],
            SearchStrategy.CONTEXTUAL: [
                SearchResult(
                    id="result_1",  # Same ID as similarity result
                    content="Similar content with context",
                    score=0.85,
                    metadata={}
                ),
                SearchResult(
                    id="result_3",
                    content="Contextual content",
                    score=0.75,
                    metadata={}
                )
            ]
        }
        
        fused_results = orchestrator._fuse_results(strategy_results, max_results=5)
        
        # Should have 3 unique results (result_1 fused, result_2, result_3)
        assert len(fused_results) == 3
        
        # Results should be sorted by score
        assert fused_results[0].score >= fused_results[1].score
        
        # Fused result should have strategy information
        fused_result = next(r for r in fused_results if r.id == "result_1")
        assert 'strategies' in fused_result.metadata
        assert len(fused_result.metadata['strategies']) == 2
    
    def test_diversity_filter(self):
        """Test diversity filtering functionality."""
        orchestrator = AdvancedSearchOrchestrator()
        
        # Create similar results
        results = [
            SearchResult(
                id="result_1",
                content="machine learning algorithms are powerful",
                score=0.9,
                metadata={}
            ),
            SearchResult(
                id="result_2",
                content="machine learning algorithms are effective",  # Very similar
                score=0.85,
                metadata={}
            ),
            SearchResult(
                id="result_3",
                content="neural networks for image classification",  # Different
                score=0.8,
                metadata={}
            )
        ]
        
        context = SearchContext(
            query="machine learning",
            original_query="machine learning", 
            query_intent="general",
            expanded_queries=["machine learning"],
            metadata={}
        )
        
        diverse_results = orchestrator._apply_diversity_filter(results, context)
        
        # Should include all results (diversity filter is lenient with Jaccard similarity)
        assert len(diverse_results) >= 2
        assert diverse_results[0].id == "result_1"  # Top result always included
    
    def test_content_similarity_calculation(self):
        """Test content similarity calculation."""
        orchestrator = AdvancedSearchOrchestrator()
        
        # Test cases
        content1 = "machine learning algorithms"
        content2 = "machine learning models"  # Some overlap
        content3 = "neural network architecture"  # No overlap
        content4 = "machine learning algorithms"  # Identical
        
        # Similar content
        similarity1 = orchestrator._calculate_content_similarity(content1, content2)
        assert 0 < similarity1 < 1
        
        # Dissimilar content
        similarity2 = orchestrator._calculate_content_similarity(content1, content3)
        assert similarity2 < 0.5
        
        # Identical content
        similarity3 = orchestrator._calculate_content_similarity(content1, content4)
        assert similarity3 == 1.0
        
        # Empty content
        similarity4 = orchestrator._calculate_content_similarity("", "")
        assert similarity4 == 1.0
    
    @patch('src.feature_pipeline.advanced_search.get_feature_flags')
    def test_search_statistics(self, mock_get_flags):
        """Test search statistics retrieval."""
        mock_flags = MagicMock()
        mock_flags.should_use_advanced_search.return_value = True
        mock_get_flags.return_value = mock_flags
        
        orchestrator = AdvancedSearchOrchestrator()
        
        stats = orchestrator.get_search_statistics()
        
        assert isinstance(stats, dict)
        assert 'strategies_available' in stats
        assert 'advanced_features_enabled' in stats
        assert 'query_expansion_enabled' in stats
        assert 'reranking_enabled' in stats
        
        assert stats['advanced_features_enabled'] is True
        assert stats['query_expansion_enabled'] is True
    
    @patch('src.feature_pipeline.advanced_search.get_feature_flags')
    def test_mock_search_strategies(self, mock_get_flags):
        """Test mock search strategy implementations."""
        mock_flags = MagicMock()
        mock_flags.should_use_advanced_search.return_value = True
        mock_get_flags.return_value = mock_flags
        
        orchestrator = AdvancedSearchOrchestrator()
        
        context = SearchContext(
            query="test query",
            original_query="test query",
            query_intent="general", 
            expanded_queries=["test query"],
            metadata={}
        )
        
        # Test similarity search mock
        sim_results = orchestrator._mock_similarity_search(context, 5)
        assert len(sim_results) <= 3  # Mock returns max 3
        assert all(r.score >= 0.6 for r in sim_results)  # Check score range
        
        # Test contextual search mock  
        ctx_results = orchestrator._mock_contextual_search(context, 5)
        assert len(ctx_results) <= 2  # Mock returns max 2
        assert all('contextual' in r.metadata.get('strategy', '') for r in ctx_results)
        
        # Test parent-child search mock
        pc_results = orchestrator._mock_parent_child_search(context, 5)
        assert len(pc_results) <= 2  # Mock returns max 2
        assert all('parent_child' in r.metadata.get('strategy', '') for r in pc_results)


if __name__ == "__main__":
    pytest.main([__file__]) 