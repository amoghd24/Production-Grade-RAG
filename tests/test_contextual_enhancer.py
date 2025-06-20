"""
Tests for Contextual Enhancement Service.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.feature_pipeline.contextual_enhancer import ContextualEnhancer
from src.models.schemas import Document, DocumentChunk, ContentSource, DocumentType
from src.config.feature_flags import FeatureFlag


class TestContextualEnhancer:
    """Test contextual enhancement functionality."""
    
    def test_enhancement_disabled_by_default(self):
        """Test that enhancement is disabled when feature flag is off."""
        enhancer = ContextualEnhancer()
        
        # Create test data
        document = Document(
            title="Test Document",
            content="Test content",
            source=ContentSource.WEB_CRAWL,
            document_type=DocumentType.WEB_PAGE
        )
        
        chunks = [
            DocumentChunk(
                id="test1",
                document_id="doc1",
                content="Original chunk content",
                chunk_index=0,
                word_count=3
            )
        ]
        
        # Enhancement should return unchanged chunks when disabled
        result = enhancer.enhance_chunks(document, chunks)
        assert len(result) == 1
        assert result[0].content == "Original chunk content"
    
    @patch('src.feature_pipeline.contextual_enhancer.get_feature_flags')
    def test_enhancement_enabled(self, mock_get_flags):
        """Test that enhancement works when feature flag is enabled."""
        # Mock feature flags to enable contextual chunking
        mock_flags = MagicMock()
        mock_flags.should_use_contextual_chunking.return_value = True
        mock_flags.get_chunking_config.return_value = {'add_document_context': True}
        mock_get_flags.return_value = mock_flags
        
        enhancer = ContextualEnhancer()
        
        # Create test data
        document = Document(
            title="AI Research Paper",
            content="Test content",
            source=ContentSource.WEB_CRAWL,
            document_type=DocumentType.WEB_PAGE
        )
        
        chunks = [
            DocumentChunk(
                id="test1",
                document_id="doc1",
                content="Machine learning is a powerful tool.",
                chunk_index=0,
                word_count=6
            )
        ]
        
        # Enhancement should add context when enabled
        result = enhancer.enhance_chunks(document, chunks)
        assert len(result) == 1
        
        enhanced_content = result[0].content
        assert "Document: AI Research Paper" in enhanced_content
        assert "Source: web_crawl" in enhanced_content
        assert "Type: web_page" in enhanced_content
        assert "Content: Machine learning is a powerful tool." in enhanced_content
    
    def test_section_header_extraction(self):
        """Test section header extraction from content."""
        enhancer = ContextualEnhancer()
        
        # Test markdown header
        content_with_header = "# Introduction\nThis is the introduction section."
        header = enhancer._extract_section_header(content_with_header)
        assert header == "Introduction"
        
        # Test uppercase header
        content_with_caps = "MACHINE LEARNING BASICS\nThis section covers basics."
        header = enhancer._extract_section_header(content_with_caps)
        assert header == "MACHINE LEARNING BASICS"
        
        # Test no header
        content_no_header = "This is just regular content without headers."
        header = enhancer._extract_section_header(content_no_header)
        assert header == "This is just regular content without headers"  # First sentence
    
    @patch('src.feature_pipeline.contextual_enhancer.get_feature_flags')
    def test_context_template_building(self, mock_get_flags):
        """Test that context template is built correctly."""
        # Mock feature flags
        mock_flags = MagicMock()
        mock_flags.should_use_contextual_chunking.return_value = True
        mock_flags.get_chunking_config.return_value = {'add_document_context': True}
        mock_get_flags.return_value = mock_flags
        
        enhancer = ContextualEnhancer()
        
        # Document with all metadata
        document = Document(
            title="Complete Document",
            content="Test content",
            source=ContentSource.NOTION,
            document_type=DocumentType.NOTION_PAGE
        )
        
        chunk = DocumentChunk(
            id="test1",
            document_id="doc1",
            content="# Section Title\nSection content here.",
            chunk_index=0,
            word_count=4
        )
        
        result = enhancer._add_context_to_chunk(document, chunk)
        
        expected_parts = [
            "Document: Complete Document",
            "Source: notion",
            "Section: Section Title",
            "Type: notion_page",
            "Content: # Section Title\nSection content here."
        ]
        
        for part in expected_parts:
            assert part in result.content
    
    def test_word_count_update(self):
        """Test that word count is updated after enhancement."""
        enhancer = ContextualEnhancer()
        
        document = Document(
            title="Test",
            content="Test content",
            source=ContentSource.WEB_CRAWL,
            document_type=DocumentType.WEB_PAGE
        )
        
        original_chunk = DocumentChunk(
            id="test1",
            document_id="doc1",
            content="Short content.",
            chunk_index=0,
            word_count=2
        )
        
        enhanced_chunk = enhancer._add_context_to_chunk(document, original_chunk)
        
        # Word count should be higher after adding context
        assert enhanced_chunk.word_count > original_chunk.word_count
        assert enhanced_chunk.word_count == len(enhanced_chunk.content.split())


if __name__ == "__main__":
    pytest.main([__file__]) 