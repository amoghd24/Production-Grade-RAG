"""
Tests for Parent-Child Chunking Service.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.feature_pipeline.parent_child_chunker import ParentChildChunker
from src.models.schemas import Document, DocumentChunk, ContentSource, DocumentType


class TestParentChildChunker:
    """Test parent-child chunking functionality."""
    
    def test_chunking_disabled_by_default(self):
        """Test that parent-child chunking is disabled when feature flag is off."""
        chunker = ParentChildChunker()
        
        # Create test data
        document = Document(
            title="Test Document",
            content="Test content for chunking",
            source=ContentSource.WEB_CRAWL,
            document_type=DocumentType.WEB_PAGE
        )
        
        original_chunks = [
            DocumentChunk(
                id="test1",
                document_id="doc1",
                content="Original chunk content",
                chunk_index=0,
                word_count=3
            )
        ]
        
        # Should return unchanged chunks when disabled
        result = chunker.create_parent_child_chunks(document, original_chunks)
        assert len(result) == 1
        assert result[0].content == "Original chunk content"
    
    @patch('src.feature_pipeline.parent_child_chunker.get_feature_flags')
    def test_parent_child_creation_enabled(self, mock_get_flags):
        """Test parent-child hierarchy creation when feature is enabled."""
        # Mock feature flags to enable parent-child chunking
        mock_flags = MagicMock()
        mock_flags.should_use_parent_retrieval.return_value = True
        mock_flags.get_chunking_config.return_value = {
            'parent_chunk_size': 500,
            'parent_overlap': 100,
            'child_chunk_size': 150,
            'child_overlap': 30
        }
        mock_get_flags.return_value = mock_flags
        
        chunker = ParentChildChunker()
        
        # Create test document with enough content for parent-child splitting
        document = Document(
            id="test_doc",
            title="ML Research Paper",
            content="""Machine learning is a powerful tool for pattern recognition and data analysis. 
It has revolutionized many fields including computer vision, natural language processing, 
and robotics. The field continues to evolve with new architectures and techniques.

Deep learning, a subset of machine learning, uses neural networks with multiple layers 
to learn complex patterns in data. Convolutional neural networks excel at image processing 
while recurrent neural networks are effective for sequential data.

Applications of machine learning span across industries from healthcare and finance 
to autonomous vehicles and recommendation systems. The technology enables computers 
to make predictions and decisions based on data.""",
            source=ContentSource.WEB_CRAWL,
            document_type=DocumentType.WEB_PAGE
        )
        
        # Create parent-child hierarchy
        result = chunker.create_parent_child_chunks(document, [])
        
        # Should create both parent and child chunks
        assert len(result) > 2  # At least some parent and child chunks
        
        # Check for parent chunks
        parent_chunks = [chunk for chunk in result if chunker.is_parent_chunk(chunk)]
        assert len(parent_chunks) > 0
        
        # Check for child chunks
        child_chunks = [chunk for chunk in result if chunker.is_child_chunk(chunk)]
        assert len(child_chunks) > 0
        
        # Child chunks should have parent references
        for child in child_chunks:
            assert 'parent_id' in child.metadata
            assert 'parent_content' in child.metadata
            assert child.metadata['chunk_type'] == 'child'
    
    @patch('src.feature_pipeline.parent_child_chunker.get_feature_flags')
    def test_parent_chunk_properties(self, mock_get_flags):
        """Test parent chunk properties and metadata."""
        # Mock feature flags
        mock_flags = MagicMock()
        mock_flags.should_use_parent_retrieval.return_value = True
        mock_flags.get_chunking_config.return_value = {
            'parent_chunk_size': 300,
            'parent_overlap': 50,
            'child_chunk_size': 100,
            'child_overlap': 20
        }
        mock_get_flags.return_value = mock_flags
        
        chunker = ParentChildChunker()
        
        document = Document(
            id="test_doc",
            title="Test Document",
            content="A" * 600,  # Long content to ensure parent-child split
            source=ContentSource.WEB_CRAWL,
            document_type=DocumentType.WEB_PAGE
        )
        
        result = chunker.create_parent_child_chunks(document, [])
        parent_chunks = [chunk for chunk in result if chunker.is_parent_chunk(chunk)]
        
        # Check parent chunk properties
        for parent in parent_chunks:
            assert parent.metadata['chunk_type'] == 'parent'
            assert 'child_count' in parent.metadata
            assert parent.metadata['parent_id'] == parent.id
            assert parent.id.endswith('_parent_0') or '_parent_' in parent.id
    
    @patch('src.feature_pipeline.parent_child_chunker.get_feature_flags')
    def test_child_chunk_properties(self, mock_get_flags):
        """Test child chunk properties and parent references."""
        # Mock feature flags
        mock_flags = MagicMock()
        mock_flags.should_use_parent_retrieval.return_value = True
        mock_flags.get_chunking_config.return_value = {
            'parent_chunk_size': 200,
            'parent_overlap': 40,
            'child_chunk_size': 80,
            'child_overlap': 15
        }
        mock_get_flags.return_value = mock_flags
        
        chunker = ParentChildChunker()
        
        document = Document(
            id="test_doc",
            title="Test Document",
            content="B" * 400,  # Long content
            source=ContentSource.WEB_CRAWL,
            document_type=DocumentType.WEB_PAGE
        )
        
        result = chunker.create_parent_child_chunks(document, [])
        child_chunks = [chunk for chunk in result if chunker.is_child_chunk(chunk)]
        
        # Check child chunk properties
        for child in child_chunks:
            assert child.metadata['chunk_type'] == 'child'
            assert 'parent_id' in child.metadata
            assert 'parent_content' in child.metadata
            assert 'child_index' in child.metadata
            assert child.id.endswith('_child_0') or '_child_' in child.id
            
            # Parent content should be longer than child content
            parent_content = child.metadata['parent_content']
            assert len(parent_content) >= len(child.content)
    
    def test_parent_context_retrieval(self):
        """Test retrieving parent context for child chunks."""
        chunker = ParentChildChunker()
        
        # Create a child chunk with parent metadata
        child_chunk = DocumentChunk(
            id="test_child",
            document_id="doc1",
            content="Child content",
            chunk_index=0,
            word_count=2,
            metadata={
                'chunk_type': 'child',
                'parent_id': 'parent_123',
                'parent_content': 'This is the full parent context with more details',
                'child_index': 0
            }
        )
        
        parent_context = chunker.get_parent_context(child_chunk)
        assert parent_context == 'This is the full parent context with more details'
        
        # Test with chunk without metadata
        chunk_no_metadata = DocumentChunk(
            id="test_no_meta",
            document_id="doc1",
            content="No metadata",
            chunk_index=0,
            word_count=2
        )
        
        parent_context = chunker.get_parent_context(chunk_no_metadata)
        assert parent_context is None
    
    def test_chunk_type_identification(self):
        """Test identifying parent and child chunks."""
        chunker = ParentChildChunker()
        
        # Create parent chunk
        parent_chunk = DocumentChunk(
            id="parent_1",
            document_id="doc1",
            content="Parent content",
            chunk_index=0,
            word_count=2,
            metadata={'chunk_type': 'parent'}
        )
        
        # Create child chunk
        child_chunk = DocumentChunk(
            id="child_1",
            document_id="doc1",
            content="Child content",
            chunk_index=1,
            word_count=2,
            metadata={'chunk_type': 'child'}
        )
        
        # Create regular chunk
        regular_chunk = DocumentChunk(
            id="regular_1",
            document_id="doc1",
            content="Regular content",
            chunk_index=2,
            word_count=2
        )
        
        assert chunker.is_parent_chunk(parent_chunk) is True
        assert chunker.is_child_chunk(parent_chunk) is False
        
        assert chunker.is_parent_chunk(child_chunk) is False
        assert chunker.is_child_chunk(child_chunk) is True
        
        assert chunker.is_parent_chunk(regular_chunk) is False
        assert chunker.is_child_chunk(regular_chunk) is False
    
    def test_chunk_relationships_mapping(self):
        """Test getting parent-child relationships."""
        chunker = ParentChildChunker()
        
        chunks = [
            DocumentChunk(
                id="parent_1",
                document_id="doc1",
                content="Parent 1",
                chunk_index=0,
                word_count=2,
                metadata={'chunk_type': 'parent'}
            ),
            DocumentChunk(
                id="child_1_1",
                document_id="doc1",
                content="Child 1.1",
                chunk_index=1,
                word_count=2,
                metadata={'chunk_type': 'child', 'parent_id': 'parent_1'}
            ),
            DocumentChunk(
                id="child_1_2",
                document_id="doc1",
                content="Child 1.2",
                chunk_index=2,
                word_count=2,
                metadata={'chunk_type': 'child', 'parent_id': 'parent_1'}
            ),
            DocumentChunk(
                id="parent_2",
                document_id="doc1",
                content="Parent 2",
                chunk_index=3,
                word_count=2,
                metadata={'chunk_type': 'parent'}
            )
        ]
        
        relationships = chunker.get_chunk_relationships(chunks)
        
        assert 'parent_1' in relationships
        assert 'parent_2' in relationships
        assert len(relationships['parent_1']) == 2
        assert 'child_1_1' in relationships['parent_1']
        assert 'child_1_2' in relationships['parent_1']
        assert len(relationships['parent_2']) == 0


if __name__ == "__main__":
    pytest.main([__file__]) 