"""
Parent-Child Chunking Service for Advanced RAG.
Creates hierarchical chunks with parent-child relationships for better context and precision.
"""

from typing import List, Dict, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.models.schemas import Document, DocumentChunk
from src.config.feature_flags import get_feature_flags
from src.utils.logger import LoggerMixin


class ParentChildChunker(LoggerMixin):
    """Creates hierarchical parent-child chunk relationships."""
    
    def __init__(self):
        """Initialize the parent-child chunker."""
        self.feature_flags = get_feature_flags()
        
        # Get configuration
        config = self.feature_flags.get_chunking_config()
        
        # Parent chunk splitter (larger chunks for context)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get('parent_chunk_size', 2000),
            chunk_overlap=config.get('parent_overlap', 400),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Child chunk splitter (smaller chunks for precision)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get('child_chunk_size', 400),
            chunk_overlap=config.get('child_overlap', 100),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def create_parent_child_chunks(self, document: Document, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Convert existing chunks to parent-child hierarchy if feature is enabled.
        
        Args:
            document: Source document
            chunks: Existing chunks to convert
            
        Returns:
            Enhanced chunks with parent-child relationships
        """
        # Check if parent-child chunking is enabled
        if not self.feature_flags.should_use_parent_retrieval():
            return chunks  # Return unchanged if feature disabled
        
        # Create parent-child hierarchy
        parent_child_chunks = self._create_hierarchy(document, chunks)
        
        self.logger.info(f"Created parent-child hierarchy with {len(parent_child_chunks)} total chunks")
        return parent_child_chunks
    
    def _create_hierarchy(self, document: Document, existing_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Create hierarchical parent-child chunk structure.
        
        Args:
            document: Source document
            existing_chunks: Current chunks to enhance
            
        Returns:
            List of chunks with parent-child relationships
        """
        # Split document into parent chunks first
        parent_texts = self.parent_splitter.split_text(document.content)
        
        all_chunks = []
        chunk_index = 0
        
        for parent_idx, parent_text in enumerate(parent_texts):
            # Create parent chunk
            parent_chunk = DocumentChunk(
                id=f"{document.id}_parent_{parent_idx}" if document.id else f"parent_{parent_idx}",
                document_id=document.id or "unknown",
                content=parent_text,
                chunk_index=chunk_index,
                word_count=len(parent_text.split()),
                start_char=None,  # We could calculate this if needed
                end_char=None,
                embedding=None,
                embedding_model=None
            )
            
            # Add parent chunk metadata
            parent_chunk.metadata = {
                "chunk_type": "parent",
                "parent_id": parent_chunk.id,
                "child_count": 0  # Will be updated below
            }
            
            chunk_index += 1
            
            # Split parent into child chunks
            child_texts = self.child_splitter.split_text(parent_text)
            child_chunks = []
            
            for child_idx, child_text in enumerate(child_texts):
                child_chunk = DocumentChunk(
                    id=f"{parent_chunk.id}_child_{child_idx}",
                    document_id=document.id or "unknown",
                    content=child_text,
                    chunk_index=chunk_index,
                    word_count=len(child_text.split()),
                    start_char=None,
                    end_char=None,
                    embedding=None,
                    embedding_model=None
                )
                
                # Add child chunk metadata with parent reference
                child_chunk.metadata = {
                    "chunk_type": "child",
                    "parent_id": parent_chunk.id,
                    "parent_content": parent_text,  # Store parent context
                    "child_index": child_idx
                }
                
                child_chunks.append(child_chunk)
                chunk_index += 1
            
            # Update parent chunk with child count
            parent_chunk.metadata["child_count"] = len(child_chunks)
            
            # Add parent and all its children
            all_chunks.append(parent_chunk)
            all_chunks.extend(child_chunks)
        
        return all_chunks
    
    def get_parent_context(self, child_chunk: DocumentChunk) -> Optional[str]:
        """
        Get parent context for a child chunk.
        
        Args:
            child_chunk: Child chunk to get parent context for
            
        Returns:
            Parent content if available, None otherwise
        """
        if not hasattr(child_chunk, 'metadata') or not child_chunk.metadata:
            return None
        
        return child_chunk.metadata.get('parent_content')
    
    def is_parent_chunk(self, chunk: DocumentChunk) -> bool:
        """Check if chunk is a parent chunk."""
        if not hasattr(chunk, 'metadata') or not chunk.metadata:
            return False
        return chunk.metadata.get('chunk_type') == 'parent'
    
    def is_child_chunk(self, chunk: DocumentChunk) -> bool:
        """Check if chunk is a child chunk."""
        if not hasattr(chunk, 'metadata') or not chunk.metadata:
            return False
        return chunk.metadata.get('chunk_type') == 'child'
    
    def get_chunk_relationships(self, chunks: List[DocumentChunk]) -> Dict[str, List[str]]:
        """
        Get parent-child relationships mapping.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary mapping parent IDs to child IDs
        """
        relationships = {}
        
        for chunk in chunks:
            if self.is_parent_chunk(chunk):
                relationships[chunk.id] = []
            elif self.is_child_chunk(chunk) and hasattr(chunk, 'metadata'):
                parent_id = chunk.metadata.get('parent_id')
                if parent_id:
                    if parent_id not in relationships:
                        relationships[parent_id] = []
                    relationships[parent_id].append(chunk.id)
        
        return relationships 