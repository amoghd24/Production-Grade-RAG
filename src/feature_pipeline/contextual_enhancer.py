"""
Contextual Enhancement Service for Advanced RAG.
Adds document context to chunks before embedding to improve retrieval quality.
"""

import re
from typing import List, Optional
from src.models.schemas import Document, DocumentChunk
from src.config.feature_flags import get_feature_flags
from src.utils.logger import LoggerMixin


class ContextualEnhancer(LoggerMixin):
    """Adds contextual information to document chunks for better embeddings."""
    
    def __init__(self):
        """Initialize the contextual enhancer."""
        self.feature_flags = get_feature_flags()
    
    def enhance_chunks(self, document: Document, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Add contextual information to chunks if feature is enabled.
        
        Args:
            document: Source document
            chunks: List of chunks to enhance
            
        Returns:
            Enhanced chunks with contextual information
        """
        # Check if contextual chunking is enabled
        if not self.feature_flags.should_use_contextual_chunking():
            return chunks  # Return unchanged if feature disabled
        
        # Get configuration for context templates
        config = self.feature_flags.get_chunking_config()
        add_context = config.get('add_document_context', True)
        
        if not add_context:
            return chunks
        
        # Enhance each chunk with context
        enhanced_chunks = []
        for chunk in chunks:
            enhanced_chunk = self._add_context_to_chunk(document, chunk)
            enhanced_chunks.append(enhanced_chunk)
        
        self.logger.info(f"Enhanced {len(enhanced_chunks)} chunks with contextual information")
        return enhanced_chunks
    
    def _add_context_to_chunk(self, document: Document, chunk: DocumentChunk) -> DocumentChunk:
        """
        Add contextual information to a single chunk.
        
        Args:
            document: Source document
            chunk: Chunk to enhance
            
        Returns:
            Enhanced chunk with context
        """
        # Extract section context from the chunk
        section_header = self._extract_section_header(chunk.content)
        
        # Build context template
        context_parts = []
        
        # Document metadata
        if document.title:
            context_parts.append(f"Document: {document.title}")
        
        if document.source:
            context_parts.append(f"Source: {document.source.value}")
        
        # Section information
        if section_header:
            context_parts.append(f"Section: {section_header}")
        
        # Document type context
        if document.document_type:
            context_parts.append(f"Type: {document.document_type.value}")
        
        # Build the enhanced content
        if context_parts:
            context_header = "\n".join(context_parts)
            enhanced_content = f"{context_header}\n\nContent: {chunk.content}"
        else:
            enhanced_content = chunk.content
        
        # Create new chunk with enhanced content
        enhanced_chunk = DocumentChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            content=enhanced_content,
            chunk_index=chunk.chunk_index,
            word_count=len(enhanced_content.split()),
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            embedding=chunk.embedding,
            embedding_model=chunk.embedding_model,
            created_at=chunk.created_at
        )
        
        return enhanced_chunk
    
    def _extract_section_header(self, content: str) -> Optional[str]:
        """
        Extract section header from chunk content.
        
        Args:
            content: Chunk content
            
        Returns:
            Section header if found, None otherwise
        """
        # Look for markdown headers
        lines = content.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            
            # Markdown headers
            if line.startswith('#'):
                return line.lstrip('#').strip()
            
            # Look for lines that might be headers (all caps, short lines)
            if (len(line) < 100 and 
                len(line) > 5 and 
                line.isupper() and 
                not line.startswith('HTTP')):
                return line
        
        # Extract first sentence as potential section context
        sentences = re.split(r'[.!?]+', content)
        if sentences and len(sentences[0]) < 80:
            return sentences[0].strip()
        
        return None 