"""
Response Handler for Second Brain AI Assistant.
Processes and formats responses from the RAG engine.
"""

from typing import Dict, Any, List, Optional
from src.utils.logger import LoggerMixin


class ResponseHandler(LoggerMixin):
    """Handles processing and formatting of RAG responses."""
    
    def __init__(self):
        """Initialize the response handler."""
        self.max_response_length = 2000
        self.include_sources = True
    
    def process_response(
        self,
        rag_response: Dict[str, Any],
        include_sources: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Process and format a RAG response.
        
        Args:
            rag_response: Raw response from RAG engine
            include_sources: Whether to include source documents
            
        Returns:
            Processed and formatted response
        """
        try:
            response = rag_response["response"]
            context_docs = rag_response.get("context_docs", [])
            usage = rag_response.get("usage", {})
            
            # Format the response
            formatted_response = self._format_response(response)
            
            # Add sources if requested
            if include_sources or self.include_sources:
                sources = self._format_sources(context_docs)
                formatted_response = f"{formatted_response}\n\nSources:\n{sources}"
            
            return {
                "content": formatted_response,
                "sources": context_docs if (include_sources or self.include_sources) else [],
                "usage": usage,
                "model": rag_response.get("model", "")
            }
            
        except Exception as e:
            self.logger.error(f"Error processing response: {str(e)}")
            raise
    
    def _format_response(self, response: str) -> str:
        """
        Format the response text.
        
        Args:
            response: Raw response text
            
        Returns:
            Formatted response text
        """
        # Truncate if too long
        if len(response) > self.max_response_length:
            response = response[:self.max_response_length] + "..."
        
        # Clean up formatting
        response = response.strip()
        response = response.replace("\n\n\n", "\n\n")
        
        return response
    
    def _format_sources(self, context_docs: List[Dict[str, Any]]) -> str:
        """
        Format source documents.
        
        Args:
            context_docs: List of context documents
            
        Returns:
            Formatted sources text
        """
        if not context_docs:
            return "No sources available."
        
        sources = []
        for i, doc in enumerate(context_docs, 1):
            title = doc.get("title", "Untitled")
            score = doc.get("score", 0.0)
            sources.append(f"{i}. {title} (Relevance: {score:.2f})")
        
        return "\n".join(sources) 