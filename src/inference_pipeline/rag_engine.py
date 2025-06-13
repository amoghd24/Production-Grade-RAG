"""
RAG Engine for Second Brain AI Assistant.
Implements Retrieval Augmented Generation using vector search and GPT-4.
"""

from typing import List, Dict, Any, Optional
from src.inference_pipeline.openai_service import OpenAIService
from src.inference_pipeline.prompt_manager import PromptManager
from src.feature_pipeline.vector_storage import MongoVectorStore, create_vector_store
from src.utils.logger import LoggerMixin


class RAGEngine(LoggerMixin):
    """Implements RAG using vector search and GPT-4."""
    
    def __init__(self):
        """Initialize the RAG engine."""
        self.openai_service = OpenAIService()
        self.prompt_manager = PromptManager()
        self.vector_store = None  # Will be initialized in setup
        self.similarity_threshold = 0.7
        # Match OpenAI service token limits
        self.max_tokens_per_doc = 10000  # Increased for GPT-4
        self.max_total_tokens = 128000   # GPT-4's context window
    
    async def setup(self):
        """Initialize the vector store."""
        self.vector_store = await create_vector_store()
    
    async def process_query(
        self,
        query: str,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a user query using RAG.
        
        Args:
            query: User's query
            similarity_threshold: Minimum similarity score for context documents
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Ensure vector store is initialized
            if self.vector_store is None:
                await self.setup()
            
            # Retrieve all relevant context
            context_docs = await self._retrieve_context(
                query,
                threshold=similarity_threshold or self.similarity_threshold
            )
            
            # Create RAG prompt with all relevant context
            prompt = self.prompt_manager.create_rag_prompt(query, context_docs)
            
            # Generate response
            response = await self.openai_service.generate_response(
                prompt=prompt,
                context=context_docs,
                system_message=self.prompt_manager.get_system_message()
            )
            
            return {
                "response": response["response"],
                "context_docs": context_docs,
                "usage": response["usage"],
                "model": response["model"]
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise
    
    async def _retrieve_context(
        self,
        query: str,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all relevant context documents using vector search.
        
        Args:
            query: User's query
            threshold: Minimum similarity score
            
        Returns:
            List of relevant context documents
        """
        try:
            # Get all documents above similarity threshold
            results = await self.vector_store.similarity_search(
                query=query,
                score_threshold=threshold
            )
            
            # Format and process results
            context_docs = []
            total_tokens = 0
            
            for doc in results:
                # Calculate approximate tokens (rough estimate: 4 chars ≈ 1 token)
                doc_tokens = len(doc.get("content", "")) // 4
                
                # Skip if document would exceed token limit
                if total_tokens + doc_tokens > self.max_total_tokens:
                    self.logger.warning(f"Skipping document '{doc.get('title', '')}' due to token limit")
                    continue
                
                # Truncate document if it's too long
                content = doc.get("content", "")
                if doc_tokens > self.max_tokens_per_doc:
                    content = content[:self.max_tokens_per_doc * 4] + "..."
                
                context_docs.append({
                    "title": doc.get("title", ""),
                    "content": content,
                    "score": doc.get("score", 0.0)
                })
                
                total_tokens += doc_tokens
            
            self.logger.info(f"Retrieved {len(context_docs)} documents with {total_tokens} total tokens")
            return context_docs
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            return [] 