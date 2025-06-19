"""
Tools for the Second Brain AI Assistant Agent.
Implements the three core tools: retriever, summarization, and help.
"""

from typing import Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import Field

from src.inference_pipeline.rag_engine import RAGEngine
from src.inference_pipeline.openai_service import OpenAIService
from src.utils.logger import LoggerMixin


class RetrieverTool(BaseTool, LoggerMixin):
    """Tool for retrieving information from the knowledge base using RAG."""
    
    name: str = "knowledge_base_search"
    description: str = """Use this tool to search for information in the user's knowledge base.
    Input should be a search query or question about the content.
    This tool performs semantic search and returns relevant information with sources."""
    
    rag_engine: RAGEngine = Field(default_factory=RAGEngine)
    
    class Config:
        arbitrary_types_allowed = True
    
    async def _arun(self, query: str) -> str:
        """Async implementation of the tool."""
        try:
            # Ensure RAG engine is set up
            if self.rag_engine.vector_store is None:
                await self.rag_engine.setup()
            
            # Process the query
            result = await self.rag_engine.process_query(query)
            
            # Format the response with sources
            response = result["response"]
            context_docs = result.get("context_docs", [])
            
            if context_docs:
                sources = "\n".join([
                    f"- {doc.get('title', 'Unknown')} (Relevance: {doc.get('score', 0.0):.2f})"
                    for doc in context_docs[:3]  # Show top 3 sources
                ])
                response += f"\n\nSources:\n{sources}"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in retriever tool: {str(e)}")
            return f"Sorry, I encountered an error while searching: {str(e)}"
    
    def _run(self, query: str) -> str:
        """Sync wrapper - not implemented for async tool."""
        raise NotImplementedError("This tool only supports async operation")


class SummarizationTool(BaseTool, LoggerMixin):
    """Tool for summarizing content using OpenAI."""
    
    name: str = "summarize_content"
    description: str = """Use this tool to summarize long content or multiple documents.
    Input should be the text content you want to summarize.
    This tool creates concise summaries focusing on key points."""
    
    openai_service: OpenAIService = Field(default_factory=OpenAIService)
    
    class Config:
        arbitrary_types_allowed = True
    
    async def _arun(self, content: str) -> str:
        """Async implementation of the tool."""
        try:
            # Create summarization prompt
            prompt = f"""Please provide a concise summary of the following content, focusing on the key points and main ideas:

{content}

Summary:"""

            # Generate summary using OpenAI
            result = self.openai_service.generate_response(
                prompt=prompt,
                system_message="You are a helpful assistant that creates clear, concise summaries.",
                temperature=0.3,  # Lower temperature for more focused summaries
                max_tokens=500    # Limit summary length
            )
            
            return result["response"]
            
        except Exception as e:
            self.logger.error(f"Error in summarization tool: {str(e)}")
            return f"Sorry, I couldn't summarize the content: {str(e)}"
    
    def _run(self, content: str) -> str:
        """Sync wrapper - not implemented for async tool."""
        raise NotImplementedError("This tool only supports async operation")


class WhatCanIDoTool(BaseTool, LoggerMixin):
    """Tool that explains the assistant's capabilities."""
    
    name: str = "what_can_i_do"
    description: str = """Use this tool when the user asks about your capabilities, features, or what you can help with.
    No input required - just call this tool to get information about available features."""
    
    def _run(self, query: str = "") -> str:
        """Explain the assistant's capabilities."""
        return """I'm your Second Brain AI Assistant! Here's what I can help you with:

🔍 **Knowledge Base Search**: I can search through your personal knowledge base to find relevant information, documents, and notes. Ask me questions about any content you've stored.

📝 **Content Summarization**: I can summarize long documents, articles, or multiple pieces of content to help you quickly understand key points.

💡 **Intelligent Assistance**: I can:
- Answer questions based on your stored knowledge
- Help you find connections between different pieces of information
- Provide detailed explanations of concepts from your knowledge base
- Combine information from multiple sources to give comprehensive answers

🎯 **How to use me**:
- Ask direct questions: "What did I learn about machine learning?"
- Request summaries: "Summarize my notes on project management"
- Seek explanations: "Explain the concept of RAG from my documents"
- Find connections: "How do these topics relate to each other?"

Just ask me anything about your knowledge base, and I'll help you find and understand the information!"""
    
    async def _arun(self, query: str = "") -> str:
        """Async implementation - same as sync for this tool."""
        return self._run(query) 