"""
Prompt Manager for Second Brain AI Assistant.
Handles prompt templates and system messages for the RAG system.
"""

from typing import Dict, Any, Optional
from src.utils.logger import LoggerMixin


class PromptManager(LoggerMixin):
    """Manages prompt templates and system messages for the RAG system."""
    
    def __init__(self):
        """Initialize the prompt manager with default templates."""
        self.system_message = """You are a helpful AI assistant that helps users find and understand information from their second brain.
        You have access to their personal knowledge base and can provide detailed, accurate responses based on this information.
        Always cite your sources when providing information."""
        
        self.prompt_templates = {
            "general_query": """Based on the provided context, please answer the following question:
            {query}
            
            If the context doesn't contain enough information to answer the question, please say so.""",
            
            "summarize": """Please provide a concise summary of the following information:
            {content}
            
            Focus on the key points and main ideas.""",
            
            "explain": """Please explain the following concept in detail:
            {concept}
            
            Use the provided context to ensure accuracy and completeness."""
        }
    
    def get_system_message(self) -> str:
        """Get the default system message."""
        return self.system_message
    
    def format_prompt(
        self,
        template_name: str,
        **kwargs: Any
    ) -> str:
        """
        Format a prompt using the specified template.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Variables to format the template with
            
        Returns:
            Formatted prompt string
        """
        if template_name not in self.prompt_templates:
            raise ValueError(f"Unknown prompt template: {template_name}")
            
        template = self.prompt_templates[template_name]
        return template.format(**kwargs)
    
    def create_rag_prompt(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a RAG-specific prompt with context.
        
        Args:
            query: User's query
            context: Optional context information
            
        Returns:
            Formatted RAG prompt
        """
        if context:
            return f"""Based on the following context:
            {context}
            
            Please answer this question: {query}
            
            If the context doesn't contain enough information, please say so."""
        else:
            return self.format_prompt("general_query", query=query) 