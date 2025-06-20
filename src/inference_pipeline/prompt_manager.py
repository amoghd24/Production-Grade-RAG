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
    
    def get_advanced_system_message(self) -> str:
        """Get enhanced system message for advanced RAG."""
        return """You are an advanced AI assistant with access to the user's comprehensive knowledge base including their Notion workspace and linked external resources.

Your capabilities:
- Access to multiple types of sources (Notion pages, web articles, PDFs, etc.)
- Understanding of document relationships and hierarchies  
- Ability to synthesize information from multiple sources
- Confidence assessment of your responses
- Conversation context awareness

When responding:
1. Provide accurate, comprehensive answers based on the available sources
2. Always cite your sources with clear attribution
3. Indicate confidence level in your response
4. Reference previous conversation context when relevant
5. If information is incomplete, clearly state what's missing
6. Organize your response logically with clear structure

Your responses should be helpful, accurate, and well-sourced."""
    
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
    
    def create_no_context_prompt(self, query: str) -> str:
        """Create prompt when no context documents are found."""
        return f"""I don't have specific information about '{query}' in my knowledge base. 

I can:
1. Provide general information about this topic if you'd like
2. Suggest how to find this information in your Notion workspace
3. Help you formulate a more specific search query

How would you like me to help?

Query: {query}"""
    
    def create_advanced_rag_prompt(
        self,
        query: str,
        structured_context: str,
        conversation_context: Optional[str] = None,
        source_count: int = 0
    ) -> str:
        """Create advanced RAG prompt with structured context and source attribution."""
        
        prompt_parts = []
        
        # Add conversation context if available
        if conversation_context:
            prompt_parts.append("=== CONVERSATION CONTEXT ===")
            prompt_parts.append(conversation_context)
            prompt_parts.append("")
        
        # Add structured context
        prompt_parts.append("=== KNOWLEDGE BASE CONTEXT ===")
        prompt_parts.append(f"Found {source_count} relevant sources for your query:")
        prompt_parts.append("")
        prompt_parts.append(structured_context)
        prompt_parts.append("")
        
        # Add instructions
        prompt_parts.append("=== INSTRUCTIONS ===")
        prompt_parts.append("Based on the above context, please provide a comprehensive answer to the user's query.")
        prompt_parts.append("Important guidelines:")
        prompt_parts.append("- Synthesize information from multiple sources when relevant")
        prompt_parts.append("- Clearly cite which sources support each point you make")
        prompt_parts.append("- If sources conflict, acknowledge the differences")
        prompt_parts.append("- Use conversation context to provide more relevant answers")
        prompt_parts.append("- Include clickable URLs when available")
        prompt_parts.append("- Indicate your confidence level (High/Medium/Low)")
        prompt_parts.append("")
        
        # Add the actual query
        prompt_parts.append(f"=== USER QUERY ===")
        prompt_parts.append(f"Query: {query}")
        
        return "\n".join(prompt_parts) 