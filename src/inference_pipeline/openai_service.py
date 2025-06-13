"""
OpenAI Service for Second Brain AI Assistant.
Handles all interactions with OpenAI's GPT-4 model.
"""

from typing import List, Dict, Optional, Any
import openai
from src.config.settings import settings
from src.utils.logger import LoggerMixin


class OpenAIService(LoggerMixin):
    """Service class for handling OpenAI API interactions."""
    
    def __init__(self):
        """Initialize the OpenAI service."""
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4"  # Using GPT-4
        self.max_input_tokens = 128000  # GPT-4's context window
        self.max_output_tokens = 4096   # Maximum tokens for response
        self.temperature = 0.7
        
    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using GPT-4.
        
        Args:
            prompt: The user's input prompt
            context: List of context documents for RAG
            system_message: Optional system message to guide the model
            temperature: Optional temperature for response generation
            max_tokens: Optional max tokens for response
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        try:
            # Prepare messages
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            # Add context if provided
            if context:
                context_str = self._format_context(context)
                messages.append({"role": "system", "content": f"Context: {context_str}"})
            
            # Add user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Calculate total input tokens
            # Rough estimate: 4 chars ≈ 1 token
            total_input_tokens = sum(len(msg["content"]) // 4 for msg in messages)
            
            # Ensure we don't exceed input token limit
            if total_input_tokens > self.max_input_tokens:
                self.logger.warning(f"Input exceeds token limit ({total_input_tokens} > {self.max_input_tokens}), truncating context")
                # Truncate context to fit within limits
                for msg in messages:
                    if msg["role"] == "system" and "Context:" in msg["content"]:
                        max_context_tokens = self.max_input_tokens - (total_input_tokens - len(msg["content"]) // 4)
                        msg["content"] = msg["content"][:max_context_tokens * 4] + "..."
                        break
            
            # Calculate available tokens for response
            available_tokens = min(
                self.max_output_tokens,
                max_tokens or self.max_output_tokens
            )
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=available_tokens
            )
            
            return {
                "response": response.choices[0].message.content,
                "usage": response.usage,
                "model": response.model
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context documents into a string."""
        formatted_context = []
        for doc in context:
            formatted_context.append(f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}")
        return "\n\n".join(formatted_context) 