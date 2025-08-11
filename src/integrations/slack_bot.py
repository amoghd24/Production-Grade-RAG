"""
Direct Slack Bot for Second Brain AI Assistant.
Connects directly to RAG engine without FastAPI.
"""

import os
import asyncio
import re
from typing import Dict, List
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from src.inference_pipeline.agent import create_second_brain_agent
from src.config.settings import settings
from src.utils.logger import LoggerMixin


class SecondBrainSlackBot(LoggerMixin):
    """Direct Slack bot for Second Brain AI Assistant."""
    
    def __init__(self):
        """Initialize the Slack bot."""
        # Initialize Slack app
        self.app = AsyncApp(token=settings.SLACK_BOT_TOKEN)
        
        # Initialize agent
        self.agent = None
        
        # Conversation history per channel
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        
        # Setup event handlers
        self._setup_handlers()
    
    async def start(self):
        """Start the Slack bot."""
        try:
            # Initialize agent
            self.logger.info("Initializing Second Brain Agent...")
            self.agent = create_second_brain_agent()
            self.logger.info("Agent initialized successfully")
            
            # Start bot with Socket Mode
            handler = AsyncSocketModeHandler(self.app, settings.SLACK_APP_TOKEN)
            self.logger.info("Starting Slack bot...")
            await handler.start_async()
            
        except Exception as e:
            self.logger.error(f"Failed to start Slack bot: {e}")
            raise
    
    def _setup_handlers(self):
        """Setup Slack event handlers."""
        
        @self.app.message("")
        async def handle_message(message, say):
            """Handle all messages."""
            try:
                # Get message details
                text = message.get("text", "").strip()
                channel = message.get("channel")
                user = message.get("user")
                
                if not text or not channel:
                    return
                
                # Get conversation context
                context = self._get_conversation_context(channel, text)
                
                # Query RAG engine with timeout handling
                self.logger.info(f"Processing query from user {user}: {text}")
                
                try:
                    # Add asyncio timeout as additional safety net
                    result = await asyncio.wait_for(
                        self.agent.process_query(context), 
                        timeout=45.0  # 45-second timeout
                    )
                    
                    # Extract the complete formatted response with sources and confidence
                    response_text = self._format_complete_rag_response(result)
                    
                    # Send the actual response
                    await say(response_text)
                    
                    # Update conversation history
                    self._add_to_conversation(channel, text, result["response"])
                    
                except asyncio.TimeoutError:
                    await say("⏰ Sorry, your query is taking longer than expected. Please try a more specific question or try again later.")
                    self.logger.error(f"Query timeout for user {user}: {text}")
                    
            except Exception as e:
                self.logger.error(f"Error handling message: {e}")
                await say("Sorry, I encountered an error processing your message.")
        
        @self.app.event("app_mention")
        async def handle_mention(event, say):
            """Handle app mentions."""
            try:
                # Extract text and remove mention
                text = event.get("text", "")
                text = self._clean_mention(text)
                
                if not text.strip():
                    await say("Hello! Ask me anything about your knowledge base.")
                    return
                
                channel = event.get("channel")
                
                # Get conversation context
                context = self._get_conversation_context(channel, text)
                
                # Query RAG engine with timeout handling
                try:
                    # Add asyncio timeout as additional safety net
                    result = await asyncio.wait_for(
                        self.agent.process_query(context), 
                        timeout=45.0  # 45-second timeout
                    )
                    
                    # Extract the complete formatted response with sources and confidence
                    response_text = self._format_complete_rag_response(result)
                    
                    await say(response_text)
                    
                    # Update conversation history
                    self._add_to_conversation(channel, text, result["response"])
                    
                except asyncio.TimeoutError:
                    await say("⏰ Sorry, your query is taking longer than expected. Please try a more specific question or try again later.")
                    self.logger.error(f"Query timeout for mention: {text}")
                
            except Exception as e:
                self.logger.error(f"Error handling mention: {e}")
                await say("Sorry, I encountered an error processing your mention.")
    
    def _get_conversation_context(self, channel_id: str, current_message: str) -> str:
        """Get conversation history as context."""
        if channel_id not in self.conversations:
            self.conversations[channel_id] = []
        
        # Add current message
        self.conversations[channel_id].append({
            "role": "user",
            "content": current_message
        })
        
        # Keep last 10 messages
        self.conversations[channel_id] = self.conversations[channel_id][-10:]
        
        # Format as conversation
        context_parts = []
        for msg in self.conversations[channel_id]:
            if msg["role"] == "user":
                context_parts.append(f"User: {msg['content']}")
            else:
                context_parts.append(f"Assistant: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def _add_to_conversation(self, channel_id: str, user_msg: str, assistant_msg: str):
        """Add assistant response to conversation history."""
        if channel_id in self.conversations:
            self.conversations[channel_id].append({
                "role": "assistant",
                "content": assistant_msg
            })
    
    def _clean_mention(self, text: str) -> str:
        """Remove bot mention from text."""
        return re.sub(r'<@[A-Z0-9]+>', '', text).strip()
    
    def _convert_markdown_to_slack(self, text: str) -> str:
        """Convert markdown formatting to Slack format."""
        # Convert [text](url) to <url|text>
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', text)
        
        # Convert **bold** to *bold* (more robust pattern)
        text = re.sub(r'\*\*([^*]+)\*\*', r'*\1*', text)
        
        # Convert markdown headers (### Header) to bold (*Header*)
        text = re.sub(r'^#{1,6}\s*(.+)$', r'*\1*', text, flags=re.MULTILINE)
        
        # Convert bare URLs in parentheses to clickable links
        text = re.sub(r'\(https?://[^)]+\)', lambda m: f'<{m.group(0)[1:-1]}>', text)
        
        return text
    
    def _format_complete_rag_response(self, result: Dict) -> str:
        """Format the complete RAG response with sources and confidence like in terminal."""
        try:
            # Get the main response
            main_response = result.get("response", "")
            
            # Try to extract intermediate steps for the full RAG response
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Look for the actual tool output that contains the full formatted response
            full_response = None
            for step in intermediate_steps:
                if len(step) >= 2:
                    tool_output = step[1]  # The second element is usually the tool output
                    if isinstance(tool_output, str) and "📚" in tool_output and "Sources" in tool_output:
                        full_response = tool_output
                        break
            
            # If we found the full formatted response from tools, use it tim
            if full_response:
                return self._convert_markdown_to_slack(full_response)
            
            # Fallback: build our own formatted response
            formatted_response = main_response
            
            # Add metadata info if available
            metadata = result.get("metadata", {})
            if metadata:
                # Add confidence and processing info
                confidence = metadata.get("confidence", 0.0)
                if confidence:
                    formatted_response += f"\n\n**Confidence Level:** {confidence:.2f}"
                
                # Add sources information
                sources_info = self._extract_detailed_sources(metadata)
                if sources_info:
                    formatted_response += f"\n\n{sources_info}"
            
            return self._convert_markdown_to_slack(formatted_response)
            
        except Exception as e:
            self.logger.error(f"Error formatting complete RAG response: {e}")
            return result.get("response", "Sorry, I encountered an error formatting the response.")
    
    def _extract_detailed_sources(self, metadata: Dict) -> str:
        """Extract detailed sources with confidence and URLs."""
        try:
            sources_text = "📚 **Sources:**"
            
            # Look for sources in metadata
            if "sources" in str(metadata).lower():
                sources_text += "\n• Based on your knowledge base"
                
            # Add confidence if available
            confidence = metadata.get("confidence", 0.0)
            if confidence:
                sources_text += f"\n• Confidence: {confidence:.2f}"
                
            # Add tools used
            tools_used = metadata.get("tools_used", [])
            if tools_used:
                sources_text += f"\n• Tools used: {', '.join(tools_used)}"
            
            return sources_text
            
        except Exception:
            return "📚 **Sources:** Based on your knowledge base"


async def main():
    """Main function to run the Slack bot."""
    bot = SecondBrainSlackBot()
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main()) 