"""
Second Brain AI Assistant Agent using Langchain's ReAct Agent.
Integrates RAG capabilities with intelligent tool selection.
"""

from typing import Dict, Any, List, Optional
import asyncio

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from src.inference_pipeline.tools import RetrieverTool, SummarizationTool
from src.inference_pipeline.prompt_manager import PromptManager
from src.config.settings import settings
from src.utils.logger import LoggerMixin


class SecondBrainAgent(LoggerMixin):
    """
    Main agent class that orchestrates tool usage for the Second Brain AI Assistant.
    Uses LangGraph's ReAct agent for reliable execution.
    """
    
    def __init__(self):
        """Initialize the agent with tools and LLM."""
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0.0,
            api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
        
        # Initialize tools
        self.tools = [
            RetrieverTool(),
            SummarizationTool()
        ]
        
        # Create system prompt for knowledge base restriction
        system_prompt = """You are a Second Brain AI Assistant with access to the user's personal knowledge base.

CRITICAL INSTRUCTIONS:
- You must ONLY provide information that exists in the knowledge base accessible through your tools
- NEVER make up, invent, or predict information that is not in the knowledge base  
- If information is not available in the knowledge base, clearly state that you don't have that information
- Always use your retrieval tools to search the knowledge base before answering
- When you find relevant information, cite your sources clearly
- If the knowledge base doesn't contain enough information to answer a question, say so explicitly

Your role is to help users find and understand information from their second brain, not to provide general knowledge or make predictions."""
        
        # Create LangGraph ReAct agent with system prompt
        self.agent_executor = create_react_agent(
            self.llm,
            self.tools,
            state_modifier=lambda state: [SystemMessage(content=system_prompt)] + state["messages"],
            interrupt_before=None,
            interrupt_after=None,
            checkpointer=None
        )
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query using the LangGraph ReAct agent.
        
        Args:
            query: User's input query
            
        Returns:
            Dictionary containing the agent's response and metadata
        """
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Execute the LangGraph agent
            result = await self.agent_executor.ainvoke(
                {"messages": [("user", query)]},
                config={"configurable": {"thread_id": "1"}}
            )
            
            # Extract the final message
            final_message = result["messages"][-1]
            
            response = {
                "response": final_message.content,
                "intermediate_steps": [],  # LangGraph handles this differently
                "agent_type": "langgraph_react",
                "tools_used": self._extract_tools_from_messages(result["messages"])
            }
            
            self.logger.info(f"Agent completed successfully. Tools used: {response['tools_used']}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in agent processing: {str(e)}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "error": str(e),
                "agent_type": "langgraph_react",
                "tools_used": []
            }
    
    def _extract_tools_from_messages(self, messages: List) -> List[str]:
        """Extract the names of tools used from LangGraph messages."""
        tools_used = []
        for message in messages:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tools_used.append(tool_call.get('name', 'unknown'))
        return list(set(tools_used))  # Remove duplicates
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the agent and its tools."""
        try:
            # Test basic agent functionality
            test_response = await self.process_query("What can you do?")
            
            return {
                "status": "healthy",
                "agent_executor_ready": self.agent_executor is not None,
                "tools_count": len(self.tools),
                "llm_model": self.llm.model_name,
                "test_response_length": len(test_response["response"])
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "agent_executor_ready": self.agent_executor is not None,
                "tools_count": len(self.tools)
            }


# Convenience function for creating agent instances
def create_second_brain_agent() -> SecondBrainAgent:
    """Create and return a configured SecondBrainAgent instance."""
    return SecondBrainAgent() 