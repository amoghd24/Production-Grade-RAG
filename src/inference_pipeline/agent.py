"""
Second Brain AI Assistant Agent using Langchain's ReAct Agent.
Integrates RAG capabilities with intelligent tool selection.
"""

from typing import Dict, Any, List, Optional
import asyncio

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.inference_pipeline.tools import RetrieverTool, SummarizationTool
from src.config.settings import settings
from src.utils.logger import LoggerMixin


class SecondBrainAgent(LoggerMixin):
    """
    Main agent class that orchestrates tool usage for the Second Brain AI Assistant.
    Uses Langchain's ReAct agent framework for reasoning and acting.
    """
    
    def __init__(self):
        """Initialize the agent with tools and LLM."""
        self.llm = ChatOpenAI(
            model="gpt-4o",  # Latest GPT-4o with 128K context
            temperature=0.1,  # Low temperature for consistent reasoning
            api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize tools
        self.tools = [
            RetrieverTool(),
            SummarizationTool()
        ]
        
        # Create the agent
        self.agent_executor = None
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the ReAct agent with custom prompt."""
        
        # Custom prompt template for Second Brain AI Assistant
        react_prompt = PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
            template="""You are the Second Brain AI Assistant, designed to help users access and understand their personal knowledge base.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Key Guidelines:
1. Always search the knowledge base first when asked about specific information
2. Use summarization when dealing with long content or multiple sources  
3. Be helpful and explain your reasoning process
4. If you can't find information, be honest about limitations
5. Always cite sources when providing information from the knowledge base

Question: {input}
Thought: {agent_scratchpad}"""
        )
        
        # Create the ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,  # Enable verbose output for debugging
            max_iterations=5,  # Limit iterations to prevent infinite loops
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query using the ReAct agent.
        
        Args:
            query: User's input query
            
        Returns:
            Dictionary containing the agent's response and metadata
        """
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Execute the agent
            result = await self.agent_executor.ainvoke({"input": query})
            
            response = {
                "response": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "agent_type": "react",
                "tools_used": self._extract_tools_used(result.get("intermediate_steps", []))
            }
            
            self.logger.info(f"Agent completed successfully. Tools used: {response['tools_used']}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in agent processing: {str(e)}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "error": str(e),
                "agent_type": "react",
                "tools_used": []
            }
    
    def _extract_tools_used(self, intermediate_steps: List) -> List[str]:
        """Extract the names of tools used during agent execution."""
        tools_used = []
        for step in intermediate_steps:
            if hasattr(step[0], 'tool'):
                tools_used.append(step[0].tool)
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