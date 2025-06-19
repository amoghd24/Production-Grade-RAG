#!/usr/bin/env python3
"""
Test script for the Second Brain AI Assistant Agent.
Run this to verify the agent implementation works correctly.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.inference_pipeline.agent import create_second_brain_agent


async def test_agent():
    """Test the agent with various queries."""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    print("🚀 Initializing Second Brain Agent...")
    
    try:
        # Create agent
        agent = create_second_brain_agent()
        
        # Test health check
        print("\n🔍 Testing agent health...")
        health = await agent.health_check()
        print(f"Agent status: {health['status']}")
        
        if health['status'] != 'healthy':
            print(f"❌ Agent is not healthy: {health}")
            return
        
        # Test queries
        test_queries = [
            "Which teams are playing in the Fifa Club World Cup 2025?",
            "What are the high protein diets and their recipes?"
            "List all the biggest ai companies"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🤖 Test {i}: {query}")
            print("-" * 50)
            
            try:
                result = await agent.process_query(query)
                print(f"Response: {result['response']}")
                print(f"Tools used: {result['tools_used']}")
                
            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
        
        print("\n✅ Agent testing completed!")
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_agent()) 