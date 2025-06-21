"""
FastAPI service layer for the Second Brain AI Assistant.
Provides REST API endpoints for the agentic system.
"""

from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
import asyncio
from datetime import datetime

from src.inference_pipeline.agent import SecondBrainAgent, create_second_brain_agent
from src.utils.logger import LoggerMixin
from src.models.agent import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    CapabilitiesResponse,
    APIInfoResponse
)





# FastAPI Application
class SecondBrainAPI(LoggerMixin):
    """FastAPI application for Second Brain AI Assistant."""
    
    def __init__(self):
        """Initialize the FastAPI application."""
        self.app = FastAPI(
            title="Second Brain AI Assistant",
            description="Intelligent assistant for your personal knowledge base",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize agent
        self.agent: Optional[SecondBrainAgent] = None
        self._setup_routes()
    

    def _setup_routes(self):
        """Set up API routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize agent on startup."""
            try:
                self.logger.info("Initializing Second Brain Agent...")
                self.agent = create_second_brain_agent()
                self.logger.info("Agent initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize agent: {str(e)}")
                raise
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown."""
            self.logger.info("Shutting down Second Brain API...")
            # Add any cleanup logic here
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                if self.agent is None:
                    return HealthResponse(
                        status="unhealthy",
                        agent_ready=False,
                        details={"error": "Agent not initialized"}
                    )
                
                # Perform agent health check
                health_details = await self.agent.health_check()
                
                return HealthResponse(
                    status=health_details["status"],
                    agent_ready=health_details["agent_executor_ready"],
                    details=health_details
                )
                
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                return HealthResponse(
                    status="unhealthy",
                    agent_ready=False,
                    details={"error": str(e)}
                )
        
        @self.app.post("/query", response_model=QueryResponse)
        async def process_query(request: QueryRequest):
            """Process a user query using the agent."""
            if self.agent is None:
                raise HTTPException(
                    status_code=503,
                    detail="Agent not initialized. Please check service health."
                )
            
            start_time = datetime.utcnow()
            
            try:
                self.logger.info(f"Processing query: {request.query}")
                
                # Process query with agent
                result = await self.agent.process_query(request.query)
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                response = QueryResponse(
                    response=result["response"],
                    tools_used=result.get("tools_used", []),
                    processing_time_ms=processing_time
                )
                
                # Add metadata if requested
                if request.include_metadata:
                    response.metadata = {
                        "agent_type": result.get("agent_type", "react"),
                        "intermediate_steps_count": len(result.get("intermediate_steps", [])),
                        "error": result.get("error")
                    }
                
                return response
                
            except Exception as e:
                self.logger.error(f"Query processing failed: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Query processing failed: {str(e)}"
                )
        
        @self.app.get("/capabilities", response_model=CapabilitiesResponse)
        async def get_capabilities():
            """Get agent capabilities."""
            if self.agent is None:
                raise HTTPException(
                    status_code=503,
                    detail="Agent not initialized"
                )
            
            try:
                # Use the agent to get its capabilities
                result = await self.agent.process_query("What can you do?")
                return CapabilitiesResponse(
                    capabilities=result["response"],
                    tools_available=[tool.name for tool in self.agent.tools],
                    agent_type="react"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to get capabilities: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get capabilities: {str(e)}"
                )
        

        @self.app.get("/", response_model=APIInfoResponse)
        async def root():
            """Root endpoint with API information."""
            return APIInfoResponse(
                name="Second Brain AI Assistant",
                version="1.0.0",
                description="Intelligent assistant for your personal knowledge base",
                endpoints={
                    "health": "/health",
                    "query": "/query",
                    "capabilities": "/capabilities",
                    "docs": "/docs"
                }
            )
    

# Create the application instance
def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    api = SecondBrainAPI()
    return api.app


# For running directly
if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 