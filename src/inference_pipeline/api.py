"""
FastAPI service layer for the Second Brain AI Assistant.
Provides REST API endpoints for the agentic system.
"""

from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio
from datetime import datetime

from src.inference_pipeline.agent import SecondBrainAgent, create_second_brain_agent
from src.utils.logger import LoggerMixin


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for agent queries."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    include_metadata: bool = Field(default=False, description="Include processing metadata")


class QueryResponse(BaseModel):
    """Response model for agent queries."""
    response: str = Field(..., description="Agent's response")
    tools_used: list = Field(default_factory=list, description="Tools used by agent")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    agent_ready: bool = Field(..., description="Agent readiness status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")





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
        
        @self.app.get("/capabilities")
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
                return {
                    "capabilities": result["response"],
                    "tools_available": [tool.name for tool in self.agent.tools],
                    "agent_type": "react"
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get capabilities: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get capabilities: {str(e)}"
                )
        

        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "Second Brain AI Assistant",
                "version": "1.0.0",
                "description": "Intelligent assistant for your personal knowledge base",
                "endpoints": {
                    "health": "/health",
                    "query": "/query",
                    "capabilities": "/capabilities",
                    "docs": "/docs"
                }
            }
    

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