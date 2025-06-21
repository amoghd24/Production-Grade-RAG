"""
Agent request/response models for the Second Brain AI Assistant.
This module defines all Pydantic models used for the agent's FastAPI endpoints.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for agent queries."""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="User query to process"
    )
    include_metadata: bool = Field(
        default=False, 
        description="Include processing metadata in response"
    )


class QueryResponse(BaseModel):
    """Response model for agent queries."""
    response: str = Field(..., description="Agent's response to the query")
    tools_used: List[str] = Field(
        default_factory=list, 
        description="List of tools used by the agent"
    )
    processing_time_ms: Optional[float] = Field(
        None, 
        description="Processing time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional processing metadata"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service health status")
    agent_ready: bool = Field(..., description="Agent readiness status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    details: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional health check details"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CapabilitiesResponse(BaseModel):
    """Response model for agent capabilities endpoint."""
    capabilities: str = Field(..., description="Description of agent capabilities")
    tools_available: List[str] = Field(
        default_factory=list,
        description="List of available tools"
    )
    agent_type: str = Field(..., description="Type of agent (e.g., 'react')")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class APIInfoResponse(BaseModel):
    """Response model for root API information endpoint."""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    endpoints: Dict[str, str] = Field(
        default_factory=dict,
        description="Available API endpoints"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 