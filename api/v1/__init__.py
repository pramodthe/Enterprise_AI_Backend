from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from backend.core.security import get_current_user
from backend.database.models.user import User

router = APIRouter()

# Import agent functions from their respective modules
from backend.agents.hr_agent import get_hr_agent_response
from backend.agents.analytics_agent import get_analytics_response
from backend.agents.document_agent import get_document_response


class AgentQuery(BaseModel):
    query: str
    agent: str = "auto"  # "hr", "analytics", "document", or "auto"


class AgentResponse(BaseModel):
    response: str
    agent_used: str
    source: str = None


@router.get("/", response_model=Dict[str, Any])
async def get_agents_info():
    """
    Get information about available agents
    """
    agents_info = {
        "agents": [
            {
                "name": "HR Assistant",
                "description": "Handles employee, skills, and organizational queries",
                "capabilities": ["employee_lookup", "skills_matching", "org_chart"]
            },
            {
                "name": "Analytics Assistant",
                "description": "Performs calculations and data analysis",
                "capabilities": ["calculations", "data_analysis", "business_metrics"]
            },
            {
                "name": "Document Assistant",
                "description": "Searches and retrieves information from company documents",
                "capabilities": ["document_search", "knowledge_retrieval", "content_extraction"]
            }
        ]
    }
    return agents_info


@router.post("/query", response_model=AgentResponse)
async def query_agent(agent_query: AgentQuery):
    """
    Route query to appropriate agent based on request
    """
    query = agent_query.query
    requested_agent = agent_query.agent.lower()

    try:
        if requested_agent == "hr" or ("employee" in query.lower() or "hr" in query.lower() or "skill" in query.lower()):
            # Route to HR agent
            response = get_hr_agent_response(query)
            return AgentResponse(
                response=response,
                agent_used="HR Assistant",
                source="Employee Data MCP Server"
            )

        elif requested_agent == "analytics" or any(word in query.lower() for word in ["calculate", "compute", "analyze", "average", "percentage", "sum", "total"]):
            # Route to analytics agent
            response, details = get_analytics_response(query)
            return AgentResponse(
                response=response,
                agent_used="Analytics Assistant",
                source="Analytics MCP Server"
            )

        elif requested_agent == "document" or any(word in query.lower() for word in ["policy", "document", "manual", "procedure", "handbook"]):
            # Route to document agent
            response, sources = get_document_response(query)
            return AgentResponse(
                response=response,
                agent_used="Document Assistant",
                source=f"Documents: {sources}"
            )

        else:
            # Default to HR agent for general queries
            response = get_hr_agent_response(query)
            return AgentResponse(
                response=response,
                agent_used="HR Assistant",  # Default to HR for general queries
                source="Employee Data MCP Server"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing agent query: {str(e)}")


@router.get("/health")
async def agents_health():
    """
    Health check for all agents
    """
    return {
        "status": "healthy",
        "service": "Agents Service",
        "available_agents": ["HR", "Analytics", "Document"]
    }