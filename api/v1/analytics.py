from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.agents.analytics_agent import get_analytics_response

router = APIRouter()

class AnalyticsQuery(BaseModel):
    query: str

class AnalyticsResponse(BaseModel):
    result: str
    calculation_details: Dict[str, Any]

@router.post("/analytics/query", response_model=AnalyticsResponse)
async def query_analytics_agent(query: AnalyticsQuery):
    """
    Query the analytics agent for data analysis and calculations
    """
    try:
        result, details = get_analytics_response(query.query)
        return AnalyticsResponse(result=result, calculation_details=details)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing analytics query: {str(e)}")

@router.get("/analytics/health")
async def analytics_agent_health():
    """
    Health check for analytics agent
    """
    return {"status": "healthy", "agent": "Analytics Agent"}