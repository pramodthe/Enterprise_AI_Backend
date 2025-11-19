from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.agents.hr_agent import get_hr_agent_response

router = APIRouter()

class EmployeeQuery(BaseModel):
    question: str

class EmployeeResponse(BaseModel):
    answer: str
    source: str = "HR Agent"

@router.post("/hr/query", response_model=EmployeeResponse)
async def query_hr_agent(query: EmployeeQuery):
    """
    Query the HR agent for employee information
    """
    try:
        response = get_hr_agent_response(query.question)
        return EmployeeResponse(answer=response, source="HR Agent")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing HR query: {str(e)}")

@router.get("/hr/health")
async def hr_agent_health():
    """
    Health check for HR agent
    """
    return {"status": "healthy", "agent": "HR Agent"}