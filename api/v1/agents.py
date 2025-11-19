from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.agents.hr_agent import get_hr_agent_response
from backend.agents.analytics_agent import get_analytics_response
from backend.agents.document_agent import get_document_response
from backend.core.opik_config import is_tracing_enabled, get_opik_metadata
from backend.core.tracing_utils import log_error

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

class AgentStatus(BaseModel):
    agent_name: str
    status: str
    last_heartbeat: float
    capabilities: List[str]

class AgentListResponse(BaseModel):
    agents: List[AgentStatus]

@router.get("/agents/status", response_model=AgentListResponse)
async def get_all_agent_status():
    """
    Get status of all available agents
    """
    # In a real implementation, this would check actual agent status
    # For now, return a static status based on our implemented agents
    agents_status = [
        AgentStatus(
            agent_name="HR Assistant",
            status="running",
            last_heartbeat=0,
            capabilities=["employee_lookup", "skills_matching", "org_chart"]
        ),
        AgentStatus(
            agent_name="Analytics Assistant", 
            status="running",
            last_heartbeat=0,
            capabilities=["calculations", "data_analysis", "business_metrics"]
        ),
        AgentStatus(
            agent_name="Document Assistant",
            status="running", 
            last_heartbeat=0,
            capabilities=["document_search", "knowledge_retrieval", "content_extraction"]
        )
    ]
    return AgentListResponse(agents=agents_status)

class UnifiedQuery(BaseModel):
    query: str
    agent: str = "auto"

class UnifiedResponse(BaseModel):
    response: str
    agent_used: str
    confidence: float


def _calculate_keyword_scores_impl(query: str) -> Dict[str, int]:
    """
    Internal implementation of keyword score calculation (without tracing decorator).
    
    Args:
        query: The user query to analyze
        
    Returns:
        Dictionary with scores for each agent type
    """
    query_lower = query.lower()
    
    # Simple keyword-based routing
    # HR keywords
    hr_keywords = ["employee", "staff", "hire", "skill", "team", "org chart", "organization", "who", "person", "people"]
    # Analytics keywords
    analytics_keywords = ["calculate", "compute", "average", "sum", "percentage", "payroll", "wage", "tax", "salary", "multiply", "divide", "add", "subtract"]
    # Document keywords
    document_keywords = ["policy", "document", "handbook", "procedure", "guideline", "search", "find information", "what is our"]
    
    # Count keyword matches
    hr_score = sum(1 for keyword in hr_keywords if keyword in query_lower)
    analytics_score = sum(1 for keyword in analytics_keywords if keyword in query_lower)
    document_score = sum(1 for keyword in document_keywords if keyword in query_lower)
    
    scores = {
        "hr": hr_score,
        "analytics": analytics_score,
        "document": document_score
    }
    
    logger.debug(f"Keyword scores - HR: {hr_score}, Analytics: {analytics_score}, Document: {document_score}")
    
    return scores


def calculate_keyword_scores(query: str) -> Dict[str, int]:
    """
    Calculate keyword matching scores for each agent type.
    
    This function is traced with Opik when tracing is enabled.
    
    Args:
        query: The user query to analyze
        
    Returns:
        Dictionary with scores for each agent type
    """
    if is_tracing_enabled():
        try:
            from opik import track
            
            # Get common metadata
            base_metadata = get_opik_metadata()
            
            trace_metadata = {
                **base_metadata,
                "operation": "keyword_matching"
            }
            
            @track(
                name="keyword_matching",
                tags=["router:matching"],
                metadata=trace_metadata
            )
            def traced_keyword_scores(query: str) -> Dict[str, int]:
                return _calculate_keyword_scores_impl(query)
            
            return traced_keyword_scores(query)
            
        except ImportError:
            logger.debug("Opik package not available. Running without tracing.")
            return _calculate_keyword_scores_impl(query)
        except Exception as e:
            logger.warning(
                f"Failed to apply tracing to keyword matching: {str(e)}. "
                "Running without tracing."
            )
            return _calculate_keyword_scores_impl(query)
    else:
        return _calculate_keyword_scores_impl(query)


def _select_agent_impl(scores: Dict[str, int], specified_agent: str) -> tuple[str, float]:
    """
    Internal implementation of agent selection (without tracing decorator).
    
    Args:
        scores: Dictionary with keyword scores for each agent type
        specified_agent: The agent specified by the user ("auto" for automatic routing)
        
    Returns:
        Tuple of (selected_agent, confidence_score)
    """
    # If agent is specified, use it with confidence 1.0
    if specified_agent != "auto":
        logger.debug(f"Using specified agent: {specified_agent}")
        return specified_agent, 1.0
    
    # Auto-route based on keyword matching
    hr_score = scores.get("hr", 0)
    analytics_score = scores.get("analytics", 0)
    document_score = scores.get("document", 0)
    
    max_score = max(hr_score, analytics_score, document_score)
    
    if max_score == 0:
        # No clear match, default to document agent for general queries
        logger.debug("No keyword match, using Document agent")
        return "document", 0.5
    elif hr_score == max_score:
        logger.debug("Using HR agent based on keywords")
        confidence = hr_score / (hr_score + analytics_score + document_score + 1)
        return "hr", round(confidence, 2)
    elif analytics_score == max_score:
        logger.debug("Using Analytics agent based on keywords")
        confidence = analytics_score / (hr_score + analytics_score + document_score + 1)
        return "analytics", round(confidence, 2)
    else:
        logger.debug("Using Document agent based on keywords")
        confidence = document_score / (hr_score + analytics_score + document_score + 1)
        return "document", round(confidence, 2)


def select_agent(scores: Dict[str, int], specified_agent: str) -> tuple[str, float]:
    """
    Select the appropriate agent based on keyword scores.
    
    This function is traced with Opik when tracing is enabled.
    
    Args:
        scores: Dictionary with keyword scores for each agent type
        specified_agent: The agent specified by the user ("auto" for automatic routing)
        
    Returns:
        Tuple of (selected_agent, confidence_score)
    """
    if is_tracing_enabled():
        try:
            from opik import track
            
            # Get common metadata
            base_metadata = get_opik_metadata()
            
            trace_metadata = {
                **base_metadata,
                "operation": "agent_selection",
                "keyword_scores": scores,
                "specified_agent": specified_agent
            }
            
            @track(
                name="agent_selection",
                tags=["router:selection"],
                metadata=trace_metadata
            )
            def traced_agent_selection(scores: Dict[str, int], specified_agent: str) -> tuple[str, float]:
                return _select_agent_impl(scores, specified_agent)
            
            return traced_agent_selection(scores, specified_agent)
            
        except ImportError:
            logger.debug("Opik package not available. Running without tracing.")
            return _select_agent_impl(scores, specified_agent)
        except Exception as e:
            logger.warning(
                f"Failed to apply tracing to agent selection: {str(e)}. "
                "Running without tracing."
            )
            return _select_agent_impl(scores, specified_agent)
    else:
        return _select_agent_impl(scores, specified_agent)


def _query_all_agents_impl(request: UnifiedQuery) -> UnifiedResponse:
    """
    Internal implementation of query routing (without tracing decorator).
    
    Args:
        request: The unified query request
        
    Returns:
        UnifiedResponse with the agent's response
    """
    logger.info(f"Received query: {request.query}")
    
    try:
        # Calculate keyword scores
        scores = calculate_keyword_scores(request.query)
        
        # Select agent based on scores
        selected_agent, confidence = select_agent(scores, request.agent)
        
        # Route to the selected agent
        if selected_agent == "hr":
            response = get_hr_agent_response(request.query)
            logger.info(f"HR response: {response[:100]}...")
            return UnifiedResponse(response=response, agent_used="HR Assistant", confidence=confidence)
        elif selected_agent == "analytics":
            response, _ = get_analytics_response(request.query)
            logger.info(f"Analytics response: {response[:100]}...")
            return UnifiedResponse(response=response, agent_used="Analytics Assistant", confidence=confidence)
        elif selected_agent == "document":
            response, _ = get_document_response(request.query)
            logger.info(f"Document response: {response[:100]}...")
            return UnifiedResponse(response=response, agent_used="Document Assistant", confidence=confidence)
        else:
            # Fallback to document agent
            logger.warning(f"Unknown agent selected: {selected_agent}. Falling back to Document agent.")
            response, _ = get_document_response(request.query)
            return UnifiedResponse(response=response, agent_used="Document Assistant", confidence=0.5)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error in query_all_agents: {error_trace}")
        
        # Log error to trace if tracing is enabled
        log_error(e, context={
            "query": request.query,
            "specified_agent": request.agent,
            "operation": "query_routing"
        })
        
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/agents/query", response_model=UnifiedResponse)
async def query_all_agents(request: UnifiedQuery):
    """
    Route query to appropriate agent based on content.
    
    This function creates a parent trace for the entire request and routes
    the query to the appropriate agent based on keyword matching.
    """
    # Determine model provider and model ID (for metadata)
    use_bedrock = os.getenv("USE_BEDROCK", "False").lower() == "true"
    model_provider = "bedrock" if use_bedrock else "anthropic"
    
    if use_bedrock:
        model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-v1:0")
    else:
        model_id = os.getenv("DEFAULT_MODEL", "claude-3-7-sonnet-20250219")
    
    if is_tracing_enabled():
        try:
            from opik import track
            
            # Get common metadata
            base_metadata = get_opik_metadata()
            
            # Add query router specific metadata
            trace_metadata = {
                **base_metadata,
                "model_provider": model_provider,
                "model_id": model_id,
                "operation": "query_router",
                "query": request.query,
                "specified_agent": request.agent
            }
            
            @track(
                name="query_router",
                tags=["router"],
                metadata=trace_metadata
            )
            def traced_query_router(request: UnifiedQuery) -> UnifiedResponse:
                return _query_all_agents_impl(request)
            
            return traced_query_router(request)
            
        except ImportError:
            logger.debug("Opik package not available. Running without tracing.")
            return _query_all_agents_impl(request)
        except Exception as e:
            logger.warning(
                f"Failed to apply tracing to query router: {str(e)}. "
                "Running without tracing."
            )
            return _query_all_agents_impl(request)
    else:
        return _query_all_agents_impl(request)