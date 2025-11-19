"""
Analytics Agent for the Enterprise AI Assistant Platform
Using MCP calculator tools adapted from existing code
"""
import os
import threading
import time
import logging
from typing import Dict, Any, Optional

# Import Opik tracing utilities
from backend.core.opik_config import is_tracing_enabled, get_opik_metadata, get_session_id

# Configure logging
logger = logging.getLogger(__name__)

# Check if using Bedrock or direct Anthropic API
if os.getenv("USE_BEDROCK", "False").lower() == "true":
    # Use AWS Bedrock
    from strands import Agent
    from strands.models.bedrock import BedrockModel
    from strands.tools.mcp.mcp_client import MCPClient
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.server import FastMCP
else:
    from strands import Agent
    from strands.models.anthropic import AnthropicModel
    from strands.tools.mcp.mcp_client import MCPClient
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.server import FastMCP

# Initialize model based on configuration
if os.getenv("USE_BEDROCK", "False").lower() == "true":
    model = BedrockModel(
        client_args={
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        },
        max_tokens=1028,
        model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-v1:0"),
        params={"temperature": 0.3}
    )
else:
    model = AnthropicModel(
        client_args={"api_key": os.getenv("api_key")},  # Required API key
        max_tokens=1028,
        model_id=os.getenv("DEFAULT_MODEL", "claude-3-7-sonnet-20250219"),
        params={"temperature": 0.3}
    )

def start_analytics_mcp_server():
    """
    Start an MCP server with analytics tools (now includes a simplified US payroll tool).
    """
    mcp = FastMCP("Analytics Server")

    # -----------------------------
    # Basic calculation tools
    # -----------------------------
    @mcp.tool(description="Add two numbers together")
    def add(x: float, y: float) -> float:
        return x + y

    @mcp.tool(description="Subtract one number from another")
    def subtract(x: float, y: float) -> float:
        return x - y

    @mcp.tool(description="Multiply two numbers together")
    def multiply(x: float, y: float) -> float:
        return x * y

    @mcp.tool(description="Divide one number by another")
    def divide(x: float, y: float) -> float:
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y

    @mcp.tool(description="Calculate percentage change between two values")
    def percentage_change(old_value: float, new_value: float) -> float:
        if old_value == 0:
            raise ValueError("Cannot calculate percentage change from zero")
        return ((new_value - old_value) / old_value) * 100

    @mcp.tool(description="Calculate average of a list of numbers")
    def calculate_average(numbers: list) -> float:
        if not numbers:
            raise ValueError("Cannot calculate average of empty list")
        return sum(numbers) / len(numbers)

    # -----------------------------
    # Simplified US payroll tool
    # -----------------------------
    @mcp.tool(description="Calculate simple US payroll for full-time (salary) or part-time (hourly) employee")
    def simple_us_payroll(
        employee_type: str,              # "fulltime" or "parttime"
        hourly_rate: float = 0.0,        # used for part-time
        hours: float = 0.0,              # used for part-time
        annual_salary: float = 0.0,      # used for full-time
        pay_periods_per_year: int = 26,  # e.g. 26 = biweekly, 12 = monthly
        federal_tax_rate: float = 0.12,  # 12% federal
        state_tax_rate: float = 0.05,    # 5% state
        pre_tax_deductions: float = 0.0, # e.g. 401k
        post_tax_deductions: float = 0.0 # e.g. union fees
    ) -> dict:
        """
        Simplified US payroll calculator.
        Calculates gross pay, employee taxes (federal, state, SS, Medicare),
        net pay, and employer match (SS, Medicare).
        """

        et = employee_type.lower().strip()
        if et not in ("fulltime", "parttime"):
            raise ValueError("employee_type must be 'fulltime' or 'parttime'")

        if et == "fulltime":
            if annual_salary <= 0:
                raise ValueError("annual_salary must be > 0 for fulltime")
            gross_pay = annual_salary / pay_periods_per_year
        else:  # parttime
            if hourly_rate <= 0 or hours <= 0:
                raise ValueError("hourly_rate and hours must be > 0 for parttime")
            gross_pay = hourly_rate * hours

        taxable_wages = max(0.0, gross_pay - pre_tax_deductions)

        # Employee taxes
        federal_tax = taxable_wages * federal_tax_rate
        state_tax = taxable_wages * state_tax_rate
        social_security = taxable_wages * 0.062
        medicare = taxable_wages * 0.0145
        total_tax = federal_tax + state_tax + social_security + medicare

        # Net pay
        net_pay = gross_pay - pre_tax_deductions - total_tax - post_tax_deductions

        # Employer payroll taxes (match on FICA only)
        employer_ss = taxable_wages * 0.062
        employer_medicare = taxable_wages * 0.0145
        employer_total_cost = gross_pay + employer_ss + employer_medicare

        return {
            "employee_type": et,
            "gross_pay": round(gross_pay, 2),
            "taxable_wages": round(taxable_wages, 2),
            "federal_tax": round(federal_tax, 2),
            "state_tax": round(state_tax, 2),
            "social_security": round(social_security, 2),
            "medicare": round(medicare, 2),
            "total_employee_tax": round(total_tax, 2),
            "net_pay": round(net_pay, 2),
            "employer_ss": round(employer_ss, 2),
            "employer_medicare": round(employer_medicare, 2),
            "employer_total_cost": round(employer_total_cost, 2)
        }

    # Run the server with Streamable HTTP transport
    print("Starting Analytics MCP Server on http://localhost:8003")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8003)


def _get_analytics_response_impl(query: str) -> tuple[str, Dict[str, Any]]:
    """
    Internal implementation of analytics agent response (without tracing decorator).
    """
    analytics_system_prompt = """
    You are a business analytics assistant that can perform calculations and data analysis.

    You can help with:
    - Basic arithmetic (addition, subtraction, multiplication, division)
    - Percentage calculations and changes
    - Averages and statistical analysis
    - Simple US payroll what-ifs for full-time (salary) and part-time (hourly).
      Use the MCP tool `simple_us_payroll` when the user asks about wages vs taxes.

    When asked to perform calculations, work through them step by step and show your work.
    Explain the calculation and show the result clearly.
    For business analytics queries, provide insights along with the numbers.

    Examples:
    - "Full-time $78,000 annually, biweekly, federal 12%, state 5%, 401k $200 per period"
    - "Part-time $25/hr for 22 hours, federal 10%, state 4%"
    """
    
    # Try to connect to MCP server for tool access
    # If MCP server is not available, agent will work without tools
    try:
        analytics_mcp_url = os.getenv("ANALYTICS_MCP_URL", "http://localhost:8003/mcp/")
        
        def create_mcp_transport():
            return streamablehttp_client(analytics_mcp_url)
        
        mcp_client = MCPClient(create_mcp_transport)
        
        with mcp_client:
            # Get tools from MCP server
            tools = mcp_client.list_tools_sync()
            
            # Wrap tools with tracing if enabled
            if is_tracing_enabled():
                traced_tools = []
                for tool in tools:
                    # Create a traced wrapper for each tool
                    original_tool = tool
                    
                    def create_traced_tool(orig_tool):
                        """Create a traced version of a tool."""
                        try:
                            from opik import track
                            
                            # Get the original tool's call method
                            original_call = orig_tool.call if hasattr(orig_tool, 'call') else orig_tool
                            
                            @track(
                                name="mcp_tool_call",
                                tags=["tool:mcp", f"tool:{orig_tool.name if hasattr(orig_tool, 'name') else 'unknown'}"],
                                metadata={
                                    **get_opik_metadata(),
                                    "tool_name": orig_tool.name if hasattr(orig_tool, 'name') else 'unknown',
                                    "agent_type": "analytics"
                                }
                            )
                            def traced_tool_call(*args, **kwargs):
                                """Traced wrapper for tool call."""
                                return original_call(*args, **kwargs)
                            
                            # Replace the call method with traced version
                            if hasattr(orig_tool, 'call'):
                                orig_tool.call = traced_tool_call
                            
                            return orig_tool
                        except Exception as e:
                            logger.debug(f"Failed to wrap tool with tracing: {str(e)}")
                            return orig_tool
                    
                    traced_tools.append(create_traced_tool(original_tool))
                
                tools = traced_tools
            
            # Create agent with MCP tools
            analytics_agent = Agent(
                model=model,
                name="Analytics Assistant",
                description="Performs business calculations and data analysis",
                system_prompt=analytics_system_prompt,
                tools=tools
            )
            
            response = analytics_agent(query)
            details = {"query": query, "timestamp": time.time(), "tools_available": len(tools)}
            return str(response), details
            
    except Exception as e:
        # If MCP connection fails, create agent without tools
        logger.debug(f"Could not connect to MCP server: {str(e)}. Running without tools.")
        
        analytics_agent = Agent(
            model=model,
            name="Analytics Assistant",
            description="Performs business calculations and data analysis",
            system_prompt=analytics_system_prompt
        )
        
        response = analytics_agent(query)
        details = {"query": query, "timestamp": time.time(), "tools_available": 0}
        return str(response), details


def get_analytics_response(query: str) -> tuple[str, Dict[str, Any]]:
    """
    Get response from analytics agent for a specific query.
    
    This function is traced with Opik when tracing is enabled.
    """
    # Determine model provider and model ID
    use_bedrock = os.getenv("USE_BEDROCK", "False").lower() == "true"
    model_provider = "bedrock" if use_bedrock else "anthropic"
    
    if use_bedrock:
        model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-v1:0")
    else:
        model_id = os.getenv("DEFAULT_MODEL", "claude-3-7-sonnet-20250219")
    
    # Check if tracing is enabled
    if is_tracing_enabled():
        try:
            from opik import track
            
            # Get common metadata
            base_metadata = get_opik_metadata()
            
            # Add Analytics agent specific metadata
            trace_metadata = {
                **base_metadata,
                "model_provider": model_provider,
                "model_id": model_id,
                "agent_type": "analytics"
            }
            
            # Apply the track decorator dynamically
            @track(
                name="analytics_agent_query",
                tags=["agent:analytics"],
                metadata=trace_metadata
            )
            def traced_analytics_response(query: str) -> tuple[str, Dict[str, Any]]:
                return _get_analytics_response_impl(query)
            
            return traced_analytics_response(query)
            
        except ImportError:
            logger.debug("Opik package not available. Running without tracing.")
            return _get_analytics_response_impl(query)
        except Exception as e:
            logger.warning(
                f"Failed to apply tracing to Analytics agent: {str(e)}. "
                "Running without tracing."
            )
            return _get_analytics_response_impl(query)
    else:
        # Tracing is disabled, run without tracing
        return _get_analytics_response_impl(query)
