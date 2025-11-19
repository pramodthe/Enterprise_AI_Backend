"""
HR Agent for the Enterprise AI Assistant Platform
Adapted from the existing A2A system in the codebase
"""
import os
from urllib.parse import urlparse
import logging
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Import Opik tracing utilities
from backend.core.opik_config import is_tracing_enabled, get_opik_metadata, get_session_id

# Configure logging
logger = logging.getLogger(__name__)

# Check if using Bedrock or direct Anthropic API
if os.getenv("USE_BEDROCK", "False").lower() == "true":
    # Use AWS Bedrock
    from strands import Agent
    from strands.models.bedrock import BedrockModel
    from strands.multiagent.a2a import A2AServer
    from strands.tools.mcp.mcp_client import MCPClient
    from mcp.client.streamable_http import streamablehttp_client

    # Use Bedrock model
    import boto3
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )
    model = BedrockModel(
        client=bedrock_runtime,
        max_tokens=1028,
        model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-v1:0"),
        temperature=0.3
    )
else:
    # Use Anthropic model (original implementation)
    from strands import Agent
    from strands.models.anthropic import AnthropicModel
    from strands.multiagent.a2a import A2AServer
    from strands.tools.mcp.mcp_client import MCPClient
    from mcp.client.streamable_http import streamablehttp_client

    model = AnthropicModel(
        client_args={
            "api_key": os.getenv("api_key"),  # Required API key
        },
        max_tokens=1028,
        model_id=os.getenv("DEFAULT_MODEL", "claude-3-7-sonnet-20250219"),  # Using the same model from original code
        params={
            "temperature": 0.3,
        }
    )

# Configuration - using the same pattern as the existing A2A system
EMPLOYEE_INFO_URL = os.getenv("EMPLOYEE_MCP_URL", "http://localhost:8002/mcp/")
EMPLOYEE_AGENT_URL = os.getenv("EMPLOYEE_AGENT_URL", "http://localhost:8001/")
HR_AGENT_PORT = int(os.getenv("HR_AGENT_PORT", "8000"))

def _get_hr_agent_response_impl(question: str) -> str:
    """
    Internal implementation of HR agent response (without tracing decorator).
    """
    # System prompt for HR agent with sample data
    hr_system_prompt = """
    You are an HR assistant that helps with employee queries.
    
    You have access to the following sample employee data:
    
    Employees:
    - John Smith (ID: E001) - Senior Software Engineer, Skills: Python, JavaScript, AWS
    - Sarah Johnson (ID: E002) - Product Manager, Skills: Agile, Product Strategy, User Research
    - Michael Chen (ID: E003) - Data Scientist, Skills: Python, Machine Learning, SQL
    - Emily Davis (ID: E004) - UX Designer, Skills: Figma, User Research, Prototyping
    - David Wilson (ID: E005) - DevOps Engineer, Skills: Kubernetes, Docker, CI/CD
    
    Organization Structure:
    - CEO: Robert Anderson
      - CTO: Jennifer Lee (manages: John Smith, David Wilson)
      - CPO: Mark Thompson (manages: Sarah Johnson, Emily Davis)
      - Head of Data: Lisa Brown (manages: Michael Chen)
    
    When asked about employees, skills, or organizational structure, 
    use this information to provide accurate responses.
    Be professional and concise in your responses.
    """
    
    # Create HR agent without MCP tools for now
    hr_agent = Agent(
        model=model,
        name="HR Assistant",
        description="Answers HR-related questions about employees",
        system_prompt=hr_system_prompt
    )
    
    # Process the question and return response
    response = hr_agent(question)
    return str(response)


def get_hr_agent_response(question: str) -> str:
    """
    Get response from HR agent for a specific question.
    
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
            
            # Add HR agent specific metadata
            trace_metadata = {
                **base_metadata,
                "model_provider": model_provider,
                "model_id": model_id,
                "agent_type": "hr"
            }
            
            # Apply the track decorator dynamically
            @track(
                name="hr_agent_query",
                tags=["agent:hr"],
                metadata=trace_metadata
            )
            def traced_hr_response(question: str) -> str:
                return _get_hr_agent_response_impl(question)
            
            return traced_hr_response(question)
            
        except ImportError:
            logger.debug("Opik package not available. Running without tracing.")
            return _get_hr_agent_response_impl(question)
        except Exception as e:
            logger.warning(
                f"Failed to apply tracing to HR agent: {str(e)}. "
                "Running without tracing."
            )
            return _get_hr_agent_response_impl(question)
    else:
        # Tracing is disabled, run without tracing
        return _get_hr_agent_response_impl(question)

# For direct usage
def create_hr_a2a_server():
    """
    Create an A2A server for the HR agent
    """
    # Create MCP client for employee data
    def create_mcp_transport():
        return streamablehttp_client(EMPLOYEE_INFO_URL)

    mcp_client = MCPClient(create_mcp_transport)
    
    # System prompt for HR agent
    hr_system_prompt = """
    You are an HR assistant that helps with employee queries.
    You have access to employee data through tools.
    When asked about employees, skills, or organizational structure, 
    use the available tools to find accurate information.
    Be professional and concise in your responses.
    """
    
    with mcp_client:
        # Get tools from MCP server
        tools = mcp_client.list_tools_sync()
        
        # Create HR agent
        hr_agent = Agent(
            model=model,
            name="HR Assistant",
            description="Answers HR-related questions about employees",
            tools=tools,
            system_prompt=hr_system_prompt
        )
        
        # Create A2A server
        a2a_server = A2AServer(
            agent=hr_agent, 
            host=urlparse(EMPLOYEE_AGENT_URL).hostname, 
            port=int(urlparse(EMPLOYEE_AGENT_URL).port)
        )
        
        return a2a_server