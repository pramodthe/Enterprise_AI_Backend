"""
Tests for AgentClient A2A communication.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import Timeout, RequestException
from backend.chatbot.agent_client import AgentClient
from backend.chatbot.models import AgentResponse


@pytest.fixture
def agent_client():
    """Create an AgentClient instance for testing."""
    return AgentClient(
        agent_url="http://localhost:8000",
        agent_name="test_agent",
        timeout=30,
        max_retries=3,
        backoff_factor=0.1  # Faster for testing
    )


def test_agent_client_initialization(agent_client):
    """Test AgentClient initialization."""
    assert agent_client.agent_url == "http://localhost:8000"
    assert agent_client.agent_name == "test_agent"
    assert agent_client.timeout == 30
    assert agent_client.max_retries == 3
    assert agent_client.session is not None


def test_agent_client_url_normalization():
    """Test that trailing slashes are removed from URLs."""
    client = AgentClient(
        agent_url="http://localhost:8000/",
        agent_name="test_agent"
    )
    assert client.agent_url == "http://localhost:8000"


@pytest.mark.anyio
async def test_query_success(agent_client):
    """Test successful query to agent."""
    # Mock the session.post method
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": "Test response from agent",
        "metadata": {"tokens": 100}
    }
    mock_response.elapsed.total_seconds.return_value = 0.5
    
    with patch.object(agent_client.session, 'post', return_value=mock_response):
        response = await agent_client.query("Test query")
        
        assert isinstance(response, AgentResponse)
        assert response.success is True
        assert response.content == "Test response from agent"
        assert response.agent_name == "test_agent"
        assert response.error is None
        assert response.metadata["tokens"] == 100
        assert response.metadata["response_time"] == 0.5


@pytest.mark.anyio
async def test_query_with_context(agent_client):
    """Test query with context information."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": "Contextual response",
        "metadata": {}
    }
    mock_response.elapsed.total_seconds.return_value = 0.3
    
    context = {"session_id": "123", "history": ["previous message"]}
    
    with patch.object(agent_client.session, 'post', return_value=mock_response) as mock_post:
        response = await agent_client.query("Test query", context=context)
        
        # Verify context was passed in the request
        call_args = mock_post.call_args
        assert call_args[1]['json']['context'] == context
        assert response.success is True


@pytest.mark.anyio
async def test_query_timeout_with_retry(agent_client):
    """Test query timeout triggers retry with exponential backoff."""
    # Mock timeout on first two attempts, success on third
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Success after retries"}
    mock_response.elapsed.total_seconds.return_value = 0.5
    
    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Timeout("Connection timeout")
        return mock_response
    
    with patch.object(agent_client.session, 'post', side_effect=side_effect):
        response = await agent_client.query("Test query")
        
        assert response.success is True
        assert response.content == "Success after retries"
        assert call_count == 3  # Should have retried twice


@pytest.mark.anyio
async def test_query_max_retries_exceeded(agent_client):
    """Test query fails after max retries."""
    with patch.object(agent_client.session, 'post', side_effect=Timeout("Connection timeout")):
        response = await agent_client.query("Test query")
        
        assert response.success is False
        assert response.error is not None
        assert "timed out" in response.error.lower()
        assert response.metadata["attempts"] == 3


@pytest.mark.anyio
async def test_query_request_exception(agent_client):
    """Test query handles RequestException."""
    with patch.object(agent_client.session, 'post', side_effect=RequestException("Network error")):
        response = await agent_client.query("Test query")
        
        assert response.success is False
        assert response.error is not None
        assert "failed after" in response.error.lower()


@pytest.mark.anyio
async def test_query_unexpected_exception(agent_client):
    """Test query handles unexpected exceptions."""
    with patch.object(agent_client.session, 'post', side_effect=ValueError("Unexpected error")):
        response = await agent_client.query("Test query")
        
        assert response.success is False
        assert response.error is not None
        assert "Unexpected error" in response.error


def test_is_available_health_endpoint(agent_client):
    """Test is_available with health endpoint."""
    mock_response = Mock()
    mock_response.status_code = 200
    
    with patch.object(agent_client.session, 'get', return_value=mock_response):
        assert agent_client.is_available() is True


def test_is_available_base_url_fallback(agent_client):
    """Test is_available falls back to base URL."""
    def side_effect(url, *args, **kwargs):
        if "/health" in url:
            raise RequestException("Health endpoint not found")
        mock_response = Mock()
        mock_response.status_code = 200
        return mock_response
    
    with patch.object(agent_client.session, 'get', side_effect=side_effect):
        assert agent_client.is_available() is True


def test_is_available_agent_down(agent_client):
    """Test is_available when agent is down."""
    with patch.object(agent_client.session, 'get', side_effect=RequestException("Connection refused")):
        assert agent_client.is_available() is False


def test_is_available_server_error(agent_client):
    """Test is_available with server error status."""
    mock_response = Mock()
    mock_response.status_code = 503
    
    with patch.object(agent_client.session, 'get', return_value=mock_response):
        assert agent_client.is_available() is False


def test_get_capabilities_success(agent_client):
    """Test get_capabilities returns capability list."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "capabilities": ["hr_queries", "employee_lookup", "org_chart"]
    }
    
    with patch.object(agent_client.session, 'get', return_value=mock_response):
        capabilities = agent_client.get_capabilities()
        
        assert len(capabilities) == 3
        assert "hr_queries" in capabilities
        assert "employee_lookup" in capabilities
        assert "org_chart" in capabilities


def test_get_capabilities_empty_response(agent_client):
    """Test get_capabilities with empty response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    
    with patch.object(agent_client.session, 'get', return_value=mock_response):
        capabilities = agent_client.get_capabilities()
        
        assert capabilities == []


def test_get_capabilities_request_failure(agent_client):
    """Test get_capabilities handles request failure."""
    with patch.object(agent_client.session, 'get', side_effect=RequestException("Connection error")):
        capabilities = agent_client.get_capabilities()
        
        assert capabilities == []


def test_get_capabilities_unexpected_error(agent_client):
    """Test get_capabilities handles unexpected errors."""
    with patch.object(agent_client.session, 'get', side_effect=ValueError("Unexpected")):
        capabilities = agent_client.get_capabilities()
        
        assert capabilities == []


def test_context_manager(agent_client):
    """Test AgentClient as context manager."""
    with patch.object(agent_client, 'close') as mock_close:
        with agent_client as client:
            assert client is agent_client
        
        mock_close.assert_called_once()


def test_close(agent_client):
    """Test close method."""
    mock_session = Mock()
    agent_client.session = mock_session
    
    agent_client.close()
    
    mock_session.close.assert_called_once()
