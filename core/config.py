from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Anthropic API Key (for Claude models via Anthropic directly)
    api_key: Optional[str] = os.getenv("api_key")
    
    # AWS Credentials for Bedrock (alternative to Anthropic API)
    aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    
    # Use Bedrock instead of Anthropic API
    use_bedrock: bool = os.getenv("USE_BEDROCK", "False").lower() == "true"
    
    # Bedrock model ID for Claude
    bedrock_model_id: str = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-v1:0")
    
    # Document processing embeddings (Bedrock or OpenAI)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./enterprise_ai.db")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "documents")
    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    # Supabase
    supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
    supabase_key: Optional[str] = os.getenv("SUPABASE_KEY")
    supabase_service_role_key: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    # Observability - Opik
    enable_tracing: bool = os.getenv("ENABLE_TRACING", "False").lower() == "true"
    opik_api_key: Optional[str] = os.getenv("OPIK_API_KEY")
    opik_workspace: Optional[str] = os.getenv("OPIK_WORKSPACE")
    
    # Application
    app_name: str = "Enterprise AI Assistant Platform"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    api_v1_prefix: str = "/api/v1"
    
    # Agent settings
    default_model: str = os.getenv("DEFAULT_MODEL", "anthropic.claude-sonnet-v1:0")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1028"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
    
    # Agent URLs
    employee_mcp_url: Optional[str] = os.getenv("EMPLOYEE_MCP_URL")
    employee_agent_url: Optional[str] = os.getenv("EMPLOYEE_AGENT_URL")
    hr_agent_port: int = int(os.getenv("HR_AGENT_PORT", "8000"))
    analytics_mcp_url: Optional[str] = os.getenv("ANALYTICS_MCP_URL")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()