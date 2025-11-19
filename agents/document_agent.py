"""
Document Agent for the Enterprise AI Assistant Platform
Using RAG system adapted from the restaurant example
"""
import os
import tempfile
import time
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient
from langchain_aws import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings

# Import Opik tracing utilities
from backend.core.opik_config import is_tracing_enabled, get_opik_metadata, get_session_id

# Configure logging
logger = logging.getLogger(__name__)

# Check if using Bedrock or direct Anthropic API
if os.getenv("USE_BEDROCK", "False").lower() == "true":
    # Use AWS Bedrock
    from strands import Agent
    from strands.models.bedrock import BedrockModel
else:
    from strands import Agent
    from strands.models.anthropic import AnthropicModel

# Initialize model based on configuration
if os.getenv("USE_BEDROCK", "False").lower() == "true":
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

# Check which embedding provider to use
if os.getenv("AWS_ACCESS_KEY_ID"):
    # Use AWS Bedrock embeddings
    import boto3
    bedrock_client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    embedding_model = BedrockEmbeddings(
        client=bedrock_client,
        model_id='amazon.titan-embed-text-v1'
    )
elif os.getenv("OPENAI_API_KEY"):
    # Use OpenAI embeddings
    embedding_model = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
else:
    # Default to a simple local embedding (for testing)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings()

# Set up vector database
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "documents")
vector_db: VectorStore = None

def initialize_vector_db():
    """Initialize the vector database"""
    global vector_db
    client = QdrantClient(
        url=qdrant_url,
        # If using Qdrant cloud, you'd also provide an API key:
        # api_key=os.getenv("QDRANT_API_KEY")
    )

    vector_db = Qdrant(
        client=client,
        collection_name=qdrant_collection_name,
        embeddings=embedding_model,
    )

# Initialize the vector database at startup
initialize_vector_db()

def clean_text(text: str) -> str:
    """Clean text by removing or replacing problematic control characters"""
    import re
    # Replace control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_document(file_path: str):
    """Load a document based on its file type"""
    file_extension = Path(file_path).suffix.lower()

    if file_extension == '.pdf':
        docs = PyPDFLoader(file_path).load()
    elif file_extension == '.docx':
        docs = Docx2txtLoader(file_path).load()
    elif file_extension in ['.txt', '.md']:
        docs = TextLoader(file_path, encoding='utf-8').load()
    else:
        # For other text-based formats
        docs = TextLoader(file_path, encoding='utf-8').load()
    
    # Clean the text content in each document
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    
    return docs


def _chunk_documents_impl(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """
    Internal implementation of document chunking (without tracing decorator).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)
    return splits


def chunk_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """
    Chunk documents into smaller pieces for embedding.
    
    This function is traced with Opik when tracing is enabled.
    """
    if is_tracing_enabled():
        try:
            from opik import track
            
            # Get common metadata
            base_metadata = get_opik_metadata()
            
            # Add chunking specific metadata
            trace_metadata = {
                **base_metadata,
                "operation": "chunking",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "agent_type": "document"
            }
            
            @track(
                name="document_chunking",
                tags=["agent:document", "operation:chunking"],
                metadata=trace_metadata
            )
            def traced_chunk_documents(documents: List, chunk_size: int, chunk_overlap: int) -> List:
                return _chunk_documents_impl(documents, chunk_size, chunk_overlap)
            
            return traced_chunk_documents(documents, chunk_size, chunk_overlap)
            
        except ImportError:
            logger.debug("Opik package not available. Running without tracing.")
            return _chunk_documents_impl(documents, chunk_size, chunk_overlap)
        except Exception as e:
            logger.warning(
                f"Failed to apply tracing to document chunking: {str(e)}. "
                "Running without tracing."
            )
            return _chunk_documents_impl(documents, chunk_size, chunk_overlap)
    else:
        return _chunk_documents_impl(documents, chunk_size, chunk_overlap)


def _generate_embeddings_impl(splits: List, embedding_model, url: str, collection_name: str) -> Qdrant:
    """
    Internal implementation of embedding generation and vector storage (without tracing decorator).
    """
    from qdrant_client.models import Distance, VectorParams
    
    # Create or get existing Qdrant client
    client = QdrantClient(url=url)
    
    # Check if collection exists, if not create it
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name not in collection_names:
        # Get embedding dimension by creating a sample embedding
        sample_embedding = embedding_model.embed_query("sample text")
        vector_size = len(sample_embedding)
        
        # Create the collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{collection_name}' with vector size {vector_size}")
    
    # Create Qdrant vector store instance
    vector_db = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding_model,
    )
    
    # Add documents to the collection
    vector_db.add_documents(splits)
    
    return vector_db


def generate_embeddings_and_store(splits: List, embedding_model, url: str, collection_name: str) -> Qdrant:
    """
    Generate embeddings for document chunks and store in vector database.
    
    This function is traced with Opik when tracing is enabled.
    """
    # Get embedding model name
    embedding_model_name = "unknown"
    if hasattr(embedding_model, 'model_id'):
        embedding_model_name = embedding_model.model_id
    elif hasattr(embedding_model, 'model'):
        embedding_model_name = embedding_model.model
    elif hasattr(embedding_model, 'model_name'):
        embedding_model_name = embedding_model.model_name
    
    if is_tracing_enabled():
        try:
            from opik import track
            
            # Get common metadata
            base_metadata = get_opik_metadata()
            
            # Add embedding specific metadata
            trace_metadata = {
                **base_metadata,
                "operation": "embedding",
                "embedding_model": embedding_model_name,
                "collection_name": collection_name,
                "chunk_count": len(splits),
                "agent_type": "document"
            }
            
            @track(
                name="embedding_generation",
                tags=["agent:document", "operation:embedding"],
                metadata=trace_metadata
            )
            def traced_generate_embeddings(splits: List, embedding_model, url: str, collection_name: str) -> Qdrant:
                return _generate_embeddings_impl(splits, embedding_model, url, collection_name)
            
            return traced_generate_embeddings(splits, embedding_model, url, collection_name)
            
        except ImportError:
            logger.debug("Opik package not available. Running without tracing.")
            return _generate_embeddings_impl(splits, embedding_model, url, collection_name)
        except Exception as e:
            logger.warning(
                f"Failed to apply tracing to embedding generation: {str(e)}. "
                "Running without tracing."
            )
            return _generate_embeddings_impl(splits, embedding_model, url, collection_name)
    else:
        return _generate_embeddings_impl(splits, embedding_model, url, collection_name)

def _process_document_upload_impl(file, content: bytes) -> str:
    """
    Internal implementation of document upload processing (without tracing decorator).
    """
    # Create a temporary file to save the upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        # Save the uploaded file to the temporary location
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Get file metadata
        file_size = len(content)
        file_type = Path(file.filename).suffix.lower()
        
        # Load the document
        documents = load_document(temp_file_path)

        # Split the documents into chunks using traced function
        splits = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)

        # Add to vector database
        client = QdrantClient(
            url=qdrant_url,
            # If using Qdrant cloud, you'd also provide an API key:
            # api_key=os.getenv("QDRANT_API_KEY")
        )

        # Add documents to the database using traced function
        vector_db = generate_embeddings_and_store(
            splits,
            embedding_model,
            qdrant_url,
            qdrant_collection_name
        )

        # Generate a document ID based on filename and timestamp
        doc_id = f"{Path(file.filename).stem}_{int(time.time())}"

        return doc_id
    finally:
        # Clean up the temporary file
        Path(temp_file_path).unlink()


def process_document_upload(file) -> str:
    """
    Process an uploaded document and add it to the knowledge base.
    
    This function is traced with Opik when tracing is enabled.
    """
    # Get file metadata for tracing
    filename = file.filename if hasattr(file, 'filename') else "unknown"
    
    # Read content to get file size
    if hasattr(file, 'file'):
        content = file.file.read()
        # Reset file pointer for processing
        file.file.seek(0)
    else:
        content = file.read()
        # Reset file pointer for processing
        file.seek(0)
    
    file_size = len(content)
    file_type = Path(filename).suffix.lower()
    
    # Determine model provider and model ID
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
            
            # Add document upload specific metadata
            trace_metadata = {
                **base_metadata,
                "model_provider": model_provider,
                "model_id": model_id,
                "agent_type": "document",
                "operation": "upload",
                "filename": filename,
                "file_size_bytes": file_size,
                "file_type": file_type
            }
            
            @track(
                name="document_upload",
                tags=["agent:document", "operation:upload"],
                metadata=trace_metadata
            )
            def traced_process_document_upload(file, content: bytes) -> str:
                return _process_document_upload_impl(file, content)
            
            return traced_process_document_upload(file, content)
            
        except ImportError:
            logger.debug("Opik package not available. Running without tracing.")
            return _process_document_upload_impl(file, content)
        except Exception as e:
            logger.warning(
                f"Failed to apply tracing to document upload: {str(e)}. "
                "Running without tracing."
            )
            return _process_document_upload_impl(file, content)
    else:
        return _process_document_upload_impl(file, content)

def _perform_similarity_search_impl(query: str, vector_db: VectorStore, k: int = 3) -> Tuple[List, List[str], List[float]]:
    """
    Internal implementation of similarity search (without tracing decorator).
    Returns documents, sources, and similarity scores.
    """
    # Perform similarity search with scores
    docs_with_scores = vector_db.similarity_search_with_score(query, k=k)
    
    docs = [doc for doc, score in docs_with_scores]
    scores = [float(score) for doc, score in docs_with_scores]
    sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
    
    return docs, sources, scores


def perform_similarity_search(query: str, vector_db: VectorStore, k: int = 3) -> Tuple[List, List[str], List[float]]:
    """
    Perform similarity search on vector database.
    
    This function is traced with Opik when tracing is enabled.
    """
    if is_tracing_enabled():
        try:
            from opik import track
            
            # Get common metadata
            base_metadata = get_opik_metadata()
            
            # Add retrieval specific metadata
            trace_metadata = {
                **base_metadata,
                "operation": "retrieval",
                "query": query,
                "k": k,
                "collection_name": qdrant_collection_name,
                "agent_type": "document"
            }
            
            @track(
                name="document_retrieval",
                tags=["agent:document", "operation:retrieval"],
                metadata=trace_metadata
            )
            def traced_similarity_search(query: str, vector_db: VectorStore, k: int) -> Tuple[List, List[str], List[float]]:
                return _perform_similarity_search_impl(query, vector_db, k)
            
            return traced_similarity_search(query, vector_db, k)
            
        except ImportError:
            logger.debug("Opik package not available. Running without tracing.")
            return _perform_similarity_search_impl(query, vector_db, k)
        except Exception as e:
            logger.warning(
                f"Failed to apply tracing to similarity search: {str(e)}. "
                "Running without tracing."
            )
            return _perform_similarity_search_impl(query, vector_db, k)
    else:
        return _perform_similarity_search_impl(query, vector_db, k)


def _get_document_response_impl(query: str) -> Tuple[str, List[str]]:
    """
    Internal implementation of document response (without tracing decorator).
    """
    global vector_db

    try:
        # Initialize Qdrant client to ensure latest documents are available
        client = QdrantClient(
            url=qdrant_url,
            # If using Qdrant cloud, you'd also provide an API key:
            # api_key=os.getenv("QDRANT_API_KEY")
        )

        # Create Qdrant vector store
        vector_db = Qdrant(
            client=client,
            collection_name=qdrant_collection_name,
            embeddings=embedding_model,
        )

        # Perform similarity search using traced function
        docs, sources, scores = perform_similarity_search(query, vector_db, k=3)

        # Format the context from retrieved documents
        context = ""
        for doc in docs:
            context += f"Document: {doc.metadata.get('source', 'Unknown')}\n"
            context += f"Content: {doc.page_content}\n\n"

        # System prompt for document agent
        document_system_prompt = f"""
        You are a document search assistant that helps find information in company documents.
        Your knowledge comes from the following context:

        {context}

        When answering questions:
        1. Base your answers on the provided context
        2. Be specific and cite relevant information
        3. If the information isn't available in the context, say so clearly
        4. Keep responses professional and concise
        """
    except Exception as e:
        # If Qdrant is not available, use a fallback
        logger.info(f"Qdrant connection failed: {e}. Using fallback mode.")
        context = """
        Sample Company Documents:
        
        Document: Employee Handbook
        - Work hours: 9 AM - 5 PM, Monday to Friday
        - Remote work policy: Hybrid model with 3 days in office
        - PTO: 20 days per year plus public holidays
        - Health insurance: Comprehensive coverage for employees and dependents
        
        Document: IT Security Policy
        - All devices must have encryption enabled
        - Use VPN when working remotely
        - Password requirements: 12+ characters, changed every 90 days
        - Report security incidents immediately to IT
        
        Document: Benefits Guide
        - 401(k) matching: Up to 6% of salary
        - Professional development: $2000 annual budget
        - Gym membership reimbursement: Up to $50/month
        - Parental leave: 16 weeks paid
        """
        sources = ["Employee Handbook", "IT Security Policy", "Benefits Guide"]
        
        document_system_prompt = f"""
        You are a document search assistant that helps find information in company documents.
        
        Note: The document database is currently not available. You have access to sample documents:
        
        {context}

        When answering questions:
        1. Base your answers on the provided sample context
        2. Be specific and cite relevant information
        3. If the information isn't available in the context, say so clearly
        4. Mention that this is sample data and the full document database is not currently connected
        5. Keep responses professional and concise
        """

    # Create document agent
    document_agent = Agent(
        model=model,
        name="Document Assistant",
        description="Finds information in company documents",
        system_prompt=document_system_prompt
    )

    # Process the query and return response
    response = document_agent(query)

    return str(response), sources


def get_document_response(query: str) -> Tuple[str, List[str]]:
    """
    Get response from document agent for a specific query.
    
    This function is traced with Opik when tracing is enabled.
    """
    # Determine model provider and model ID
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
            
            # Add document response specific metadata
            trace_metadata = {
                **base_metadata,
                "model_provider": model_provider,
                "model_id": model_id,
                "agent_type": "document",
                "operation": "query"
            }
            
            @track(
                name="document_query",
                tags=["agent:document", "operation:query"],
                metadata=trace_metadata
            )
            def traced_get_document_response(query: str) -> Tuple[str, List[str]]:
                return _get_document_response_impl(query)
            
            return traced_get_document_response(query)
            
        except ImportError:
            logger.debug("Opik package not available. Running without tracing.")
            return _get_document_response_impl(query)
        except Exception as e:
            logger.warning(
                f"Failed to apply tracing to document response: {str(e)}. "
                "Running without tracing."
            )
            return _get_document_response_impl(query)
    else:
        return _get_document_response_impl(query)