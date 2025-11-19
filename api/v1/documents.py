from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.agents.document_agent import get_document_response, process_document_upload

router = APIRouter()

class DocumentQuery(BaseModel):
    query: str

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str

class DocumentResponse(BaseModel):
    answer: str
    source_documents: List[str]

@router.post("/documents/query", response_model=DocumentResponse)
async def query_document_agent(query: DocumentQuery):
    """
    Query the document agent for information from company documents
    """
    try:
        answer, sources = get_document_response(query.query)
        return DocumentResponse(answer=answer, source_documents=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document query: {str(e)}")

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the knowledge base
    """
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.doc', '.docx', '.txt', '.md']
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File type {file_ext} not supported. Allowed types: {', '.join(allowed_extensions)}")
        
        doc_id = process_document_upload(file)
        return {"message": f"Document {file.filename} uploaded successfully", "document_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error uploading document: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log to console
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@router.get("/documents/health")
async def document_agent_health():
    """
    Health check for document agent
    """
    return {"status": "healthy", "agent": "Document Agent"}