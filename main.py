from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import uuid

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.config import settings
from backend.api.v1 import agents, hr, analytics, documents

# Create FastAPI app
app = FastAPI(
    title="Enterprise AI Assistant Platform",
    description="A comprehensive AI assistant platform for enterprise use",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(agents.router, prefix="/api/v1", tags=["agents"])
app.include_router(hr.router, prefix="/api/v1", tags=["hr"])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])

@app.get("/")
async def root():
    return {"message": "Enterprise AI Assistant Platform API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Enterprise AI Assistant Platform"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)