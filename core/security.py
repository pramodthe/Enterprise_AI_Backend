from fastapi import HTTPException, status
from typing import Optional
import os


def get_current_user():
    # For this initial implementation, we'll just return a basic user
    # In a real application, you'd implement proper authentication
    api_key = os.getenv("api_key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key not configured",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"username": "enterprise_user", "id": "1"}