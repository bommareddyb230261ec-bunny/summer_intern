from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class UserResponse(BaseModel):
    """Public representation of an authenticated Google user."""

    id: int
    google_id: str
    email: str
    name: str
    picture: str | None = None
    email_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: datetime

    model_config = ConfigDict(from_attributes=True)


class AuthResponse(BaseModel):
    """JSON payload returned after a successful OAuth callback."""

    message: str
    user: UserResponse
    access_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    """Small reusable response schema for simple API messages."""

    message: str


class ErrorResponse(BaseModel):
    """Consistent shape for documented API errors."""

    detail: str


class JobResponse(BaseModel):
    job_id: str
    message: str


class ProcessStartRequest(BaseModel):
    job_id: str


class ProcessStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    stage: str
    message: str
    result: Any | None = None


class ResultItem(BaseModel):
    face_id: str
    label: str
    similarity: float
    timestamp: str
    matched_face_image: str | None = None
    frame_name: str | None = None
    bounding_box: Any | None = None


class ResultsResponse(BaseModel):
    job_id: str
    status: str
    results: list[ResultItem]
