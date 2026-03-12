"""Pydantic models for institutional actors. Matches parl8 schema."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Actor(BaseModel):
    """Actor data model for EU institutional actors (Commissioners, etc.).

    Field names aligned with meps table where applicable (parl8 compatibility).
    """

    actor_id: str = Field(..., description="Actor identifier")
    fullName: str = Field(..., description="Full name (aligned with meps.fullName)")

    actor_type: Optional[str] = None
    profile_url: Optional[str] = None
    image_url: Optional[str] = None
    role: Optional[str] = None
    country: Optional[str] = None
    portfolio: Optional[str] = None
    term_start: Optional[str] = None
    term_end: Optional[str] = None
    contacts: Optional[Dict[str, Any]] = None
    responsibilities: Optional[Dict[str, List[str]]] = None
    speeches: Optional[List[Dict[str, Any]]] = None
    latest_news: Optional[List[Dict[str, Any]]] = None
    calendar: Optional[List[Dict[str, Any]]] = None
    transparency: Optional[Dict[str, Any]] = None
    biography: Optional[List[Dict[str, str]]] = None
    documents: Optional[List[Dict[str, Any]]] = None

    team: List[Dict[str, Any]] = Field(default_factory=list)
    declarations: List[Dict[str, Any]] = Field(default_factory=list)
    past_meetings: List[Dict[str, Any]] = Field(default_factory=list)

    description: Optional[str] = None
    parliament: Optional[str] = None

    # Gold AI fields (kept for parl8 compatibility, will be null)
    role_summary: Optional[str] = None
    key_topics: Optional[List[str]] = None
    declarations_summary: Optional[str] = None
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None

    status: str = Field(default="active")

    class Config:
        extra = "allow"
