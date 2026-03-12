"""Pydantic models for MEP data validation."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Member(BaseModel):
    """Member data model used across all pipeline stages.

    Field names match parl8 for backwards compatibility.
    """

    mepid: str = Field(..., description="MEP identifier")
    full_name: str = Field(..., description="Full name")
    country: str = Field(..., description="Country code")
    political_group: str = Field(..., description="Political group")
    status: str = Field(default="active", description="MEP status: active or inactive")

    national_party: Optional[str] = Field(None)
    profile_url: Optional[str] = Field(None)
    image_url: Optional[str] = Field(None)
    role: Optional[str] = Field(None)
    birth_date: Optional[str] = Field(None)
    birth_place: Optional[str] = Field(None)
    parliament: Optional[str] = Field(None)

    socials: Dict[str, str] = Field(default_factory=dict)
    navigation_links: Dict[str, str] = Field(default_factory=dict)
    committees: List[Dict[str, Any]] = Field(default_factory=list)
    contacts: List[Dict[str, Any]] = Field(default_factory=list)
    cv: List[Dict[str, Any]] = Field(default_factory=list)
    assistants: List[Dict[str, Any]] = Field(default_factory=list)
    declarations: List[Dict[str, Any]] = Field(default_factory=list)
    past_meetings: List[Dict[str, Any]] = Field(default_factory=list)

    # Gold AI fields (kept for parl8 compatibility, will be null)
    speech_summary: Optional[str] = Field(None)
    speech_top_words: List[Dict[str, Any]] = Field(default_factory=list)
    speech_sources: List[Dict[str, str]] = Field(default_factory=list)
    declarations_summary: Optional[str] = Field(None)
    embedding: Optional[List[float]] = Field(None)

    class Config:
        extra = "allow"
