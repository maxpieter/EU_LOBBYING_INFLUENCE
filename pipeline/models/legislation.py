"""Pydantic models for legislation data. Matches parl8 schema."""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProcedureEvent(BaseModel):
    """Timeline event within a legislative procedure (stored in JSONB)."""

    event_id: str
    event_date: Optional[date] = None
    event_type: str
    activity_type: str
    title: Optional[str] = None
    description: Optional[str] = None
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    summary_text: Optional[str] = None
    parliament_code: Optional[str] = None
    source: Optional[str] = None


class ProcedureActor(BaseModel):
    """Actor involved in a legislative procedure (stored in JSONB)."""

    actor_type: str
    role: str
    mep_id: Optional[int] = None
    mep_name: Optional[str] = None
    committee_code: Optional[str] = None
    committee_name: Optional[str] = None
    institution_name: Optional[str] = None
    commissioner_name: Optional[str] = None
    is_active: bool = True
    name_resolution: Optional[str] = None


class Procedure(BaseModel):
    """Legislative procedure data model for OEIL procedures.

    Follows parl8 medallion architecture for compatibility.
    """

    # Bronze layer
    id: str = Field(..., description="OEIL reference (e.g., '2024/0003(COD)')")
    process_id: str
    title: str
    procedure_type: str

    status: Optional[str] = None
    stage: Optional[str] = None
    api_uri: Optional[str] = None

    # Silver layer
    description: Optional[str] = None
    legal_basis: List[str] = Field(default_factory=list)
    policy_area: Optional[str] = None
    subjects: List[str] = Field(default_factory=list)

    proposal_date: Optional[date] = None
    last_activity_date: Optional[date] = None
    decision_date: Optional[date] = None

    events: List[ProcedureEvent] = Field(default_factory=list)
    actors: List[ProcedureActor] = Field(default_factory=list)

    foreseen_activities: List[Dict[str, Any]] = Field(default_factory=list)
    commission_document: Optional[str] = None
    amending_acts: List[Dict[str, Optional[str]]] = Field(default_factory=list)
    background_documents: List[Dict[str, Optional[str]]] = Field(default_factory=list)
    celex_number: Optional[str] = None

    oeil_url: Optional[str] = None
    eurlex_proposal_url: Optional[str] = None
    eurlex_final_act_url: Optional[str] = None

    # Flat actor fields (denormalised from actors JSONB for easy querying)
    responsible_committee: Optional[str] = None
    rapporteurs: List[Dict[str, Any]] = Field(default_factory=list)
    shadow_rapporteurs: List[Dict[str, Any]] = Field(default_factory=list)
    rapporteurs_for_opinion: List[Dict[str, Any]] = Field(default_factory=list)
    commission_dg: Optional[str] = None
    commissioner: Optional[str] = None

    # Flat timeline date fields
    amendments_tabled_date: Optional[date] = None
    amendment_vote_date: Optional[date] = None
    regulation_vote_date: Optional[date] = None
    date_of_final_act_signed: Optional[date] = None

    # Gold layer (kept for parl8 compatibility, will be null)
    ai_summary: Optional[str] = None
    ai_next_steps: Optional[str] = None
    ai_impact_analysis: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None

    # Diamond layer
    is_deleted: bool = False
    deleted_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        extra = "allow"


# Backward compatibility alias
Legislation = Procedure
