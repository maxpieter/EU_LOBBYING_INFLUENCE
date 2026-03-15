"""Pydantic models for lobbying data. Matches parl8 schema."""

import hashlib
import re
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def normalize_org_name_for_id(name: str) -> str:
    """Normalize organization name for ID generation.

    Ensures "Fuels Europe" and "FuelsEurope" get the same ID.
    """
    if not name:
        return ""
    normalized = name.lower().replace(" ", "")
    for char in [".", ",", "-", "_", "'", '"', "(", ")", "&"]:
        normalized = normalized.replace(char, "")
    return normalized


def generate_organization_id(name: str, eu_transparency_register_id: Optional[str] = None) -> str:
    """Generate a deterministic ID for an organization."""
    if eu_transparency_register_id and eu_transparency_register_id.strip():
        return eu_transparency_register_id.strip()
    else:
        normalized_name = normalize_org_name_for_id(name)
        source_string = f"name_{normalized_name}"
        hash_hex = hashlib.sha256(source_string.encode("utf-8")).hexdigest()
        return f"org_{hash_hex[:16]}"


def sanitize_string(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"[\u2028\u2029\u0000-\u001F\u007F]", " ", s)
    s = re.sub(r" +", " ", s)
    return s.strip()


class OrganizationType(str, Enum):
    COMPANIES_GROUPS = "Companies & groups"
    NON_GOVERNMENTAL_ORGANISATIONS = (
        "Non-governmental organisations, platforms and networks and similar"
    )
    TRADE_BUSINESS_ASSOCIATIONS = "Trade and business associations"
    TRADE_UNIONS_PROFESSIONAL_ASSOCIATIONS = "Trade unions and professional associations"
    PROFESSIONAL_CONSULTANCIES = "Professional consultancies"
    THINK_TANKS_RESEARCH_INSTITUTIONS = "Think tanks and research institutions"
    OTHER_ORGANISATIONS_PUBLIC_MIXED = "Other organisations, public or mixed entities"
    ACADEMIC_INSTITUTIONS = "Academic institutions"
    ASSOCIATIONS_NETWORKS_PUBLIC_AUTHORITIES = "Associations and networks of public authorities"
    SELF_EMPLOYED_INDIVIDUALS = "Self-employed individuals"
    LAW_FIRMS = "Law firms"
    CHURCHES_RELIGIOUS_COMMUNITIES = (
        "Organisations representing churches and religious communities"
    )
    THIRD_COUNTRY_ENTITIES = "Entities, offices or networks established by third countries"


class IndustrySector(str, Enum):
    TECHNOLOGY = "Technology"
    ENERGY = "Energy"
    FINANCE = "Finance"
    HEALTHCARE = "Healthcare"
    AGRICULTURE = "Agriculture"
    TRANSPORT = "Transport"
    ENVIRONMENT = "Environment"
    OTHER = "Other"


class MeetingType(str, Enum):
    FORMAL = "Formal"
    INFORMAL = "Informal"
    COMMITTEE = "Committee"
    EVENT = "Event"
    BRIEFING = "Briefing"


class TransparencyLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Organization(BaseModel):
    id: Optional[str] = None
    name: str
    normalized_name: Optional[str] = None
    official_name: Optional[str] = None
    website: Optional[str] = None
    organization_type: Optional[OrganizationType] = None
    industry_sector: Optional[IndustrySector] = None
    country: Optional[str] = None
    eu_transparency_register_id: Optional[str] = None
    description: Optional[str] = None
    founding_year: Optional[int] = None
    employee_count_range: Optional[str] = None
    annual_revenue_range: Optional[str] = None

    total_meetings_count: int = 0
    unique_meps_met: int = 0
    influence_score: Optional[float] = None
    transparency_score: Optional[float] = None
    activity_level: Optional[str] = None

    scraped_at: Optional[datetime] = None
    logo_url: Optional[str] = None
    social_media: Dict[str, str] = Field(default_factory=dict)
    key_personnel: List[Dict[str, Any]] = Field(default_factory=list)
    policy_focus_areas: List[str] = Field(default_factory=list)

    acronym: Optional[str] = None
    city: Optional[str] = None
    address: Optional[str] = None
    post_code: Optional[str] = None
    level_of_interest: Optional[str] = None
    interests_represented: Optional[str] = None
    form_of_entity: Optional[str] = None
    source_of_funding: Optional[str] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("*", mode="before")
    @classmethod
    def sanitize_all_strings(cls, v):
        if isinstance(v, str):
            return sanitize_string(v)
        return v

    def __init__(self, **data):
        super().__init__(**data)
        if not self.id:
            self.id = generate_organization_id(self.name, self.eu_transparency_register_id)

    @property
    def transparency_register_url(self) -> Optional[str]:
        if self.eu_transparency_register_id:
            base = "https://ec.europa.eu/transparencyregister/public/consultation/displaylobbyist.do"
            return f"{base}?id={self.eu_transparency_register_id}"
        return None


class LobbyingMeeting(BaseModel):
    id: Optional[str] = None
    mep_id: Optional[int] = None
    organization_id: Optional[str] = None

    meeting_date: Optional[date] = None
    title: Optional[str] = None
    location: Optional[str] = None
    capacity: Optional[str] = None
    related_procedure: Optional[str] = None
    committee_acronym: Optional[str] = None

    meeting_type: Optional[MeetingType] = None
    transparency_level: Optional[TransparencyLevel] = None

    @field_validator("*", mode="before")
    @classmethod
    def sanitize_all_strings(cls, v):
        if isinstance(v, str):
            return sanitize_string(v)
        return v
