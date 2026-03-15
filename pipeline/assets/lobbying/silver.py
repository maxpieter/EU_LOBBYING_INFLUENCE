"""Silver stage: Transform raw lobbying data into structured models.

This stage:
1. Processes raw Transparency Register XML data into Organization objects with IDs.
2. Processes raw meeting CSV data into LobbyingMeeting objects.
3. Extracts additional organizations from meetings (those not in transparency register).
4. Links meetings to organizations using deterministic IDs.
"""

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pipeline.models.form_entity_mapping import (
    get_organization_type_from_form_entity,
)
from pipeline.models.lobbying_models import (
    normalize_org_name_for_id,  # Import the normalization function
)
from pipeline.models.lobbying_models import (
    LobbyingMeeting,
    Organization,
    OrganizationType,
)


def clean_org_name(name: str) -> str:
    """Strip person names and noise from organization name.

    Handles patterns like:
    - "TotalEnergies Mr. Eric Quenet, M. Giovanni Butelli"
    - "ECSA, Mr. Marc du Moulin"
    - "Name1|Name2|Name3" (takes first part)
    """
    # Take first pipe-separated segment
    if "|" in name:
        name = name.split("|")[0].strip()
    # Remove everything from person title onward
    name = re.split(
        r"\s*(?:,\s*)?(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|M\.\s|Mme\.?\s)\s",
        name,
    )[0]
    # Remove trailing comma + capitalized name pattern
    name = re.sub(r"\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+(?:,.*)?$", "", name)
    return name.rstrip(", ").strip()


def process_transparency_data(
    raw_data: List[Dict[str, Any]], logger: Optional[Any] = None
) -> List[Organization]:
    """Process raw transparency register XML data into Organization objects.

    Organizations from the transparency register get their ID from identificationCode.

    Args:
        raw_data: Raw XML data from bronze layer (list of dicts with XML fields)
        logger: Optional logger

    Returns:
        List of Organization objects with IDs assigned
    """
    organizations = []

    for row in raw_data:
        # Extract name
        name = row.get("name")
        if not name:
            continue

        # Extract transparency register ID (this becomes the organization ID)
        transparency_id = row.get("identificationCode")
        if not transparency_id:
            if logger:
                logger.warning(f"Organization {name} has no identificationCode, skipping")
            continue

        # Extract category for organization type
        category = row.get("registrationCategory")

        # Determine organization type from category
        org_type = None
        if category:
            try:
                # Match enum values
                for t in OrganizationType:
                    if t.value == category:
                        org_type = t
                        break
            except ValueError:
                pass

        # If no type found, try to infer from entityForm
        if not org_type:
            entity_form = row.get("entityForm")
            if entity_form:
                org_type = get_organization_type_from_form_entity(entity_form)

        # Fallback to OTHER
        if not org_type:
            org_type = OrganizationType.OTHER_ORGANISATIONS_PUBLIC_MIXED

        # Extract head office info
        head_office = row.get("headOffice", {})

        # Extract policy focus areas from interests
        policy_focus_areas = row.get("interests", [])

        # Extract employee count from members data
        members = row.get("members", {})
        employee_count_range = None
        if members:
            try:
                fte = float(members.get("membersFTE", 0) or 0)
                if fte > 0:
                    if fte <= 5:
                        employee_count_range = "1-5"
                    elif fte <= 10:
                        employee_count_range = "6-10"
                    elif fte <= 25:
                        employee_count_range = "11-25"
                    elif fte <= 50:
                        employee_count_range = "26-50"
                    elif fte <= 100:
                        employee_count_range = "51-100"
                    elif fte <= 250:
                        employee_count_range = "101-250"
                    elif fte <= 500:
                        employee_count_range = "251-500"
                    else:
                        employee_count_range = "500+"
            except (ValueError, TypeError):
                pass

        # Create organization object (ID will be auto-generated from transparency_id)
        org = Organization(
            name=name,
            normalized_name=name,
            official_name=name,
            organization_type=org_type,
            eu_transparency_register_id=transparency_id,
            # Additional fields from XML
            acronym=row.get("acronym"),
            website=row.get("webSiteURL"),
            description=row.get("goals"),
            country=head_office.get("country") if head_office else None,
            city=head_office.get("city") if head_office else None,
            address=head_office.get("address") if head_office else None,
            post_code=head_office.get("postCode") if head_office else None,
            # Interests and activities
            level_of_interest=(
                ", ".join(row.get("levelsOfInterest", [])) if row.get("levelsOfInterest") else None
            ),
            interests_represented=row.get("interestRepresented"),
            form_of_entity=row.get("entityForm"),
            policy_focus_areas=policy_focus_areas if policy_focus_areas else [],
            employee_count_range=employee_count_range,
        )

        organizations.append(org)

    if logger:
        logger.info(f"Processed {len(organizations)} organizations from transparency register")

    return organizations


def process_meetings(
    raw_meetings: List[Dict[str, Any]],
    existing_orgs: List[Organization],
    logger: Optional[Any] = None,
) -> Tuple[List[LobbyingMeeting], List[Organization]]:
    """Process raw meetings into LobbyingMeeting objects and extract new organizations.

    This function:
    1. Links meetings to existing organizations (by transparency ID or name)
    2. Creates new Organization objects for orgs not in transparency register
    3. Generates deterministic IDs for new organizations (hash-based)
    4. Creates LobbyingMeeting objects with proper organization_id references

    Args:
        raw_meetings: Raw CSV meeting data from bronze layer
        existing_orgs: Organizations from transparency register (with IDs)
        logger: Optional logger

    Returns:
        Tuple of (meetings, new_organizations)
    """
    meetings = []
    new_orgs = []

    # Index existing orgs by transparency ID and normalized name
    orgs_by_transparency_id = {
        org.eu_transparency_register_id: org
        for org in existing_orgs
        if org.eu_transparency_register_id
    }
    # Index by fully normalized name (spaces/punctuation removed)
    orgs_by_normalized_name = {normalize_org_name_for_id(org.name): org for org in existing_orgs}
    # Index by org ID
    orgs_by_id = {org.id: org for org in existing_orgs}
    # Index by acronym (lowered)
    orgs_by_acronym = {
        org.acronym.lower(): org for org in existing_orgs if org.acronym
    }

    # Track new orgs we create to avoid duplicates within this batch
    new_orgs_by_name = {}

    for meeting_row in raw_meetings:
        # Extract organization info from meeting
        org_name = (meeting_row.get("attendees") or "").strip()
        transparency_id = (meeting_row.get("lobbyist_id") or "").strip()

        if not org_name:
            continue

        # Find or create organization
        org = None

        # Strategy 1: Try by transparency ID (most reliable)
        if transparency_id and transparency_id in orgs_by_transparency_id:
            org = orgs_by_transparency_id[transparency_id]

        # Strategy 2: Try by normalized name
        elif normalize_org_name_for_id(org_name) in orgs_by_normalized_name:
            org = orgs_by_normalized_name[normalize_org_name_for_id(org_name)]
            # If we found org by name and meeting has transparency_id, update it
            if transparency_id and not org.eu_transparency_register_id:
                org.eu_transparency_register_id = transparency_id
                # Re-index with new ID
                orgs_by_transparency_id[transparency_id] = org

        # Strategy 2b: Clean name (strip person names, pipe segments) and retry
        else:
            cleaned = clean_org_name(org_name)
            if cleaned != org_name:
                cleaned_norm = normalize_org_name_for_id(cleaned)
                if cleaned_norm in orgs_by_normalized_name:
                    org = orgs_by_normalized_name[cleaned_norm]
                elif cleaned.lower() in orgs_by_acronym:
                    org = orgs_by_acronym[cleaned.lower()]

        # Strategy 2c: Try acronym match (for short names like "IFAW")
        if org is None and org_name.lower() in orgs_by_acronym:
            org = orgs_by_acronym[org_name.lower()]

        # Strategy 2d: Try parenthetical match — "FEAP (Federation of...)" or "Federation of... (FEAP)"
        if org is None:
            paren_match = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", org_name)
            if paren_match:
                outer, inner = paren_match.group(1).strip(), paren_match.group(2).strip()
                for part in [outer, inner]:
                    norm = normalize_org_name_for_id(part)
                    if norm in orgs_by_normalized_name:
                        org = orgs_by_normalized_name[norm]
                        break
                    if part.lower() in orgs_by_acronym:
                        org = orgs_by_acronym[part.lower()]
                        break

        # Strategy 2e: Prefix match — "Toyota" → "TOYOTA MOTOR EUROPE"
        # Only for names long enough to avoid false positives
        if org is None and len(org_name) >= 5:
            cleaned = clean_org_name(org_name)
            name_lower = cleaned.lower()
            candidates = [
                v for k, v in orgs_by_normalized_name.items()
                if k.startswith(normalize_org_name_for_id(cleaned))
            ]
            if len(candidates) == 1:
                org = candidates[0]

        if org is not None:
            pass  # Already matched above

        # Strategy 3: Check if we already created it in this batch
        elif org_name.lower() in new_orgs_by_name:
            org = new_orgs_by_name[org_name.lower()]

        # Strategy 4: Create new organization (not in register)
        else:
            # Infer organization type from name (best effort)
            org_type = get_organization_type_from_form_entity(org_name)
            if not org_type:
                org_type = OrganizationType.OTHER_ORGANISATIONS_PUBLIC_MIXED

            # Create new organization
            # The Organization.__init__ will auto-generate ID from normalized name hash
            org = Organization(
                name=org_name,
                normalized_name=org_name,
                organization_type=org_type,
                eu_transparency_register_id=(transparency_id if transparency_id else None),
            )

            # Check if this ID already exists (fuzzy match via ID)
            if org.id in orgs_by_id:
                # Use existing org with this ID
                org = orgs_by_id[org.id]
                if logger:
                    logger.debug(
                        f"Matched '{org_name}' to existing org '{org.name}' via normalized ID"
                    )
            else:
                # Track new org
                new_orgs.append(org)
                new_orgs_by_name[org_name.lower()] = org
                orgs_by_id[org.id] = org
                orgs_by_normalized_name[normalize_org_name_for_id(org_name)] = org

                # Index it for subsequent meetings
                if transparency_id:
                    orgs_by_transparency_id[transparency_id] = org

        # Now create the meeting with proper organization_id
        meeting_date_str = meeting_row.get("meeting_date")
        meeting_date = None
        if meeting_date_str:
            try:
                meeting_date = datetime.strptime(meeting_date_str, "%Y-%m-%d").date()
            except ValueError:
                if logger:
                    logger.warning(f"Could not parse meeting date: {meeting_date_str}")

        # Extract MEP ID
        mep_id_str = meeting_row.get("member_id", "").strip()
        mep_id = None
        if mep_id_str:
            try:
                mep_id = int(mep_id_str)
            except ValueError:
                if logger:
                    logger.warning(f"Could not parse member_id: {mep_id_str}")

        if not mep_id:
            if logger:
                logger.debug(f"Meeting without MEP ID: {meeting_row.get('title', '')[:60]}")

        # Generate deterministic meeting ID
        # Include org_name in hash to ensure unique IDs when same meeting has multiple orgs
        # Use member_name as fallback when mep_id is missing
        mep_key = str(mep_id) if mep_id else meeting_row.get("member_name", "unknown")
        id_key = f"{meeting_date}_{mep_key}_{org.id}_{org_name}"
        meeting_id = hashlib.sha256(id_key.encode("utf-8")).hexdigest()

        # Determine meeting type from title
        title = meeting_row.get("title", "").strip()
        meeting_type = "Formal"  # Default
        title_lower = title.lower()
        if any(term in title_lower for term in ["hearing", "committee"]):
            meeting_type = "Committee"
        elif any(term in title_lower for term in ["conference", "workshop", "event"]):
            meeting_type = "Event"
        elif any(term in title_lower for term in ["bilateral", "meeting"]):
            meeting_type = "Informal"
        elif "briefing" in title_lower:
            meeting_type = "Briefing"

        # Create meeting object
        meeting = LobbyingMeeting(
            id=meeting_id,
            mep_id=mep_id,
            organization_id=org.id,  # Critical: link to organization
            meeting_date=meeting_date,
            title=title,
            location=meeting_row.get("location"),
            meeting_type=meeting_type,
            capacity=meeting_row.get("member_capacity"),
            related_procedure=meeting_row.get("procedure_reference"),
            committee_acronym=meeting_row.get("committee_code"),  # NEW: Committee code from HTML
            transparency_level="High" if org.eu_transparency_register_id else "Low",
        )

        meetings.append(meeting)

    if logger:
        logger.info(f"Created {len(meetings)} meetings")
        logger.info(
            f"Extracted {len(new_orgs)} additional organizations from meetings (not in register)"
        )

    return meetings, new_orgs
