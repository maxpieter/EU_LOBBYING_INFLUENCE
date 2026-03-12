"""Enrichment logic to merge OEIL and v2 API data in Silver layer.

This module combines:
- OEIL events + v2 events -> unified events timeline
- OEIL actors + v2 participations -> unified actors list
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


def _select_best_document_url(files: List[Dict[str, Any]]) -> Optional[str]:
    """Select the best document URL from available file formats.

    Priority:
    1. DOCX (preferred for A-10 amendment lists - has complete structure)
    2. HTML (fallback)
    3. PDF (last resort - harder to parse)

    Args:
        files: List of file dicts with 'url' and 'format' keys

    Returns:
        Best URL or None if no files available
    """
    if not files:
        return None

    def _fmt(f: Dict[str, Any]) -> str:
        """Normalize file_type to short uppercase code, handling both short ('DOCX')
        and full-URI ('http://.../file-type/DOCX') formats stored in bronze cache."""
        raw = f.get("file_type", "") or ""
        return raw.split("/")[-1].upper()

    # Separate files by format (v2 API stores format under "file_type" key)
    docx_files = [f for f in files if _fmt(f) == "DOCX"]
    html_files = [f for f in files if _fmt(f) == "HTML"]
    pdf_files = [f for f in files if _fmt(f) == "PDF"]

    # Return first available in priority order
    if docx_files:
        return docx_files[0].get("url")
    if html_files:
        return html_files[0].get("url")
    if pdf_files:
        return pdf_files[0].get("url")

    # Fallback: return first file's URL
    return files[0].get("url")


def _map_v2_activity_to_oeil(v2_activity_uri: Optional[str]) -> Optional[str]:
    """Map v2 activity type URI to OEIL activity type string.

    Args:
        v2_activity_uri: v2 activity type URI (e.g., "def/ep-activities/REFERRAL")

    Returns:
        OEIL-compatible activity type string
    """
    if not v2_activity_uri:
        return None

    # Extract last part of URI
    if "/" in v2_activity_uri:
        activity_code = v2_activity_uri.split("/")[-1]
    else:
        activity_code = v2_activity_uri

    # Map v2 codes to OEIL codes (these are already compatible in most cases)
    return activity_code


def _extract_person_id_from_uri(person_uri: str) -> Optional[int]:
    """Extract MEP ID from person URI.

    Args:
        person_uri: Person URI (e.g., "person/28617" or full URL)

    Returns:
        MEP ID as integer
    """
    if not person_uri:
        return None

    # Extract last part: "person/28617" -> "28617"
    if "/" in person_uri:
        id_str = person_uri.split("/")[-1]
    else:
        id_str = person_uri

    try:
        return int(id_str)
    except (ValueError, TypeError):
        return None


def _extract_org_code_from_uri(org_uri: str) -> Optional[str]:
    """Extract organization code from URI.

    Args:
        org_uri: Organization URI (e.g., "org/INTA" or full URL)

    Returns:
        Organization code (e.g., "INTA")
    """
    if not org_uri:
        return None

    # Extract last part: "org/INTA" -> "INTA"
    if "/" in org_uri:
        return org_uri.split("/")[-1]

    return org_uri


def _map_v2_role_to_oeil(v2_role_uri: Optional[str]) -> Optional[str]:
    """Map v2 participation role URI to OEIL actor role string.

    Args:
        v2_role_uri: v2 role URI (e.g., "def/ep-roles/RAPPORTEUR")

    Returns:
        OEIL-compatible role string
    """
    if not v2_role_uri:
        return None

    # Extract last part of URI
    if "/" in v2_role_uri:
        role_code = v2_role_uri.split("/")[-1]
    else:
        role_code = v2_role_uri

    # Map v2 roles to OEIL roles
    role_map = {
        "RAPPORTEUR": "rapporteur",
        "RAPPORTEUR_SHADOW": "shadow_rapporteur",
        "RAPPORTEUR_OPINION": "opinion_rapporteur",
        "RAPPORTEUR_SHADOW_OPINION": "shadow_rapporteur",
        "COMMITTEE_LEAD": "committee_responsible",
        "COMMITTEE_OPINION": "committee_for_opinion",
        "COMMITTEE_BUDGETARY_ASSESSMENT": "committee_for_opinion",
    }

    return role_map.get(role_code, role_code.lower())


def enrich_events_with_v2_data(
    procedure: Dict[str, Any],
    logger: Optional[Any] = None,
) -> None:
    """Enrich OEIL events with v2 event data (in-place).

    Strategy:
    1. Match events by date and activity type
    2. Add v2 documents to matched events
    3. Append unmatched v2 events as new events
    4. Remove _v2_events field after processing

    Args:
        procedure: Procedure dict (modified in-place)
        logger: Optional logger
    """
    v2_events = procedure.get("_v2_events", [])
    if not v2_events:
        if logger:
            logger.debug(f"No v2 events to merge for {procedure.get('id')}")
        return

    oeil_events = procedure.get("events", [])

    # Build lookup: (date, activity_type) -> event
    oeil_event_map: Dict[tuple, Dict[str, Any]] = {}
    for event in oeil_events:
        event_date = event.get("event_date")
        activity_type = event.get("activity_type")
        if event_date and activity_type:
            # Convert date to string for comparison
            date_str = (
                event_date.isoformat() if hasattr(event_date, "isoformat") else str(event_date)
            )
            key = (date_str, activity_type)
            oeil_event_map[key] = event

    matched_v2_event_ids = set()
    new_events = []

    for v2_event in v2_events:
        v2_date = v2_event.get("activity_date")
        v2_activity_uri = v2_event.get("had_activity_type")
        v2_activity_type = _map_v2_activity_to_oeil(v2_activity_uri)

        if not v2_date or not v2_activity_type:
            continue

        key = (v2_date, v2_activity_type)

        # Try to match with OEIL event
        if key in oeil_event_map:
            # Enrich existing OEIL event with v2 documents
            oeil_event = oeil_event_map[key]
            v2_docs = v2_event.get("resolved_docs", [])

            # Add v2 documents to event (avoid duplicates by doc_id)
            existing_doc_ids = {doc.get("id") for doc in oeil_event.get("documents", [])}

            for v2_doc in v2_docs:
                doc_id = v2_doc.get("doc_id")
                if doc_id and doc_id not in existing_doc_ids:
                    # Convert v2 doc format to OEIL format
                    # Select best URL format (prefer DOCX for A-10 documents)
                    oeil_event["documents"].append(
                        {
                            "id": doc_id,
                            "relationship": "based_on",  # Default, v2 doesn't specify
                            "url": _select_best_document_url(v2_doc.get("files", [])),
                            "_v2_files": v2_doc.get("files", []),  # Keep all file formats
                            "_v2_work_type": v2_doc.get("work_type"),
                            "_v2_title": v2_doc.get("title"),
                        }
                    )
                    existing_doc_ids.add(doc_id)

            matched_v2_event_ids.add(v2_event.get("event_id"))
        else:
            # No match - this is a new event from v2
            v2_docs = v2_event.get("resolved_docs", [])
            documents = []

            for v2_doc in v2_docs:
                doc_id = v2_doc.get("doc_id")
                if doc_id:
                    # Select best URL format (prefer DOCX for A-10 documents)
                    documents.append(
                        {
                            "id": doc_id,
                            "relationship": "based_on",
                            "url": _select_best_document_url(v2_doc.get("files", [])),
                            "_v2_files": v2_doc.get("files", []),
                            "_v2_work_type": v2_doc.get("work_type"),
                            "_v2_title": v2_doc.get("title"),
                        }
                    )

            # Create new event in OEIL format
            new_event = {
                "event_id": v2_event.get("event_id"),
                "event_date": datetime.fromisoformat(v2_date).date() if v2_date else None,
                "event_type": "Activity",  # v2 doesn't distinguish, default to Activity
                "activity_type": v2_activity_type,
                "title": v2_activity_type,  # Use activity type as title
                "description": None,
                "documents": documents,
                "summary_text": None,
                "parliament_code": None,
            }
            new_events.append(new_event)

    # Append new events to the end
    if new_events:
        oeil_events.extend(new_events)
        if logger:
            logger.info(
                f"Added {len(new_events)} new events from v2 API for {procedure.get('id')} "
                f"(matched {len(matched_v2_event_ids)} existing events)"
            )

    # Remove _v2_events field (cleanup)
    if "_v2_events" in procedure:
        del procedure["_v2_events"]


def enrich_actors_with_v2_participations(
    procedure: Dict[str, Any],
    logger: Optional[Any] = None,
) -> None:
    """Use v2 API as source of truth for roles, override OEIL misclassifications (in-place).

    Strategy:
    1. Build v2 lookup: {mep_id: [correct_roles]}
    2. Correct OEIL actor roles using v2 lookup (v2 overrides OEIL)
    3. Add missing actors from v2 that OEIL didn't capture
    4. Remove consultative_body (invented type)
    5. Deduplicate by (mep_id, role)

    Args:
        procedure: Procedure dict (modified in-place)
        logger: Optional logger
    """
    v2_participations = procedure.get("_v2_participations", [])
    if not v2_participations:
        if logger:
            logger.debug(f"No v2 participations to merge for {procedure.get('id')}")
        return

    oeil_actors = procedure.get("actors", [])

    # Step 1: Build v2 lookup {mep_id: [roles]}
    v2_mep_roles: Dict[int, List[str]] = {}

    for v2_part in v2_participations:
        v2_role_uri = v2_part.get("participation_role")
        v2_role = _map_v2_role_to_oeil(v2_role_uri)

        if not v2_role:
            continue

        # Check if this is a person or organization participation
        person_uris = v2_part.get("had_participant_person") or []

        # Handle person participations
        for person_uri in person_uris:
            mep_id = _extract_person_id_from_uri(person_uri)
            if mep_id:
                if mep_id not in v2_mep_roles:
                    v2_mep_roles[mep_id] = []
                if v2_role not in v2_mep_roles[mep_id]:
                    v2_mep_roles[mep_id].append(v2_role)

    # Step 2: Correct OEIL actor roles using v2 data
    corrected_actors = []
    seen_identities: set[tuple] = set()  # {(mep_id, role)} or {(committee_code, role)}

    for actor in oeil_actors:
        # Filter out consultative_body (invented type)
        if actor.get("actor_type") == "consultative_body":
            if logger:
                logger.debug(
                    f"Filtered out consultative_body actor: {actor.get('institution_name')}"
                )
            continue

        mep_id = actor.get("mep_id")

        # If this MEP has v2 data, use v2 roles (override OEIL)
        if mep_id and mep_id in v2_mep_roles:
            v2_roles = v2_mep_roles[mep_id]

            # Create actor entry for each v2 role
            for v2_role in v2_roles:
                identity = (mep_id, v2_role)
                if identity not in seen_identities:
                    corrected_actor = actor.copy()
                    corrected_actor["role"] = v2_role  # Override with v2 role
                    corrected_actors.append(corrected_actor)
                    seen_identities.add(identity)

            # Mark this mep_id as processed
            del v2_mep_roles[mep_id]
        else:
            # No v2 data for this MEP, keep OEIL role
            # OR non-MEP actor (committee, commission)
            if mep_id:
                identity = (mep_id, actor.get("role"))
            elif actor.get("committee_code"):
                identity = (actor.get("committee_code"), actor.get("role"))
            elif actor.get("commissioner_name"):
                identity = (actor.get("commissioner_name"), actor.get("role"))
            else:
                identity = (id(actor), actor.get("role"))

            if identity not in seen_identities:
                corrected_actors.append(actor)
                seen_identities.add(identity)

    # Step 3: Add MEPs from v2 that OEIL missed
    for mep_id, roles in v2_mep_roles.items():
        for role in roles:
            identity = (mep_id, role)
            if identity not in seen_identities:
                corrected_actors.append(
                    {
                        "actor_type": "mep",
                        "role": role,
                        "mep_id": mep_id,
                        "mep_name": None,
                        "committee_code": None,
                        "committee_name": None,
                        "institution_name": None,
                        "commissioner_name": None,
                        "is_active": True,
                    }
                )
                seen_identities.add(identity)

                if logger:
                    logger.debug(f"Added missing MEP {mep_id} with role {role} from v2 API")

    # Update procedure actors
    procedure["actors"] = corrected_actors

    if logger:
        oeil_count = len(oeil_actors)
        corrected_count = len(corrected_actors)
        if corrected_count != oeil_count:
            logger.info(
                f"Actor correction for {procedure.get('id')}: "
                f"{oeil_count} OEIL actors -> {corrected_count} corrected actors "
                f"(using v2 roles as source of truth)"
            )

    # Remove _v2_participations field (cleanup)
    if "_v2_participations" in procedure:
        del procedure["_v2_participations"]


def _add_institutional_actors(
    procedure: Dict[str, Any],
    logger: Optional[Any] = None,
) -> None:
    """Add institutional actors (Council, Parliament) inferred from events (in-place).

    Strategy:
    - Scan events for institutional activities
    - Add minimal institutional actors if not present

    Args:
        procedure: Procedure dict (modified in-place)
        logger: Optional logger
    """
    actors = procedure.get("actors", [])
    events = procedure.get("events", [])

    # Check for Council activity
    has_council_event = any("COUNCIL" in e.get("activity_type", "").upper() for e in events)

    # Check if Council already exists
    has_council_actor = any(
        a.get("institution_name") == "Council of the European Union" for a in actors
    )

    if has_council_event and not has_council_actor:
        actors.append(
            {
                "actor_type": "institution",
                "role": "co_legislator",
                "mep_id": None,
                "mep_name": None,
                "committee_code": None,
                "committee_name": None,
                "institution_name": "Council of the European Union",
                "commissioner_name": None,
                "is_active": True,
            }
        )
        if logger:
            logger.debug(f"Added Council as institutional actor for {procedure.get('id')}")

    # Check for Parliament decision (plenary activity)
    has_parliament_event = any("PLENARY" in e.get("activity_type", "").upper() for e in events)

    has_parliament_actor = any(a.get("institution_name") == "European Parliament" for a in actors)

    if has_parliament_event and not has_parliament_actor:
        actors.append(
            {
                "actor_type": "institution",
                "role": "co_legislator",
                "mep_id": None,
                "mep_name": None,
                "committee_code": None,
                "committee_name": None,
                "institution_name": "European Parliament",
                "commissioner_name": None,
                "is_active": True,
            }
        )
        if logger:
            logger.debug(f"Added Parliament as institutional actor for {procedure.get('id')}")


def enrich_procedure_with_v2_data(
    procedure: Dict[str, Any],
    logger: Optional[Any] = None,
) -> None:
    """Main entry point to enrich a procedure with v2 API data (in-place).

    Args:
        procedure: Procedure dict (modified in-place)
        logger: Optional logger
    """
    enrich_events_with_v2_data(procedure, logger=logger)
    enrich_actors_with_v2_participations(procedure, logger=logger)
    _add_institutional_actors(procedure, logger=logger)

    # Also remove _v2_stats (cleanup)
    if "_v2_stats" in procedure:
        del procedure["_v2_stats"]
