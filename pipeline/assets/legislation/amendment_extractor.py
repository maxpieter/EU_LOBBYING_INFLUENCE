"""Amendment extraction and MEP matching for Diamond layer.

Extracts amendments from Silver layer events and matches MEP names to IDs
for contributor statistics and tracking.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


def clean_mep_name(name: str) -> str:
    """Clean MEP name by removing XML tags and normalizing whitespace.

    Args:
        name: Raw name possibly containing XML tags like "<Depute>Name</Depute>"

    Returns:
        Cleaned name string
    """
    if not name:
        return ""
    # Remove XML/HTML tags
    clean = re.sub(r"<[^>]+>", "", name)
    # Normalize whitespace
    clean = " ".join(clean.split())
    return clean.strip()


def normalize_name_for_matching(name: str) -> str:
    """Normalize name for fuzzy matching.

    Handles variations like:
    - "Inese VAIDERE" vs "Inese Vaidere"
    - "VAIDERE Inese" vs "Inese Vaidere"

    Args:
        name: Name to normalize

    Returns:
        Normalized lowercase name with parts sorted alphabetically
    """
    if not name:
        return ""
    # Clean and lowercase
    clean = clean_mep_name(name).lower()
    # Split into parts and sort (handles firstname-lastname vs lastname-firstname)
    parts = clean.split()
    # Return sorted parts joined (for comparison)
    return " ".join(sorted(parts))


def build_mep_lookup_cache(
    supabase_resource: Any,
    logger: Optional[Any] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build MEP name lookup caches from database.

    Args:
        supabase_resource: Supabase resource for querying
        logger: Optional logger

    Returns:
        Tuple of (exact_match_cache, normalized_cache)
        - exact_match_cache: fullName -> mep_id
        - normalized_cache: normalized_name -> mep_id
    """
    try:
        result = supabase_resource.select(
            table="meps",
            columns="id,fullName",
            filters={"status": "active"},
        )

        exact_cache: Dict[str, int] = {}
        normalized_cache: Dict[str, int] = {}

        for mep in result.data:
            mep_id = mep["id"]
            full_name = mep.get("fullName", "")

            if full_name:
                # Exact match cache
                exact_cache[full_name] = mep_id
                # Normalized cache for fuzzy matching
                normalized = normalize_name_for_matching(full_name)
                normalized_cache[normalized] = mep_id

        if logger:
            logger.info(f"Built MEP lookup cache with {len(exact_cache)} MEPs")

        return exact_cache, normalized_cache

    except Exception as e:
        if logger:
            logger.warning(f"Failed to build MEP cache: {e}")
        return {}, {}


def match_mep_name(
    name: str,
    exact_cache: Dict[str, int],
    normalized_cache: Dict[str, int],
) -> Optional[int]:
    """Match MEP name to ID using exact and fuzzy matching.

    Args:
        name: Name to match (may contain XML tags)
        exact_cache: Exact name -> ID mapping
        normalized_cache: Normalized name -> ID mapping

    Returns:
        MEP ID if found, None otherwise
    """
    if not name:
        return None

    clean_name = clean_mep_name(name)

    # Try exact match first
    if clean_name in exact_cache:
        return exact_cache[clean_name]

    # Try normalized fuzzy match
    normalized = normalize_name_for_matching(clean_name)
    if normalized in normalized_cache:
        return normalized_cache[normalized]

    return None


def parse_target_element(target: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse target element string into type and element.

    Args:
        target: Target string like "Recital 1", "Article 5(2)"

    Returns:
        Tuple of (target_type, target_element)
    """
    if not target:
        return None, None

    target_lower = target.lower().strip()

    if target_lower.startswith("recital"):
        return "recital", target
    elif target_lower.startswith("article"):
        return "article", target
    elif target_lower.startswith("annex"):
        return "annex", target
    elif target_lower.startswith("title"):
        return "title", target
    elif target_lower.startswith("citation"):
        return "citation", target
    else:
        return "other", target


def extract_amendments_from_procedure(
    procedure: Dict[str, Any],
    exact_cache: Dict[str, int],
    normalized_cache: Dict[str, int],
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Extract all amendments from a procedure's events.

    Args:
        procedure: Procedure dict with events containing _amendments
        exact_cache: Exact MEP name -> ID mapping
        normalized_cache: Normalized MEP name -> ID mapping
        logger: Optional logger

    Returns:
        List of amendment records ready for database insert
    """
    procedure_id = procedure.get("id")
    if not procedure_id:
        return []

    amendments: List[Dict[str, Any]] = []

    for event in procedure.get("events", []):
        # Check for _amendments in event
        amendments_data = event.get("_amendments")
        if not amendments_data:
            continue

        # Get event metadata
        event_date = event.get("event_date")  # Silver uses "event_date" not "date"
        work_type = amendments_data.get("work_type") or event.get("activity_type")

        # Get amendments list (nested under "amendments" key)
        amendments_list = amendments_data.get("amendments", [])

        for amendment in amendments_list:
            # Parse target element
            target_element = amendment.get("target_article")
            target_type, _ = parse_target_element(target_element)

            # Match rapporteur to MEP ID
            rapporteur_raw = amendment.get("rapporteur", "")
            rapporteur_mep_id = match_mep_name(rapporteur_raw, exact_cache, normalized_cache)

            # Match submitted_by list to MEP IDs (simple array of IDs)
            submitted_by_raw = amendment.get("submitted_by", [])
            submitted_by: List[int] = []

            for submitter in submitted_by_raw:
                if isinstance(submitter, str):
                    submitter_name = submitter
                elif isinstance(submitter, dict):
                    submitter_name = submitter.get("name", "")
                else:
                    continue

                mep_id = match_mep_name(submitter_name, exact_cache, normalized_cache)
                if mep_id is not None:
                    submitted_by.append(mep_id)

            # Build amendment record
            record = {
                "procedure_id": procedure_id,
                "event_date": event_date,  # From event.date
                "document_id": amendment.get("document_id", ""),
                "work_type": work_type,
                "amendment_number": amendment.get("amendment_number"),
                "target_element": target_element,
                "target_type": target_type,
                "original_text": amendment.get("original"),
                "amended_text": amendment.get("amended"),
                "justification": amendment.get("justification"),
                "committee": amendment.get("committee"),
                "rapporteur_mep_id": rapporteur_mep_id,
                "submitted_by": submitted_by if submitted_by else [],  # Array of MEP IDs
                "adopted": None,  # TODO: Determine from final text comparison
            }

            amendments.append(record)

    if logger and amendments:
        logger.info(f"Extracted {len(amendments)} amendments from {procedure_id}")

    return amendments


def extract_amendments_from_procedures(
    procedures: List[Dict[str, Any]],
    supabase_resource: Any,
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Extract amendments from multiple procedures with MEP matching.

    Args:
        procedures: List of procedure dicts from Gold layer
        supabase_resource: Supabase resource for MEP lookup
        logger: Optional logger

    Returns:
        List of all amendment records ready for database insert
    """
    # Build MEP lookup cache once
    exact_cache, normalized_cache = build_mep_lookup_cache(supabase_resource, logger)

    all_amendments: List[Dict[str, Any]] = []

    for procedure in procedures:
        amendments = extract_amendments_from_procedure(
            procedure,
            exact_cache,
            normalized_cache,
            logger,
        )
        all_amendments.extend(amendments)

    if logger:
        logger.info(f"Extracted {len(all_amendments)} total amendments")

    return all_amendments


def strip_amendments_from_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove _amendments from events (moved to separate table).

    Args:
        events: List of event dicts

    Returns:
        Events with _amendments removed
    """
    cleaned_events = []
    for event in events:
        # Create copy without _amendments
        cleaned = {k: v for k, v in event.items() if k != "_amendments"}
        cleaned_events.append(cleaned)
    return cleaned_events
