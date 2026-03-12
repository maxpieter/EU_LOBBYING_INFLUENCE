"""Post-enrichment utilities for Silver layer.

Handles:
1. Stage/status inference from events
2. Event provenance (source field)
3. Actor name resolution status
"""

from typing import Any, Dict, List, Optional


def _infer_stage_and_status_from_events(
    events: List[Dict[str, Any]],
    current_status: Optional[str] = None,
    logger: Optional[Any] = None,
) -> tuple[str, str]:
    """Infer final stage and status from event timeline and OEIL status text.

    Strategy:
    1. Parse OEIL descriptive status for stage hints (e.g., "Awaiting committee decision")
    2. Scan events for terminal activities (publication, adoption)
    3. Return most advanced stage and corresponding normalized status

    Args:
        events: List of procedure events
        current_status: OEIL status text (may be descriptive)
        logger: Optional logger

    Returns:
        (stage, status) tuple with normalized values
    """
    # First, try to infer from OEIL descriptive status text
    if current_status:
        status_lower = current_status.lower()

        # Map OEIL status text to stage and normalized status
        # Check for "awaiting publication" first (most specific)
        if (
            "awaiting publication" in status_lower
            or "awaiting official publication" in status_lower
        ):
            return ("Awaiting Publication", "completed")
        if "procedure completed" in status_lower and "awaiting" in status_lower:
            return ("Awaiting Publication", "completed")
        if "awaiting committee" in status_lower:
            return ("1st reading - Parliament", "in_progress")
        if "awaiting parliament" in status_lower and "1st" in status_lower:
            return ("1st reading - Parliament", "in_progress")
        if (
            "awaiting council" in status_lower or "awaiting position" in status_lower
        ) and "1st" in status_lower:
            return ("1st reading - Council", "in_progress")
        if "awaiting parliament" in status_lower and "2nd" in status_lower:
            return ("2nd reading - Parliament", "in_progress")
        if "awaiting council" in status_lower and "2nd" in status_lower:
            return ("2nd reading - Council", "in_progress")
        if "trilogue" in status_lower:
            return ("Trilogue", "in_progress")
        if "conciliation" in status_lower:
            return ("Conciliation", "in_progress")
        if "procedure completed" in status_lower or "procedure is completed" in status_lower:
            # Completed but need to check if it's adopted or rejected
            # Will be determined by events below
            pass

    # Scan events to find most advanced activity
    most_advanced_stage = ("Commission", "in_progress")  # Default

    for event in events:
        activity_type = event.get("activity_type", "").upper()

        # Check for publication (most advanced - procedure completed as law)
        if "PUBLICATION" in activity_type and "OFFICIAL_JOURNAL" in activity_type:
            return ("Published", "in_force")

        # Check for signature (agreement reached, awaiting publication)
        if "SIGNATURE" in activity_type:
            # After signature, procedure is completed but awaiting publication
            most_advanced_stage = ("Awaiting Publication", "completed")
            continue

        # Check for Council adoption
        if "COUNCIL" in activity_type and "ADOPT" in activity_type:
            most_advanced_stage = ("Council - adopted", "adopted")
            continue

        # Check for Parliament adoption
        if "PLENARY" in activity_type and "ADOPT" in activity_type:
            if most_advanced_stage[0] not in ["Council", "Signature", "Published"]:
                most_advanced_stage = ("Parliament - adopted", "adopted")
            continue

        # Check for trilogue
        if "TRILOGUE" in activity_type:
            if most_advanced_stage[0] in [
                "Commission",
                "1st reading - Parliament",
                "1st reading - Council",
            ]:
                most_advanced_stage = ("Trilogue", "in_progress")
            continue

        # Check for conciliation
        if "CONCILIATION" in activity_type:
            most_advanced_stage = ("Conciliation", "in_progress")
            continue

        # Check for committee stage (1st reading)
        if "COMMITTEE" in activity_type:
            if most_advanced_stage[0] == "Commission":
                most_advanced_stage = ("1st reading - Parliament", "in_progress")

    return most_advanced_stage


def add_event_provenance(
    procedure: Dict[str, Any],
    logger: Optional[Any] = None,
) -> None:
    """Add source provenance to all events (in-place).

    Strategy:
    - OEIL events: have summary_text or parliament_code
    - v2 events: have _v2_files in documents
    - Inferred events: neither of above

    Args:
        procedure: Procedure dict (modified in-place)
        logger: Optional logger
    """
    events = procedure.get("events", [])

    for event in events:
        # Determine source
        has_summary = event.get("summary_text") is not None
        has_parliament_code = event.get("parliament_code") is not None

        # Check for v2 documents
        has_v2_docs = any("_v2_files" in doc for doc in event.get("documents", []))

        # Determine source
        if has_summary or has_parliament_code:
            event["source"] = "OEIL"
        elif has_v2_docs:
            event["source"] = "EP_V2"
        elif event.get("documents"):
            # Has documents but not from v2, likely EUR-Lex
            event["source"] = "EUR_LEX"
        else:
            # No clear source, mark as inferred
            event["source"] = "INFERRED"

    if logger:
        sources = {}
        for event in events:
            src = event.get("source", "UNKNOWN")
            sources[src] = sources.get(src, 0) + 1

        logger.debug(
            f"Event provenance for {procedure.get('id')}: "
            f"{', '.join(f'{k}={v}' for k, v in sources.items())}"
        )


def mark_actor_name_resolution_status(
    procedure: Dict[str, Any],
    logger: Optional[Any] = None,
) -> None:
    """Mark actors with missing names as needing resolution (in-place).

    Strategy:
    - If mep_id exists but mep_name is None, mark as "pending"
    - If both exist, mark as "resolved"
    - If neither exist (committees, institutions), mark as "not_applicable"

    Args:
        procedure: Procedure dict (modified in-place)
        logger: Optional logger
    """
    actors = procedure.get("actors", [])
    pending_count = 0

    for actor in actors:
        mep_id = actor.get("mep_id")
        mep_name = actor.get("mep_name")

        if mep_id and not mep_name:
            actor["name_resolution"] = "pending"
            pending_count += 1
        elif mep_id and mep_name:
            actor["name_resolution"] = "resolved"
        else:
            # Committee, institution, or commission
            actor["name_resolution"] = "not_applicable"

    if logger and pending_count > 0:
        logger.info(
            f"{pending_count} actors with pending name resolution for {procedure.get('id')}"
        )


def finalize_silver_enrichment(
    procedure: Dict[str, Any],
    logger: Optional[Any] = None,
) -> None:
    """Apply final enrichments to Silver layer (in-place).

    1. Infer stage and status from events if not already set
    2. Add event provenance (OEIL vs V2 vs EUR-Lex)
    3. Mark actor name resolution status

    Args:
        procedure: Procedure dict (modified in-place)
        logger: Optional logger
    """
    # Normalize V2 stage: "Signature" means act is signed and awaiting Official Journal publication
    if procedure.get("stage") == "Signature":
        procedure["stage"] = "Awaiting Publication"
        if logger:
            logger.debug(
                f"Normalized stage for {procedure.get('id')}: Signature -> Awaiting Publication"
            )

    # 1. Infer stage and status from events if not already properly set
    current_status = procedure.get("status", "")
    current_stage = procedure.get("stage")

    # Only infer if stage is null/empty OR status is raw OEIL text (not normalized)
    normalized_statuses = [
        "completed",
        "in_progress",
        "in_force",
        "adopted",
        "rejected",
        "withdrawn",
        "terminated",
    ]
    needs_inference = not current_stage or (
        current_status and current_status.lower() not in normalized_statuses
    )

    if needs_inference:
        inferred_stage, inferred_status = _infer_stage_and_status_from_events(
            procedure.get("events", []), current_status=current_status, logger=logger
        )

        # Update stage if not set
        if not current_stage:
            procedure["stage"] = inferred_stage
            if logger:
                logger.debug(f"Inferred stage for {procedure.get('id')}: {inferred_stage}")

        # Update status if it's raw text (not normalized)
        if current_status and current_status.lower() not in normalized_statuses:
            procedure["status"] = inferred_status
            if logger:
                logger.debug(
                    f"Normalized status for {procedure.get('id')}: {current_status} -> {inferred_status}"
                )

    # 2. Add event provenance
    add_event_provenance(procedure, logger=logger)

    # 3. Mark actor name resolution status
    mark_actor_name_resolution_status(procedure, logger=logger)
