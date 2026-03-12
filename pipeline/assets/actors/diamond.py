"""Diamond layer: Upload actors to Supabase."""

from typing import Any, Dict, List, Optional


def prepare_actor_record(actor: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare actor record for Supabase upsert.

    Maps Actor fields to Supabase actors table schema.
    Field names are aligned with meps table where applicable.

    Args:
        actor: Actor dictionary with all processed fields

    Returns:
        Dictionary with only fields needed for Supabase actors table
    """
    return {
        # Core identity (aligned with meps table)
        "actor_id": actor.get("actor_id"),
        "fullName": actor.get("fullName"),  # Aligned with meps.fullName
        "actor_type": actor.get("actor_type"),
        "profile_url": actor.get("profile_url"),  # Aligned with meps.profile_url
        "image_url": actor.get("image_url"),
        "role": actor.get("role"),
        "country": actor.get("country"),
        "portfolio": actor.get("portfolio"),
        "term_start": actor.get("term_start"),
        "term_end": actor.get("term_end"),
        "description": actor.get("description"),
        "parliament": actor.get("parliament", "eu"),
        # Contact (aligned with meps.contacts)
        "contacts": actor.get("contacts"),  # Aligned with meps.contacts
        # Structured data (like MEP equivalents)
        "team": actor.get("team", []),  # Cabinet members (like assistants)
        "responsibilities": actor.get("responsibilities"),  # Policy areas
        "declarations": actor.get("declarations", []),  # Declaration of interests
        "past_meetings": actor.get("past_meetings", []),  # Transparency meetings
        # Activity data
        "speeches": actor.get("speeches", []),
        "latest_news": actor.get("latest_news", []),
        "calendar": actor.get("calendar", []),
        "transparency": actor.get("transparency"),
        "biography": actor.get("biography", []),
        "documents": actor.get("documents", []),
        # AI-generated fields
        "role_summary": actor.get("role_summary"),
        "key_topics": actor.get("key_topics", []),
        "declarations_summary": actor.get("declarations_summary"),
        "embedding": actor.get("embedding"),
        "embedding_model": actor.get("embedding_model"),
        # Meta
        "status": actor.get("status", "active"),
    }


def upload_actors(
    actors: List[Dict[str, Any]],
    supabase_resource: Any,
    logger: Optional[Any] = None,
) -> Dict[str, int]:
    """Upload actors to Supabase actors table.

    Args:
        actors: List of Actor records with all processed fields
        supabase_resource: SupabaseResource instance
        logger: Optional Dagster logger for logging progress

    Returns:
        Dictionary with 'success' and 'failed' counts
    """
    records = [prepare_actor_record(actor) for actor in actors]

    if logger:
        logger.info(f"Uploading {len(records)} actors to Supabase")

    result = supabase_resource.batch_upsert(
        table="actors",
        data=records,
        batch_size=50,
        on_conflict="actor_id",
    )

    if logger:
        logger.info(f"Upload complete: {result['success']} success, {result['failed']} failed")

    return result
