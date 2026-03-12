from typing import Any, Dict, List, Optional, Set

import requests

# Constants
OUTGOING_MEPS_API = "https://data.europarl.europa.eu/api/v2/meps/show-outgoing"


def fetch_outgoing_mep_ids(logger) -> Set[str]:
    """Fetch list of outgoing (inactive) MEP IDs from EU API.

    The API returns RDF/XML format with foaf:Person elements containing dcterms:identifier.

    Returns:
        Set of MEP IDs that are inactive/outgoing
    """
    try:
        logger.info(f"Fetching outgoing MEPs from {OUTGOING_MEPS_API}")
        response = requests.get(OUTGOING_MEPS_API, timeout=30)
        response.raise_for_status()

        # Parse RDF/XML response
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(response.content, "xml")
        outgoing_ids = set()

        # Find all foaf:Person elements and extract dcterms:identifier
        for person in soup.find_all("foaf:Person"):
            identifier = person.find("dcterms:identifier")
            if identifier and identifier.text:
                outgoing_ids.add(str(identifier.text.strip()))

        logger.info(f"Found {len(outgoing_ids)} outgoing MEPs")
        return outgoing_ids
    except Exception as e:
        logger.warning(f"Failed to fetch outgoing MEPs: {e}. Marking all as active.")
        return set()


def prepare_mep_record(mep: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare an MEP record for Supabase upsert."""
    # Convert mepid (string) to id (bigint) for meps table primary key
    mepid_str = mep.get("mepid")
    mep_id = int(mepid_str) if mepid_str and mepid_str.isdigit() else None

    return {
        "id": mep_id,  # meps table uses 'id' (bigint) as primary key
        "fullName": mep.get("full_name"),  # camelCase in database
        "country": mep.get("country"),
        "politicalGroup": mep.get("political_group"),  # camelCase in database
        "nationalPoliticalGroup": mep.get("national_party"),  # camelCase in database
        "profile_url": mep.get("profile_url"),
        "image_url": mep.get("image_url"),  # Note: 'image_url' not 'photo_url'
        "role": mep.get("role"),
        "birth_date": mep.get("birth_date"),
        "birth_place": mep.get("birth_place"),
        "status": mep.get("status", "active"),  # active/inactive status
        "socials": mep.get("socials", {}),  # Note: 'socials' not 'social_links'
        "committees": mep.get("committees", []),
        "navigation_links": mep.get("navigation_links", {}),
        "contacts": mep.get("contacts", []),
        "cv": mep.get("cv", []),  # Note: 'cv' not 'cv_entries'
        "assistants": mep.get("assistants", []),
        "declarations": mep.get("declarations", []),
        "past_meetings": mep.get("past_meetings", []),
        "declarations_summary": mep.get(
            "declarations_summary"
        ),  # Note: 'declarations_summary' not 'declaration_summary'
        "speech_summary": mep.get("speech_summary"),
        "speech_top_words": mep.get(
            "speech_top_words", []
        ),  # Note: 'speech_top_words' not 'top_topics'
        "speech_sources": mep.get("speech_sources", []),
    }


def create_placeholder_mep_record(mep_id: int, fullname: Optional[str] = None) -> Dict[str, Any]:
    """Create placeholder record for inactive MEP that doesn't exist in database.

    Args:
        mep_id: MEP ID number
        fullname: Optional MEP full name from data (if available)

    Returns:
        Dict with placeholder data and status set to 'inactive'
    """
    return {
        "id": mep_id,
        "fullName": fullname if fullname else "Member is inactive",
        "country": "Not available",
        "politicalGroup": "Not available",
        "nationalPoliticalGroup": "Not available",
        "profile_url": None,
        "image_url": None,
        "role": "Not available",
        "birth_date": None,
        "birth_place": "Not available",
        "status": "inactive",
        "socials": {},
        "committees": [],
        "navigation_links": {},
        "contacts": [],
        "cv": [],
        "assistants": [],
        "declarations": [],
        "past_meetings": [],
        "declarations_summary": None,
        "speech_summary": None,
        "speech_top_words": [],
        "speech_sources": [],
    }


def upload_meps(
    meps: List[Dict[str, Any]],
    supabase_resource: Any,
    logger: Optional[Any] = None,
) -> Dict[str, int]:
    """Upload MEPs to Supabase.

    Only uploads active MEPs with full data.
    Inactive MEP status updates are handled separately via mark_inactive_meps().
    """
    if logger:
        logger.info(f"Uploading {len(meps)} active MEPs with full data")

    # Prepare all records (all should be active at this point)
    records = [prepare_mep_record(mep) for mep in meps if mep.get("status") == "active"]

    # Upload via batch upsert
    result = supabase_resource.batch_upsert(
        table="meps",
        data=records,
        batch_size=50,
        on_conflict="id",  # meps table uses 'id' (bigint) as primary key
    )

    if logger:
        logger.info(f"Upload complete: {result['success']} success, {result['failed']} failed")
        if result["failed"] > 0:
            logger.error(f"Failed to upload {result['failed']} MEPs")
            raise RuntimeError(
                f"Failed to upload {result['failed']} MEPs to Supabase. " f"Check logs for details."
            )

    return result


def mark_inactive_meps(
    inactive_meps: List[Dict[str, Any]],
    supabase_resource: Any,
    logger: Optional[Any] = None,
) -> Dict[str, int]:
    """Mark MEPs as inactive - update status if they exist, insert with placeholders if they don't.

    For each inactive MEP:
    1. If MEP exists in database: Update only the status field to 'inactive' (preserving all other data)
    2. If MEP doesn't exist in database: Insert new record with status 'inactive' and placeholder data

    Args:
        inactive_meps: List of inactive MEP data objects (must contain 'mepid' and optionally 'full_name')
        supabase_resource: Supabase resource for database access
        logger: Optional logger for progress reporting

    Returns:
        Dict with 'updated', 'inserted', and 'failed' counts
    """
    if not inactive_meps:
        if logger:
            logger.info("No inactive MEPs to process")
        return {"updated": 0, "inserted": 0, "failed": 0}

    if logger:
        logger.info(f"Processing {len(inactive_meps)} inactive MEPs")

    client = supabase_resource.get_client()

    # Extract MEP IDs and create a mapping of ID -> full_name
    mep_id_to_data = {}
    for mep in inactive_meps:
        mepid_str = mep.get("mepid")
        if mepid_str and mepid_str.isdigit():
            mep_id = int(mepid_str)
            mep_id_to_data[mep_id] = {"id": mep_id, "fullname": mep.get("full_name")}

    if not mep_id_to_data:
        if logger:
            logger.warning("No valid MEP IDs found in inactive_meps")
        return {"updated": 0, "inserted": 0, "failed": 0}

    mep_ids = list(mep_id_to_data.keys())

    # Query database to find which MEPs already exist
    try:
        response = client.table("meps").select("id").in_("id", mep_ids).execute()
        existing_ids = {row["id"] for row in response.data}

        if logger:
            logger.info(
                f"Found {len(existing_ids)} existing MEPs (will update), "
                f"{len(mep_ids) - len(existing_ids)} missing MEPs (will insert with placeholders)"
            )
    except Exception as e:
        if logger:
            logger.error(f"Failed to query existing MEPs: {e}")
        raise RuntimeError(f"Failed to query existing MEPs: {e}")

    # Split MEPs into existing (update) and missing (insert) groups
    meps_to_update = [mep_id for mep_id in mep_ids if mep_id in existing_ids]
    meps_to_insert = [mep_id for mep_id in mep_ids if mep_id not in existing_ids]

    updated_count = 0
    inserted_count = 0
    failed_count = 0

    # Update existing MEPs (status only)
    for mep_id in meps_to_update:
        try:
            client.table("meps").update({"status": "inactive"}).eq("id", mep_id).execute()
            updated_count += 1

            if updated_count % 50 == 0 and logger:
                logger.debug(f"Progress: {updated_count}/{len(meps_to_update)} MEPs updated")
        except Exception as e:
            if logger:
                logger.error(f"Failed to update MEP {mep_id} status: {e}")
            failed_count += 1

    # Insert missing MEPs with placeholder data
    if meps_to_insert:
        placeholder_records = [
            create_placeholder_mep_record(
                mep_id=mep_id, fullname=mep_id_to_data[mep_id]["fullname"]
            )
            for mep_id in meps_to_insert
        ]

        # Use batch upsert for inserts
        try:
            result = supabase_resource.batch_upsert(
                table="meps",
                data=placeholder_records,
                batch_size=50,
                on_conflict="id",
            )
            inserted_count = result["success"]
            failed_count += result["failed"]

            if logger:
                logger.info(
                    f"Inserted {inserted_count} placeholder MEPs (failed: {result['failed']})"
                )
        except Exception as e:
            if logger:
                logger.error(f"Failed to insert placeholder MEPs: {e}")
            failed_count += len(meps_to_insert)

    if logger:
        logger.info(
            f"Inactive MEP processing complete: {updated_count} updated, "
            f"{inserted_count} inserted, {failed_count} failed"
        )
        if failed_count > 0:
            logger.error(f"Failed to process {failed_count} inactive MEPs")
            raise RuntimeError(
                f"Failed to process {failed_count} inactive MEPs. Check logs for details."
            )

    return {
        "updated": updated_count,
        "inserted": inserted_count,
        "failed": failed_count,
    }


def upload_meps_and_mark_inactive(
    meps: List[Dict[str, Any]],
    supabase_resource: Any,
    logger: Optional[Any] = None,
) -> Dict[str, int]:
    """Upload active MEPs and mark inactive ones from received data.

    Separates MEPs by status:
    - Active MEPs: Full upsert with all data
    - Inactive MEPs: Check existence, then either update status or insert with placeholders

    Args:
        meps: List of MEP records (both active and inactive)
        supabase_resource: Supabase resource for database access
        logger: Optional logger for progress reporting

    Returns:
        Dict with 'success', 'failed', 'inactive_updated', and 'inactive_inserted' counts
    """
    # Separate active and inactive MEPs from received data
    active_meps = [m for m in meps if m.get("status") == "active"]
    inactive_meps = [m for m in meps if m.get("status") == "inactive"]

    if logger:
        logger.info(
            f"Processing {len(meps)} MEPs: {len(active_meps)} active (full upload), "
            f"{len(inactive_meps)} inactive (update or insert with placeholders)"
        )

    # Upload active MEPs with full data
    result = upload_meps(
        meps=active_meps,
        supabase_resource=supabase_resource,
        logger=logger,
    )

    # Mark inactive MEPs (pass full MEP data for fullname extraction)
    inactive_result = mark_inactive_meps(
        inactive_meps=inactive_meps,
        supabase_resource=supabase_resource,
        logger=logger,
    )

    # Add inactive counts to result
    result["inactive_updated"] = inactive_result["updated"]
    result["inactive_inserted"] = inactive_result["inserted"]
    result["failed"] += inactive_result["failed"]

    return result
