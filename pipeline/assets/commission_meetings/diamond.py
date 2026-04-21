"""Diamond layer: Upload commission meetings to Supabase."""

import json
from typing import Any, Optional


def upload_commission_meetings(
    silver_data: dict[str, list[dict]],
    supabase,
    logger: Optional[Any] = None,
) -> dict[str, int]:
    """Upload meetings and organization links to Supabase.

    Args:
        silver_data: {"meetings": [...], "meeting_organizations": [...]}
        supabase: SupabaseResource instance
        logger: Dagster logger
    """
    meetings = silver_data["meetings"]
    meeting_orgs = silver_data["meeting_organizations"]

    if not meetings:
        if logger:
            logger.warning("No meetings to upload")
        return {"meetings": 0, "meeting_organizations": 0}

    # Prepare meeting records for upload
    meeting_records = []
    for m in meetings:
        record = {
            "id": m["id"],
            "actor_id": m.get("actor_id"),
            "commissioner_name": m["commissioner_name"],
            "commissioner_portfolio": m.get("commissioner_portfolio"),
            "host_id": m.get("host_id"),
            "meeting_type": m.get("meeting_type", "commissioner"),
            "meeting_date": m.get("meeting_date"),
            "location": m.get("location"),
            "subject": m.get("subject"),
            "commission_representatives": json.dumps(m.get("commission_representatives", [])),
            "organizations_raw": m.get("organizations_raw"),
            "transparency_register_ids": m.get("transparency_register_ids", []),
            "points_raised": m.get("points_raised"),
            "conclusions": m.get("conclusions"),
            "ares_number": m.get("ares_number"),
            "minutes_url": m.get("minutes_url"),
            "source_url": m.get("source_url"),
            "raw_data": json.dumps(m.get("raw_data", {})),
        }
        meeting_records.append(record)

    # Upload meetings
    if logger:
        logger.info(f"Uploading {len(meeting_records)} meetings...")

    result = supabase.batch_upsert(
        table="commission_meetings",
        data=meeting_records,
        batch_size=50,
        logger=logger,
    )

    if logger:
        logger.info(
            f"Meetings: {result['success']} success, {result['failed']} failed"
        )

    # Upload organization links
    # First delete existing links for these meetings (to handle updates)
    meeting_ids = [m["id"] for m in meetings]
    client = supabase.get_client()

    try:
        for batch_start in range(0, len(meeting_ids), 50):
            batch_ids = meeting_ids[batch_start : batch_start + 50]
            client.table("commission_meeting_organizations").delete().in_(
                "meeting_id", batch_ids
            ).execute()
    except Exception as e:
        if logger:
            logger.warning(f"Failed to clear old org links (may not exist yet): {e}")

    # Insert new org links
    # Check which organization IDs exist in DB to avoid FK violations.
    # Stub IDs (org_...) from OrgResolver may not be in the organizations table.
    existing_org_ids: set[str] = set()
    all_org_ids = [o["organization_id"] for o in meeting_orgs if o.get("organization_id")]
    for batch_start in range(0, len(all_org_ids), 500):
        batch_ids = all_org_ids[batch_start : batch_start + 500]
        try:
            resp = client.table("organizations").select("id").in_("id", batch_ids).execute()
            existing_org_ids.update(r["id"] for r in (resp.data or []))
        except Exception:
            pass

    org_records = []
    nulled_count = 0
    for org in meeting_orgs:
        record = {
            "meeting_id": org["meeting_id"],
            "organization_name": org["organization_name"],
        }
        org_id = org.get("organization_id")
        if org_id and org_id in existing_org_ids:
            record["organization_id"] = org_id
        elif org_id:
            nulled_count += 1
        if org.get("eu_transparency_register_id"):
            record["eu_transparency_register_id"] = org["eu_transparency_register_id"]
        org_records.append(record)

    if logger and nulled_count:
        logger.info(f"Nulled {nulled_count} org IDs not found in organizations table")

    org_result = {"success": 0, "failed": 0}
    if org_records:
        if logger:
            logger.info(f"Uploading {len(org_records)} organization links...")

        org_result = supabase.batch_upsert(
            table="commission_meeting_organizations",
            data=org_records,
            batch_size=50,
            logger=logger,
        )

        if logger:
            logger.info(
                f"Org links: {org_result['success']} success, "
                f"{org_result['failed']} failed"
            )

    return {
        "meetings": result["success"],
        "meeting_organizations": org_result["success"],
    }
