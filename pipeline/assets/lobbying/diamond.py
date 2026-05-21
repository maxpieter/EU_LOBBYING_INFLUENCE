"""Diamond stage: Upload lobbying meetings to Supabase.

Organisation upload is handled by eu_organizations_diamond.
This module is a pure meetings uploader.
"""

from typing import Any, Dict, List, Optional

from pipeline.models.lobbying_models import LobbyingMeeting, Organization


def upload_lobbying_data(
    meetings: List[LobbyingMeeting],
    organizations: List[Organization],
    supabase_resource: Any,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Upload lobbying meetings to Supabase.

    Args:
        meetings: List of LobbyingMeeting models
        organizations: Stub orgs created in silver for names not present in
            Supabase. Upserted before meetings to satisfy the FK, covering
            the race where bronze landed after the last orgs_diamond run.
        supabase_resource: Supabase resource instance
        logger: Optional logger

    Returns:
        Dict with upload results and view refresh status
    """
    results = {}

    # 0. Upsert silver-created stub orgs so meeting FKs resolve.
    # Failures here are non-fatal: the GIN trigram index on organizations.name
    # can push individual upserts past statement_timeout under DB load. Any
    # meetings referencing a stub that didn't make it are dropped from this
    # run and will be retried on re-materialisation.
    failed_stub_ids: set[str] = set()
    if organizations:
        if logger:
            logger.info(f"Upserting {len(organizations)} silver stub organisations")

        org_records = []
        seen_ids: set[str] = set()
        for o in organizations:
            if not o.id or o.id in seen_ids:
                continue
            seen_ids.add(o.id)
            org_records.append({
                "id": o.id,
                "name": o.name,
                "normalized_name": o.normalized_name,
                "official_name": o.official_name,
                "organization_type": (
                    o.organization_type.value if o.organization_type else None
                ),
                "eu_transparency_register_id": o.eu_transparency_register_id,
                "acronym": o.acronym,
            })

        stub_result = supabase_resource.batch_upsert(
            table="organizations",
            data=org_records,
            batch_size=50,
            on_conflict="id",
            logger=logger,
        )
        results["stubs"] = stub_result
        results["stubs_uploaded"] = stub_result["success"]
        if logger:
            logger.info(
                f"Stub upload complete: {stub_result['success']} succeeded, "
                f"{stub_result['failed']} failed"
            )
        if stub_result["failed"] > 0:
            failed_ids = stub_result.get("failed_ids") or []
            failed_stub_ids = set(failed_ids)
            if logger:
                logger.warning(
                    f"{stub_result['failed']} stub orgs failed to upsert "
                    f"(likely DB statement_timeout). Meetings referencing "
                    f"these stubs will be dropped from this run and retried next time. "
                    f"Failed IDs: {sorted(failed_stub_ids)[:5]}"
                    + ("..." if len(failed_stub_ids) > 5 else "")
                )

    # 1. Create placeholder MEPs for missing/inactive members
    if meetings:
        if logger:
            logger.info("Checking for missing MEPs and creating placeholders if needed")

        # Extract unique MEP IDs from meetings
        unique_mep_ids = set(m.mep_id for m in meetings if m.mep_id)

        if unique_mep_ids:
            if logger:
                logger.info(f"Found {len(unique_mep_ids)} unique MEP IDs in meetings")

            # Query which MEPs already exist
            try:
                existing_meps_response = (
                    supabase_resource.get_client()
                    .table("meps")
                    .select("id")
                    .in_("id", list(unique_mep_ids))
                    .execute()
                )
                existing_mep_ids = set(row["id"] for row in existing_meps_response.data)

                # Determine missing MEPs
                missing_mep_ids = unique_mep_ids - existing_mep_ids

                if missing_mep_ids:
                    if logger:
                        logger.info(
                            f"Found {len(missing_mep_ids)} MEPs not in database - creating placeholders"
                        )

                    # Create placeholder MEP records
                    placeholder_meps = []
                    for mep_id in missing_mep_ids:
                        placeholder = {
                            "id": mep_id,
                            "fullName": f"Former MEP (ID: {mep_id})",
                            "country": "Not available",
                            "politicalGroup": "Not available",
                            "nationalPoliticalGroup": None,
                            "status": "inactive",
                            "profile_url": None,
                            "image_url": None,
                            "role": "Former Member",
                            "birth_date": None,
                            "birth_place": None,
                        }
                        placeholder_meps.append(placeholder)

                    # Upload placeholder MEPs
                    results["placeholder_meps"] = supabase_resource.batch_upsert(
                        table="meps",
                        data=placeholder_meps,
                        batch_size=100,
                        on_conflict="id",
                        logger=logger,
                    )

                    if logger:
                        placeholder_result = results["placeholder_meps"]
                        logger.info(
                            f"Placeholder MEP upload complete: {placeholder_result['success']} succeeded, "
                            f"{placeholder_result['failed']} failed"
                        )
                else:
                    if logger:
                        logger.info("All MEPs exist in database - no placeholders needed")

            except Exception as e:
                if logger:
                    logger.error(f"Error checking for missing MEPs: {e}")
                raise

    # 3. Upload Meetings
    if meetings:
        if logger:
            logger.info(f"Preparing {len(meetings)} meetings for upload")

        # Drop any meetings whose stub org failed to upsert — uploading them
        # would FK-fail. They'll be retried on the next materialisation.
        if failed_stub_ids:
            before = len(meetings)
            meetings = [m for m in meetings if m.organization_id not in failed_stub_ids]
            dropped = before - len(meetings)
            if logger and dropped:
                logger.warning(
                    f"Dropped {dropped} meetings referencing {len(failed_stub_ids)} "
                    f"stub orgs that failed to upsert"
                )

        # Deduplicate meetings by ID to avoid "cannot affect row a second time" error
        seen_ids = set()
        unique_meetings = []
        duplicate_count = 0

        for m in meetings:
            if m.id not in seen_ids:
                unique_meetings.append(m)
                seen_ids.add(m.id)
            else:
                duplicate_count += 1

        if logger and duplicate_count > 0:
            logger.warning(f"Removed {duplicate_count} duplicate meetings (same ID)")

        meeting_records = []

        for m in unique_meetings:
            record = {
                "id": m.id,
                "mep_id": m.mep_id,
                "organization_id": m.organization_id,
                "meeting_date": m.meeting_date.isoformat() if m.meeting_date else None,
                "title": m.title,
                "location": m.location,
                "capacity": m.capacity,
                "related_procedure": m.related_procedure,
                "committee_acronym": m.committee_acronym,
                "meeting_type": m.meeting_type,
                "transparency_level": m.transparency_level,
                "org_match_method": m.org_match_method,
            }
            meeting_records.append(record)

        if logger:
            logger.info(f"Uploading {len(meeting_records)} meetings to Supabase")

        results["meetings"] = supabase_resource.batch_upsert(
            table="lobbying_meetings",
            data=meeting_records,
            batch_size=100,
            logger=logger,
        )
        results["meetings_uploaded"] = results["meetings"]["success"]

        if logger:
            meeting_result = results["meetings"]
            logger.info(
                f"Meeting upload complete: {meeting_result['success']} succeeded, {meeting_result['failed']} failed"
            )
            if meeting_result["failed"] > 0:
                logger.error(f"Failed to upload {meeting_result['failed']} meetings")
                raise RuntimeError(
                    f"Failed to upload {meeting_result['failed']} meetings to Supabase. "
                    f"Check logs for details."
                )
    else:
        if logger:
            logger.warning("No meetings provided for upload")

    # 4. Refresh materialized views
    if logger:
        logger.info("Refreshing materialized views...")

    try:
        supabase_resource.rpc("refresh_all_materialized_views")
        results["views_refreshed"] = {"success": 1, "failed": 0}
        if logger:
            logger.info("Successfully refreshed all materialized views")
    except Exception as e:
        results["views_refreshed"] = {"success": 0, "failed": 1}
        if logger:
            logger.warning(f"Could not refresh materialized views: {e}")
            logger.info(
                "Views can be refreshed manually later with: SELECT refresh_all_materialized_views();"
            )

    return results
