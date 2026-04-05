"""Diamond stage: Upload structured lobbying data to Supabase."""

from typing import Any, Dict, List, Optional

from pipeline.models.lobbying_models import LobbyingMeeting, Organization

from .fuzzy_match import resolve_stubs


def upload_lobbying_data(
    meetings: List[LobbyingMeeting],
    organizations: List[Organization],
    supabase_resource: Any,
    logger: Optional[Any] = None,
) -> Dict[str, Dict[str, int]]:
    """Upload lobbying data to Supabase.

    Args:
        meetings: List of LobbyingMeeting models
        organizations: List of Organization models
        supabase_resource: Supabase resource instance
        logger: Optional logger

    Returns:
        Dict with upload results and view refresh status
    """
    results = {}

    # 0. Resolve stub orgs via fuzzy matching before upload
    if organizations and meetings:
        organizations, meetings = resolve_stubs(
            organizations, meetings, supabase_resource, logger
        )

    # 1. Upload Organizations
    if organizations:
        if logger:
            logger.info(f"Preparing {len(organizations)} organizations for upload")

        org_records = []
        for org in organizations:
            record = {
                "id": org.id,
                "name": org.name,
                "normalized_name": org.normalized_name,
                "official_name": org.official_name,
                "website": org.website,
                "organization_type": (
                    org.organization_type.value if org.organization_type else None
                ),
                "industry_sector": (org.industry_sector.value if org.industry_sector else None),
                "country": org.country,
                "eu_transparency_register_id": org.eu_transparency_register_id,
                "description": org.description,
                "founding_year": org.founding_year,
                "employee_count_range": org.employee_count_range,
                "annual_revenue_range": org.annual_revenue_range,
                "transparency_score": org.transparency_score,
                "scraped_at": org.scraped_at.isoformat() if org.scraped_at else None,
                "logo_url": org.logo_url,
                "social_media": org.social_media,
                "key_personnel": org.key_personnel,
                "policy_focus_areas": org.policy_focus_areas,
                "acronym": org.acronym,
                "city": org.city,
                "address": org.address,
                "post_code": org.post_code,
                "level_of_interest": org.level_of_interest,
                "interests_represented": org.interests_represented,
                "form_of_entity": org.form_of_entity,
                "source_of_funding": org.source_of_funding,
            }
            org_records.append(record)

        if logger:
            logger.info(f"Uploading {len(org_records)} organizations to Supabase")

        # Upsert organizations - use 'id' as the primary key for conflict resolution
        results["organizations"] = supabase_resource.batch_upsert(
            table="organizations",
            data=org_records,
            batch_size=100,
            on_conflict="id",
            logger=logger,
        )

        if logger:
            org_result = results["organizations"]
            logger.info(
                f"Organization upload complete: {org_result['success']} succeeded, {org_result['failed']} failed"
            )
            if org_result["failed"] > 0:
                logger.error(
                    f"Failed to upload {org_result['failed']} organizations - check Supabase logs"
                )
                raise RuntimeError(
                    f"Failed to upload {org_result['failed']} organizations to Supabase. "
                    f"Check logs for details."
                )
    else:
        if logger:
            logger.warning("No organizations provided for upload")

    # 2. Create placeholder MEPs for missing/inactive members
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
        org_lookup = {org.id: org for org in organizations}

        for m in unique_meetings:
            # Try to find org
            org = org_lookup.get(m.organization_id)

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
