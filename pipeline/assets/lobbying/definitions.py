"""Lobbying asset definitions for EU Parliament transparency data.

Pipeline: Bronze → Silver → Diamond
- Bronze: Scrape meetings + load transparency register
- Silver: Entity resolution, link meetings to organizations
- Diamond: Upload to Supabase
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

from dagster import (
    AssetExecutionContext,
    AssetIn,
    AssetOut,
    Config,
    asset,
    multi_asset,
)

from pipeline.models.lobbying_models import LobbyingMeeting, Organization
from pipeline.partitions.definitions import weekly_partitions
from pipeline.resources.supabase import SupabaseResource

from .bronze import (
    close_browser_pool,
    fetch_meetings_scraped,
    load_transparency_register,
)
from .diamond import upload_lobbying_data
from .silver import (
    process_meetings,
    process_transparency_data,
)


@asset(
    name="eu_lobbying_bronze_meetings",
    group_name="eu_bronze",
    description="Extract lobbying meetings via progressive web-scraping for the partition week.",
    compute_kind="extraction",
    partitions_def=weekly_partitions,
)
def eu_lobbying_bronze_meetings(
    context: AssetExecutionContext,
) -> List[Dict[str, Any]]:
    """Bronze layer: Extract lobbying meetings via progressive web-scraping."""
    partition_key = context.partition_key
    context.log.info(f"Fetching lobbying meetings for week starting {partition_key}")

    start_date = datetime.strptime(partition_key, "%Y-%m-%d")
    end_date = start_date + timedelta(days=6)

    start_str = start_date.strftime("%d/%m/%Y")
    end_str = end_date.strftime("%d/%m/%Y")

    context.log.info(f"Fetching week: {start_date.date()} to {end_date.date()}")

    try:
        meetings, expected_count = fetch_meetings_scraped(
            from_date=start_str,
            to_date=end_str,
            logger=context.log,
        )
        context.log.info(f"Scraped {len(meetings)} meetings for week {partition_key}")

        context.add_output_metadata(
            {
                "meetings_count": len(meetings),
                "expected_count": expected_count,
            }
        )

        return meetings
    except Exception as e:
        context.log.error(f"Error scraping meetings: {e}")
        raise RuntimeError(
            f"Failed to fetch meetings for partition {partition_key}. "
            f"Not materializing to prevent data loss."
        ) from e
    finally:
        close_browser_pool()


@asset(
    name="eu_lobbying_bronze_organizations",
    group_name="eu_bronze",
    description="Extract lobbying organizations from transparency register XML.",
    compute_kind="extraction",
)
def eu_lobbying_bronze_organizations(
    context: AssetExecutionContext,
) -> List[Dict[str, Any]]:
    """Bronze layer: Extract organizations from transparency register."""
    context.log.info("Loading transparency register data from API")

    organizations = load_transparency_register(logger=context.log)

    context.log.info(f"Loaded {len(organizations)} organizations")
    return organizations


@multi_asset(
    outs={
        "eu_lobbying_silver_organizations": AssetOut(
            group_name="eu_silver",
            description="Organizations from transparency register + extracted from meetings.",
        ),
        "eu_lobbying_silver_meetings": AssetOut(
            group_name="eu_silver",
            description="Lobbying meetings linked to organizations.",
        ),
    },
    compute_kind="transformation",
    partitions_def=weekly_partitions,
    ins={
        "meetings_bronze": AssetIn("eu_lobbying_bronze_meetings"),
        "organizations_bronze": AssetIn("eu_lobbying_bronze_organizations"),
    },
)
def eu_lobbying_silver(
    context: AssetExecutionContext,
    meetings_bronze: List[Dict[str, Any]] | None,
    organizations_bronze: List[Dict[str, Any]] | None,
):
    """Silver layer: Process raw data into Organization and LobbyingMeeting models.

    Upserts ALL transparency register organizations (not just those in meetings)
    to ensure fields like interests_represented are always populated.
    """
    partition_key = context.partition_key

    if meetings_bronze is None:
        raise ValueError(
            f"No meetings data file for partition {partition_key}. "
            f"Bronze layer returned None (file missing)."
        )

    if len(meetings_bronze) == 0:
        context.log.info(f"No meetings for partition {partition_key} (quiet week).")

    if organizations_bronze is None:
        context.log.error(f"Organizations bronze data is None for partition {partition_key}.")
        return [], []

    context.log.info(
        f"Processing {len(meetings_bronze)} meetings against "
        f"{len(organizations_bronze)} transparency records"
    )

    # 1. Process Transparency Data (all register orgs)
    existing_orgs = process_transparency_data(organizations_bronze, context.log)

    # 2. Process Meetings (and extract new orgs not in register)
    meetings, new_orgs = process_meetings(meetings_bronze, existing_orgs, context.log)

    # 3. Keep ALL register orgs + new orgs from meetings
    all_orgs = existing_orgs + new_orgs

    context.log.info(
        f"Total: {len(existing_orgs)} orgs from register, "
        f"{len(new_orgs)} new orgs from meetings = {len(all_orgs)} total"
    )

    # Serialize to dicts for JSON IO Manager
    orgs_serialized = [o.model_dump(mode="json") for o in all_orgs]
    meetings_serialized = [m.model_dump(mode="json") for m in meetings]

    context.add_output_metadata(
        metadata={
            "meetings_count": len(meetings),
            "organizations_new": len(new_orgs),
        },
        output_name="eu_lobbying_silver_meetings",
    )

    context.add_output_metadata(
        metadata={
            "organizations_total": len(all_orgs),
            "organizations_existing": len(orgs_used_in_meetings),
            "organizations_new": len(new_orgs),
        },
        output_name="eu_lobbying_silver_organizations",
    )

    return orgs_serialized, meetings_serialized


@asset(
    name="eu_lobbying_diamond",
    group_name="eu_diamond",
    description="Upload lobbying data to Supabase.",
    compute_kind="loading",
    partitions_def=weekly_partitions,
    ins={
        "organizations": AssetIn("eu_lobbying_silver_organizations"),
        "meetings": AssetIn("eu_lobbying_silver_meetings"),
        "meps_diamond": AssetIn("eu_members_diamond"),
    },
)
def eu_lobbying_diamond(
    context: AssetExecutionContext,
    organizations: List[Dict[str, Any]] | None,
    meetings: List[Dict[str, Any]] | None,
    meps_diamond: Any,
    supabase: SupabaseResource,
) -> Dict[str, Any]:
    """Diamond layer: Upload processed lobbying data to Supabase.

    Depends on eu_members_diamond to ensure MEPs are uploaded first
    (lobbying_meetings.mep_id has a foreign key constraint to meps.id).
    """
    if organizations is None or meetings is None:
        context.log.warning("Silver data is None - skipping upload")
        return {"organizations_uploaded": 0, "meetings_uploaded": 0}

    if not organizations and not meetings:
        context.log.info("No organizations or meetings to upload")
        return {"organizations_uploaded": 0, "meetings_uploaded": 0}

    context.log.info(
        f"Uploading {len(meetings)} meetings and {len(organizations)} organizations"
    )

    meetings_models = [LobbyingMeeting(**m) for m in meetings]
    orgs_models = [Organization(**o) for o in organizations]

    results = upload_lobbying_data(meetings_models, orgs_models, supabase, context.log)

    context.add_output_metadata(
        {
            "meetings_uploaded": results.get("meetings_uploaded", 0),
            "organizations_uploaded": results.get("organizations_uploaded", 0),
        }
    )

    return results


class OrgDedupConfig(Config):
    """Configuration for the org deduplication asset."""

    dry_run: bool = True
    """When True (default), pass 4 (TR web search) writes a CSV report but
    makes no database changes. Passes 1-3 are always applied."""


@asset(
    name="eu_lobbying_org_dedup",
    group_name="eu_silver",
    compute_kind="dedup",
    required_resource_keys={"supabase"},
    description=(
        "Deduplicate stub organisations by relinking lobbying meetings to canonical "
        "Transparency Register entries. Four passes: TR ID extraction from name, "
        "case-insensitive name match, acronym match, TR web search with AI confirmation."
    ),
)
def eu_lobbying_org_dedup(context: AssetExecutionContext, config: OrgDedupConfig):
    from .org_dedup import run_org_dedup

    supabase: SupabaseResource = context.resources.supabase
    client = supabase.get_client()

    stats = run_org_dedup(client, logger=context.log, dry_run=config.dry_run)

    context.add_output_metadata({
        "tr_id_relinked": stats["tr_id_relinked"],
        "name_relinked": stats["name_relinked"],
        "acronym_relinked": stats["acronym_relinked"],
        "tr_search_high": stats.get("tr_search_high", 0),
        "tr_search_medium": stats.get("tr_search_medium", 0),
        "tr_search_applied": stats.get("tr_search_applied", 0),
        "total_relinked": stats["total"],
        "dry_run": config.dry_run,
    })
    return stats


lobbying_assets = [
    eu_lobbying_bronze_meetings,
    eu_lobbying_bronze_organizations,
    eu_lobbying_silver,
    eu_lobbying_diamond,
    eu_lobbying_org_dedup,
]
