"""Lobbying asset definitions for EU Parliament transparency data.

Pipeline: Bronze → Silver → Diamond
- Bronze: Scrape meetings + load transparency register
- Silver: Entity resolution, link meetings to organizations
- Diamond: Upload to Supabase

Organisation resolution is handled by pipeline.assets.organizations.
This module handles meetings only (silver) and uploading (diamond).
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

from dagster import (
    AssetExecutionContext,
    AssetIn,
    Config,
    asset,
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
from .silver import process_meetings


@asset(
    name="eu_lobbying_bronze_meetings",
    group_name="eu_bronze",
    description=(
        "Scrape MEP lobbying meeting declarations from europarl.europa.eu for one weekly partition. "
        "Uses a pool of headless Chrome instances (Selenium) to paginate through results. Extracts "
        "MEP name, meeting date, title, attendee organisations, procedure references, committee "
        "codes, and Transparency Register IDs. Recovers MEP integer IDs via EP search facet matching."
    ),
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
    description=(
        "Download the full EU Transparency Register via the open data XML API "
        "(~104 MB, ~17k registered entities). Extracts name, acronym, TR ID, registration "
        "category, head office country, website, declared interests, policy areas, entity form, "
        "and employee headcount for each organisation."
    ),
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


@asset(
    name="eu_lobbying_silver_meetings",
    group_name="eu_silver",
    compute_kind="transformation",
    partitions_def=weekly_partitions,
    ins={
        "meetings_bronze": AssetIn("eu_lobbying_bronze_meetings"),
    },
    required_resource_keys={"supabase"},
    description=(
        "Lobbying meetings with organisation references resolved to canonical org IDs. "
        "Fetches organisations from Supabase and uses the unified OrgResolver 8-step "
        "cascade. Handles pipe-separated multi-org attendees."
    ),
)
def eu_lobbying_silver_meetings(
    context: AssetExecutionContext,
    meetings_bronze: List[Dict[str, Any]] | None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Silver layer: Link raw meetings to pre-resolved organisations.

    Output contains both meetings and any new stub organisations the
    resolver created for unmatched names. Diamond uploads the stubs
    before the meetings so FK constraints are satisfied even if
    orgs_diamond hasn't re-run since the new bronze landed.
    """
    partition_key = context.partition_key

    if meetings_bronze is None:
        raise ValueError(
            f"No meetings data file for partition {partition_key}. "
            f"Bronze layer returned None (file missing)."
        )

    if len(meetings_bronze) == 0:
        context.log.info(f"No meetings for partition {partition_key} (quiet week).")

    from pipeline.assets.organizations.resolution import OrgResolver

    # Fetch orgs from Supabase
    supabase: SupabaseResource = context.resources.supabase
    client = supabase.get_client()
    org_rows: List[Dict] = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            client.table("organizations")
            .select("id,name,normalized_name,official_name,acronym,eu_transparency_register_id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        org_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    org_models = [
        Organization(
            id=row["id"],
            name=row.get("name") or "",
            normalized_name=row.get("normalized_name"),
            official_name=row.get("official_name"),
            acronym=row.get("acronym"),
            eu_transparency_register_id=row.get("eu_transparency_register_id"),
        )
        for row in org_rows
    ]
    # Track which org IDs came from Supabase — stubs created during this
    # resolve pass are the diff, and those are what diamond needs to upsert.
    existing_ids = {o.id for o in org_models}
    resolver = OrgResolver(org_models)
    context.log.info(
        f"Built OrgResolver with {len(org_models)} organisations from Supabase"
    )

    from .silver import process_meetings_v2

    meetings = process_meetings_v2(meetings_bronze, resolver, context.log)

    new_stubs = [s for s in resolver.get_stubs() if s.id not in existing_ids]

    context.log.info(
        f"Created {len(meetings)} meetings and {len(new_stubs)} new stub orgs"
    )

    meetings_serialized = [m.model_dump(mode="json") for m in meetings]
    stubs_serialized = [s.model_dump(mode="json") for s in new_stubs]

    context.add_output_metadata({
        "meetings_count": len(meetings),
        "new_stubs_count": len(new_stubs),
    })

    return {"meetings": meetings_serialized, "stubs": stubs_serialized}


@asset(
    name="eu_lobbying_diamond",
    group_name="eu_diamond",
    description=(
        "Upsert lobbying meetings to Supabase (PostgreSQL). Uses deterministic "
        "primary keys for idempotent writes. Depends on eu_members_diamond for "
        "the mep_id FK and eu_organizations_diamond for the organization_id FK."
    ),
    compute_kind="loading",
    partitions_def=weekly_partitions,
    ins={
        "meetings": AssetIn("eu_lobbying_silver_meetings"),
        "meps_diamond": AssetIn("eu_members_diamond"),
        "orgs_diamond": AssetIn("eu_organizations_diamond"),
    },
)
def eu_lobbying_diamond(
    context: AssetExecutionContext,
    meetings: Dict[str, List[Dict[str, Any]]] | None,
    meps_diamond: Any,
    orgs_diamond: Any,
    supabase: SupabaseResource,
) -> Dict[str, Any]:
    """Diamond layer: Upload lobbying meetings to Supabase.

    Depends on eu_members_diamond and eu_organizations_diamond to ensure
    MEPs and orgs are uploaded first (FK constraints). Any stubs created
    in silver for this partition are also upserted here to cover the race
    between bronze scrapes and the next orgs_diamond refresh.
    """
    if meetings is None:
        context.log.warning("Silver meetings data is None - skipping upload")
        return {"meetings_uploaded": 0}

    meeting_dicts = meetings.get("meetings", []) if isinstance(meetings, dict) else meetings
    stub_dicts = meetings.get("stubs", []) if isinstance(meetings, dict) else []

    if not meeting_dicts:
        context.log.info("No meetings to upload")
        return {"meetings_uploaded": 0}

    context.log.info(
        f"Uploading {len(meeting_dicts)} meetings and {len(stub_dicts)} silver stubs"
    )

    meetings_models = [LobbyingMeeting(**m) for m in meeting_dicts]
    stubs_models = [Organization(**s) for s in stub_dicts]

    results = upload_lobbying_data(meetings_models, stubs_models, supabase, context.log)

    context.add_output_metadata({
        "meetings_uploaded": results.get("meetings_uploaded", 0),
        "stubs_uploaded": results.get("stubs_uploaded", 0),
    })

    return results


lobbying_assets = [
    eu_lobbying_bronze_meetings,
    eu_lobbying_bronze_organizations,
    eu_lobbying_silver_meetings,
    eu_lobbying_diamond,
]
