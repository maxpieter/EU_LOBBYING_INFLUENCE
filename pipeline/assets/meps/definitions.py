"""MEP Assets - Bronze → Diamond (no Gold/AI layer).

Scrapes MEP data from EU Parliament and uploads to Supabase.
"""

import os
from typing import Any, Dict, List, Optional, Set

from dagster import AssetExecutionContext, AssetIn, Config, asset

from pipeline.models.members import Member
from pipeline.resources.supabase import SupabaseResource

from .scraper import scrape_all_meps

PARLIAMENT_ID = "eu"


class MembersBronzeConfig(Config):
    """Configuration for MEP bronze scraping."""

    max_meps: Optional[int] = None


def fetch_outgoing_mep_ids(logger) -> Set[str]:
    """Fetch list of outgoing (inactive) MEP IDs from EU API."""
    from .diamond import fetch_outgoing_mep_ids as _fetch

    return _fetch(logger)


@asset(
    name="eu_members_bronze",
    group_name="eu_bronze",
    description=(
        "Scrape all active MEPs from the EU Parliament XML member list and individual profile "
        "pages. Collects: full name, nationality, political group, national party, committee "
        "memberships with roles, and active/inactive status. Creates minimal placeholder records "
        "for outgoing (inactive) MEPs to preserve referential integrity with historical meeting data."
    ),
    compute_kind="scraping",
)
def eu_members_bronze(
    context: AssetExecutionContext,
    config: MembersBronzeConfig,
) -> List[Dict[str, Any]]:
    """Bronze layer: Complete MEP scraping for active members."""
    max_meps = config.max_meps or (
        int(os.getenv("MEP_TEST_LIMIT")) if os.getenv("MEP_TEST_LIMIT") else None
    )

    if max_meps:
        context.log.info(f"TEST MODE: Limiting to {max_meps} MEPs")

    context.log.info("Scraping active MEPs from EU Parliament")

    outgoing_ids = fetch_outgoing_mep_ids(context.log)

    active_members = scrape_all_meps(
        logger=context.log,
        max_meps=max_meps,
    )

    context.log.info(
        f"Scraped {len(active_members)} active MEPs, "
        f"{len(outgoing_ids)} inactive MEPs identified"
    )

    # Create minimal records for inactive MEPs
    inactive_members = []
    for mep_id in outgoing_ids:
        inactive_members.append(
            {
                "mepid": mep_id,
                "full_name": "Inactive MEP",
                "country": "Unknown",
                "political_group": "Unknown",
                "status": "inactive",
            }
        )

    all_members = active_members + inactive_members

    # Validate with Member model
    validated_members = []
    for m in all_members:
        try:
            m["parliament"] = PARLIAMENT_ID
            if "status" not in m:
                m["status"] = "active"
            validated_members.append(Member(**m).model_dump())
        except Exception as e:
            context.log.warning(f"Validation error for MEP {m.get('mepid')}: {e}")

    context.add_output_metadata(
        {
            "count": len(validated_members),
            "active_meps": sum(1 for m in validated_members if m.get("status") == "active"),
            "outgoing_meps": len(outgoing_ids),
        }
    )

    return validated_members


@asset(
    name="eu_members_diamond",
    group_name="eu_diamond",
    description=(
        "Upsert active MEP records to the Supabase meps table and mark outgoing MEPs as "
        "inactive. Uses deterministic primary keys for idempotent re-runs. Must run before "
        "the lobbying diamond asset (foreign key dependency)."
    ),
    compute_kind="upload",
    ins={"bronze_data": AssetIn("eu_members_bronze")},
)
def eu_members_diamond(
    context: AssetExecutionContext,
    supabase: SupabaseResource,
    bronze_data: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Diamond layer: Upload active MEPs and mark inactive ones."""
    if not bronze_data:
        return {"success": 0, "failed": 0, "marked_inactive": 0}

    from .diamond import upload_meps_and_mark_inactive

    result = upload_meps_and_mark_inactive(
        meps=bronze_data,
        supabase_resource=supabase,
        logger=context.log,
    )

    context.add_output_metadata(
        {
            "uploaded_active": result.get("success", 0),
            "failed": result.get("failed", 0),
            "marked_inactive": result.get("marked_inactive", 0),
        }
    )

    return result


members_assets = [
    eu_members_bronze,
    eu_members_diamond,
]
