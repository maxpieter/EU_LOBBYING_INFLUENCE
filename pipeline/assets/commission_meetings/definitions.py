"""Commission Meetings Assets.

New pipeline (not in parl8) for European Commission meetings.
Data source: EC Transparency Initiative + Meeting Minutes PDFs.

Bronze: Scrape commissioner pages → meetings → parse PDFs
Silver: Entity resolution (org names → organizations table)
Diamond: Upload to Supabase

Supports both EP9 (2019-2024) and EP10 (2024-2029) commissions.
"""

from dagster import AssetIn, Config, asset

from pipeline.resources.supabase import SupabaseResource


class CommissionMeetingsBronzeConfig(Config):
    """Configuration for commission meetings bronze asset."""
    terms: list[str] = ["EP9", "EP10"]


@asset(
    name="eu_commission_meetings_bronze",
    group_name="commission_meetings",
    compute_kind="scraper",
    deps=["eu_actors_diamond"],
    required_resource_keys={"supabase"},
    description="Scrape Commission meetings from EC Transparency Initiative + parse minutes PDFs (EP9 + EP10)",
)
def eu_commission_meetings_bronze(context, config: CommissionMeetingsBronzeConfig):
    from .bronze import scrape_commission_meetings, scrape_ep9_commission_meetings

    all_meetings = []

    # EP10 (2024-2029): actor-based discovery
    if "EP10" in config.terms:
        supabase: SupabaseResource = context.resources.supabase
        result = supabase.select(
            "actors",
            columns="actor_id,\"fullName\",portfolio,profile_url,actor_type",
        )
        actors = [a for a in (result.data or []) if a.get("profile_url")]
        context.log.info(f"EP10: Loaded {len(actors)} actors with profile URLs")
        ep10_meetings = scrape_commission_meetings(context, actors=actors)
        all_meetings.extend(ep10_meetings)
        context.log.info(f"EP10: {len(ep10_meetings)} meetings")

    # EP9 (2019-2024): direct commissioner discovery
    if "EP9" in config.terms:
        ep9_meetings = scrape_ep9_commission_meetings(context)
        all_meetings.extend(ep9_meetings)
        context.log.info(f"EP9: {len(ep9_meetings)} meetings")

    # Deduplicate by meeting ID (in case of overlap)
    seen_ids = set()
    unique = []
    for m in all_meetings:
        mid = m.get("id")
        if mid and mid not in seen_ids:
            seen_ids.add(mid)
            unique.append(m)

    context.log.info(f"Total unique meetings: {len(unique)} (from {len(all_meetings)} raw)")
    return unique


@asset(
    name="eu_commission_meetings_silver",
    group_name="commission_meetings",
    compute_kind="python",
    ins={"bronze_data": AssetIn("eu_commission_meetings_bronze")},
    required_resource_keys={"supabase"},
    description="Entity resolution: link meeting organizations to transparency register",
)
def eu_commission_meetings_silver(context, bronze_data: list[dict]):
    from .silver import process_commission_meetings

    supabase: SupabaseResource = context.resources.supabase

    # Fetch existing organizations for entity resolution
    context.log.info("Fetching existing organizations for entity resolution...")
    result = supabase.select(
        "organizations",
        columns="id,name,normalized_name,official_name,acronym,eu_transparency_register_id",
    )
    existing_orgs = result.data if result.data else []
    context.log.info(f"Loaded {len(existing_orgs)} organizations")
    if not existing_orgs:
        context.log.warning(
            "Organizations table is empty — run the lobbying pipeline first "
            "to populate it from the Transparency Register. "
            "Proceeding with 0 org matches."
        )

    return process_commission_meetings(bronze_data, existing_orgs, logger=context.log)


@asset(
    name="eu_commission_meetings_diamond",
    group_name="commission_meetings",
    compute_kind="supabase",
    ins={"silver_data": AssetIn("eu_commission_meetings_silver")},
    required_resource_keys={"supabase"},
    description="Upload commission meetings and organization links to Supabase",
)
def eu_commission_meetings_diamond(context, silver_data: dict):
    from .diamond import upload_commission_meetings

    supabase: SupabaseResource = context.resources.supabase
    result = upload_commission_meetings(silver_data, supabase, logger=context.log)
    context.log.info(f"Upload complete: {result}")
    return result


commission_meetings_assets = [
    eu_commission_meetings_bronze,
    eu_commission_meetings_silver,
    eu_commission_meetings_diamond,
]
