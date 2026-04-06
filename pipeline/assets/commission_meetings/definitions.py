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
    group_name="eu_bronze",
    compute_kind="scraper",
    required_resource_keys={"supabase"},
    description=(
        "Scrape European Commission meeting records from the EC Transparency Initiative "
        "(ec.europa.eu/transparencyinitiative). Covers both EP9 (2019-2024, direct commissioner "
        "discovery) and EP10 (2024-2029, actor profile URLs from Supabase). Downloads and parses "
        "meeting minutes PDFs using pypdf to extract: date, subject, organisations present, and "
        "free-text 'points raised' capturing substantive policy positions expressed by lobbyists. "
        "Deduplicates by meeting ID across terms."
    ),
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
    group_name="eu_silver",
    compute_kind="python",
    ins={"bronze_data": AssetIn("eu_commission_meetings_bronze")},
    required_resource_keys={"supabase"},
    description=(
        "Entity resolution for commission meeting organisations. Matches organisation names from "
        "scraped meetings against canonical Transparency Register entries using: (1) TR ID exact "
        "match, (2) normalised name match, (3) prefix match for short names (e.g. 'Toyota' → "
        "'Toyota Motor Europe'). Unmatched names are recorded as stubs with deterministic hash IDs "
        "for later batch deduplication. Requires the organisations table to be populated first "
        "via the lobbying pipeline."
    ),
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
    group_name="eu_diamond",
    compute_kind="supabase",
    ins={"silver_data": AssetIn("eu_commission_meetings_silver")},
    required_resource_keys={"supabase"},
    description=(
        "Upsert commission meetings and meeting-organisation junction records to Supabase. "
        "Creates entries in commission_meetings and commission_meeting_organizations tables "
        "with deterministic primary keys for idempotent re-runs."
    ),
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
