"""Commission Meetings Assets.

Pipeline for European Commission meetings.
Data source: EC Transparency Initiative (ec.europa.eu/transparency-initiative).

Bronze: Excel exports for meeting lists + PDF minutes for EP10
Silver: Entity resolution (org names → organizations table)
Diamond: Upload to Supabase

Supports both EP9 (2019-2024) and EP10 (2024-2029) commissions.
EP9 meeting minutes PDFs are no longer available from the EC.
"""

import re as _re
from typing import Any

from dagster import AssetIn, Config, asset

from pipeline.resources.supabase import SupabaseResource


class CommissionMeetingsBronzeConfig(Config):
    """Configuration for commission meetings bronze asset."""
    terms: list[str] = ["EP9", "EP10"]
    max_commissioners: int = 0
    """Limit number of commissioners to scrape (0 = all). Useful for testing."""


@asset(
    name="eu_commission_meetings_bronze",
    group_name="eu_bronze",
    compute_kind="scraper",
    description=(
        "Scrape European Commission meeting records from the EC Transparency Initiative. "
        "Uses structured Excel exports for meeting lists (date, location, orgs, subject) "
        "and HTML page scraping for minutes PDF UUIDs. Downloads and parses EP10 meeting "
        "minutes PDFs using pypdf. EP9 minutes are no longer available from the EC. "
        "Covers both EP9 (2019-2024) and EP10 (2024-2029) commissions."
    ),
)
def eu_commission_meetings_bronze(context, config: CommissionMeetingsBronzeConfig):
    from .bronze import _make_session, scrape_commission_meetings_v2

    session = _make_session()
    all_meetings = []

    # EP10 (2024-2029): Excel exports + PDF minutes
    if "EP10" in config.terms:
        context.log.info("=== EP10 (2024-2029) ===")
        ep10 = scrape_commission_meetings_v2(
            session, college_id=0, logger=context.log,
            max_commissioners=config.max_commissioners,
            skip_minutes=False,
        )
        all_meetings.extend(ep10)
        context.log.info(f"EP10: {len(ep10)} meetings")

    # EP9 (2019-2024): Excel exports only (PDFs no longer available)
    if "EP9" in config.terms:
        context.log.info("=== EP9 (2019-2024) ===")
        try:
            ep9 = scrape_commission_meetings_v2(
                session, college_id=1, logger=context.log,
                max_commissioners=config.max_commissioners,
                skip_minutes=True,  # EP9 PDFs return 404
            )
            all_meetings.extend(ep9)
            context.log.info(f"EP9: {len(ep9)} meetings")
        except Exception as e:
            context.log.warning(f"EP9 scrape failed (non-fatal): {e}")
            context.log.info("Continuing with EP10 data only")

    session.close()

    # Deduplicate: pass 1 — by meeting ID
    seen_ids = set()
    unique = []
    for m in all_meetings:
        mid = m.get("id")
        if mid and mid not in seen_ids:
            seen_ids.add(mid)
            unique.append(m)

    id_dupes = len(all_meetings) - len(unique)

    # Deduplicate: pass 2 — by content (catches cross-term duplicates)
    content_seen = set()
    final = []
    for m in unique:
        content_key = (
            (m.get("commissioner_name") or "").lower().strip(),
            m.get("meeting_date", ""),
            _re.sub(r"\s+", " ", (m.get("subject") or "").lower().strip()),
            _re.sub(r"\s+", " ", (m.get("organizations_raw") or "").lower().strip()),
        )
        if content_key not in content_seen:
            content_seen.add(content_key)
            final.append(m)

    content_dupes = len(unique) - len(final)
    context.log.info(
        f"Dedup: {len(all_meetings)} raw → {len(final)} unique "
        f"({id_dupes} ID dupes, {content_dupes} content dupes)"
    )
    return final


@asset(
    name="eu_commission_meetings_silver",
    group_name="eu_silver",
    compute_kind="python",
    ins={
        "bronze_data": AssetIn("eu_commission_meetings_bronze"),
    },
    required_resource_keys={"supabase"},
    description=(
        "Entity resolution for commission meeting organisations. Fetches the "
        "canonical organisations table directly from Supabase and uses the "
        "unified OrgResolver 8-step cascade for matching."
    ),
)
def eu_commission_meetings_silver(context, bronze_data: list[dict]):
    from pipeline.assets.organizations.resolution import OrgResolver
    from pipeline.models.lobbying_models import Organization

    from .silver import build_meeting_records

    supabase: SupabaseResource = context.resources.supabase
    client = supabase.get_client()
    org_rows = []
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
    context.log.info(f"Fetched {len(org_rows)} organisations from Supabase")

    # Build Organization models and OrgResolver
    orgs = []
    for row in org_rows:
        orgs.append(Organization(
            id=row["id"],
            name=row.get("name") or "",
            normalized_name=row.get("normalized_name"),
            official_name=row.get("official_name"),
            acronym=row.get("acronym"),
            eu_transparency_register_id=row.get("eu_transparency_register_id"),
        ))
    resolver = OrgResolver(orgs)
    context.log.info(f"Built OrgResolver with {len(orgs)} organisations")

    return build_meeting_records(bronze_data, resolver, logger=context.log)


@asset(
    name="eu_commission_meetings_diamond",
    group_name="eu_diamond",
    compute_kind="supabase",
    ins={
        "silver_data": AssetIn("eu_commission_meetings_silver"),
    },
    required_resource_keys={"supabase"},
    description=(
        "Upsert commission meetings and meeting-organisation junction records to Supabase. "
        "Creates entries in commission_meetings and commission_meeting_organizations tables "
        "with deterministic primary keys for idempotent re-runs. Depends on "
        "eu_organizations_diamond for the organisation FK constraint."
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
