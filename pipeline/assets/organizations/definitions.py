"""Dagster asset definitions for consolidated organisation resolution.

Three assets:
- eu_organizations_silver: deterministic resolution (runs every time, ~minutes)
- eu_organizations_diamond: upload canonical org table to Supabase
- eu_organizations_fuzzy: rapidfuzz + AI for stubs + backfill (on demand, ~30min)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dagster import AssetExecutionContext, AssetIn, Config, asset

from pipeline.models.lobbying_models import Organization
from pipeline.resources.supabase import SupabaseResource

from pipeline.assets.lobbying.silver import process_transparency_data

from .resolution import OrgResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def _scan_lobbying_partition_orgs() -> list[dict[str, str]]:
    """Scan all lobbying bronze meeting partition files for org names + TR IDs.

    Pipe-splits attendees and deduplicates per meeting to mirror the per-org
    resolution done in lobbying silver (process_meetings_v2). Keeping the two
    paths aligned is what ensures stub IDs match between orgs_silver and
    lobbying_silver_meetings.
    """
    meetings_dir = _DATA_DIR / "eu_lobbying_bronze_meetings"
    if not meetings_dir.exists():
        return []

    org_refs: list[dict[str, str]] = []
    for path in sorted(meetings_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as f:
                meetings = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for m in meetings:
            attendees_raw = (m.get("attendees") or "").strip()
            tr_id = (m.get("lobbyist_id") or "").strip()
            if not attendees_raw:
                continue
            names = [n.strip() for n in attendees_raw.split("|") if n.strip()]
            seen: set[str] = set()
            unique_names: list[str] = []
            for n in names:
                if n.lower() in seen:
                    continue
                seen.add(n.lower())
                unique_names.append(n)
            for name in unique_names:
                ref_tr_id = tr_id if len(unique_names) == 1 else ""
                org_refs.append({"name": name, "tr_id": ref_tr_id})

    return org_refs


def _collect_commission_org_names(bronze_data: list[dict]) -> list[dict[str, str]]:
    """Extract org names from commission meetings bronze data.

    Returns list of {name, tr_id} dicts.
    """
    from pipeline.assets.commission_meetings.silver import parse_organizations_from_raw

    org_refs: list[dict[str, str]] = []
    for m in bronze_data:
        # TR IDs from PDF
        for tr_id in m.get("transparency_register_ids", []):
            org_refs.append({"name": "", "tr_id": tr_id})
        # Org names from HTML or raw text
        org_names = m.get("organizations", []) or parse_organizations_from_raw(
            m.get("organizations_raw", "")
        )
        for name in org_names:
            org_refs.append({"name": name, "tr_id": ""})

    return org_refs


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

@asset(
    name="eu_organizations_silver",
    group_name="eu_silver",
    compute_kind="transformation",
    deps=["eu_lobbying_bronze_meetings"],
    ins={
        "tr_bronze": AssetIn("eu_lobbying_bronze_organizations"),
        "commission_bronze": AssetIn("eu_commission_meetings_bronze"),
    },
    description=(
        "Canonical organisation records from the Transparency Register plus stub "
        "organisations discovered in lobbying meetings and commission meetings. "
        "Runs ALL org names through a unified 8-step deterministic cascade: "
        "TR ID exact, normalised name, cleaned name, acronym, parenthetical, "
        "prefix, TR ID extraction, stub creation. Unpartitioned — reads all "
        "lobbying partition files via filesystem scan."
    ),
)
def eu_organizations_silver(
    context: AssetExecutionContext,
    tr_bronze: Optional[List[Dict[str, Any]]],
    commission_bronze: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Build the canonical org table from TR register + all meeting sources."""
    if tr_bronze is None:
        context.log.error("TR bronze data is None")
        return []

    # Step 1: Process TR register into Organization models
    tr_orgs = process_transparency_data(tr_bronze, context.log)
    context.log.info(f"TR register: {len(tr_orgs)} organisations")

    # Step 2: Build resolver with TR orgs
    resolver = OrgResolver(tr_orgs)

    # Step 3: Collect org names from all lobbying partitions
    lobbying_refs = _scan_lobbying_partition_orgs()
    context.log.info(f"Lobbying partitions: {len(lobbying_refs)} org references")

    # Step 4: Collect org names from commission meetings
    commission_refs = _collect_commission_org_names(commission_bronze or [])
    context.log.info(f"Commission meetings: {len(commission_refs)} org references")

    # Step 5: Resolve all names through the cascade
    all_refs = lobbying_refs + commission_refs
    methods: dict[str, int] = {}

    for ref in all_refs:
        name = ref["name"]
        tr_id = ref["tr_id"] or None
        if not name and not tr_id:
            continue
        if not name and tr_id:
            # TR ID only (from commission PDF) — just verify it exists
            if tr_id in resolver._by_tr_id:
                methods["tr_id_only"] = methods.get("tr_id_only", 0) + 1
            continue
        _, method = resolver.resolve(name, tr_id)
        methods[method] = methods.get(method, 0) + 1

    all_orgs = resolver.get_all_organizations()
    stubs = resolver.get_stubs()

    context.log.info(f"Resolution methods: {methods}")
    context.log.info(
        f"Total: {len(all_orgs)} organisations "
        f"({resolver.canonical_count} canonical, {len(stubs)} stubs)"
    )

    context.add_output_metadata({
        "total_organizations": len(all_orgs),
        "canonical_count": resolver.canonical_count,
        "stub_count": len(stubs),
        "resolution_methods": methods,
    })

    return [o.model_dump(mode="json") for o in all_orgs]


@asset(
    name="eu_organizations_diamond",
    group_name="eu_diamond",
    compute_kind="loading",
    ins={"organizations": AssetIn("eu_organizations_silver")},
    description=(
        "Upsert the canonical organisation table to Supabase. Must run before "
        "lobbying and commission diamond assets (FK dependency)."
    ),
)
def eu_organizations_diamond(
    context: AssetExecutionContext,
    organizations: Optional[List[Dict[str, Any]]],
    supabase: SupabaseResource,
) -> Dict[str, Any]:
    """Upload canonical orgs to Supabase."""
    if not organizations:
        context.log.warning("No organizations to upload")
        return {"organizations_uploaded": 0}

    context.log.info(f"Uploading {len(organizations)} organisations to Supabase")

    org_records = []
    for o in organizations:
        record = {
            "id": o["id"],
            "name": o["name"],
            "normalized_name": o.get("normalized_name"),
            "official_name": o.get("official_name"),
            "website": o.get("website"),
            "organization_type": o.get("organization_type"),
            "industry_sector": o.get("industry_sector"),
            "country": o.get("country"),
            "eu_transparency_register_id": o.get("eu_transparency_register_id"),
            "description": o.get("description"),
            "founding_year": o.get("founding_year"),
            "employee_count_range": o.get("employee_count_range"),
            "annual_revenue_range": o.get("annual_revenue_range"),
            "transparency_score": o.get("transparency_score"),
            "scraped_at": o.get("scraped_at"),
            "logo_url": o.get("logo_url"),
            "social_media": o.get("social_media", {}),
            "key_personnel": o.get("key_personnel", []),
            "policy_focus_areas": o.get("policy_focus_areas", []),
            "acronym": o.get("acronym"),
            "city": o.get("city"),
            "address": o.get("address"),
            "post_code": o.get("post_code"),
            "level_of_interest": o.get("level_of_interest"),
            "interests_represented": o.get("interests_represented"),
            "form_of_entity": o.get("form_of_entity"),
            "source_of_funding": o.get("source_of_funding"),
            "dedup_status": o.get("dedup_status"),
        }
        org_records.append(record)

    result = supabase.batch_upsert(
        table="organizations",
        data=org_records,
        batch_size=100,
        on_conflict="id",
        logger=context.log,
    )

    context.log.info(
        f"Upload complete: {result['success']} succeeded, {result['failed']} failed"
    )
    if result["failed"] > 0:
        raise RuntimeError(f"Failed to upload {result['failed']} organisations")

    # Cleanup: delete orphaned stubs (no TR ID, no meeting references).
    # Non-fatal — the uploaded data is already correct; orphans just waste
    # rows. The RPC scans two anti-joins and can exceed statement_timeout
    # on large tables. Retry it manually or rely on a scheduled job.
    client = supabase.get_client()
    orphans_deleted = 0
    try:
        cleanup_resp = client.rpc("cleanup_orphaned_stubs").execute()
        orphans_deleted = (cleanup_resp.data or 0) if cleanup_resp.data else 0
        context.log.info(f"Cleanup: {orphans_deleted} orphaned stubs deleted")
    except Exception as e:
        context.log.warning(
            f"cleanup_orphaned_stubs failed (non-fatal): {e}. "
            "Uploaded orgs are correct; run the RPC manually when the DB is idle."
        )

    context.add_output_metadata({
        "organizations_uploaded": result["success"],
        "orphaned_stubs_deleted": orphans_deleted,
    })
    return result


class FuzzyConfig(Config):
    """Configuration for the fuzzy resolution asset."""
    dry_run: bool = True
    auto_accept_threshold: int = 96
    min_score: int = 93
    ai_batch_size: int = 50
    workers: int = 5
    backfill: bool = True


@asset(
    name="eu_organizations_fuzzy",
    group_name="eu_silver",
    compute_kind="ai_matching",
    ins={"organizations": AssetIn("eu_organizations_silver")},
    required_resource_keys={"supabase"},
    description=(
        "Fuzzy resolution of stub organisations via rapidfuzz + Anthropic API. "
        "Downloads TR XML dump, fuzzy-matches stubs locally, sends ambiguous "
        "candidates (score 50-96) to Claude Haiku for classification. Uses the "
        "dedup_status column on organizations table as persistent ledger — stubs "
        "with any status are never re-sent. Optionally backfills "
        "commission_meeting_organizations.organization_id for NULL rows. "
        "First run ~$3, subsequent runs ~$0."
    ),
)
def eu_organizations_fuzzy(
    context: AssetExecutionContext,
    organizations: Optional[List[Dict[str, Any]]],
    config: FuzzyConfig,
) -> Dict[str, Any]:
    """Resolve stubs via fuzzy matching + AI."""
    from .fuzzy import (
        backfill_unmatched_junction_rows,
        build_tr_lookup,
        download_tr_dump,
        resolve_stubs,
    )

    if not organizations:
        return {"resolved": 0, "backfilled": 0}

    supabase: SupabaseResource = context.resources.supabase
    client = supabase.get_client()

    # Build OrgResolver from current orgs
    org_models = [Organization(**o) for o in organizations]
    resolver = OrgResolver(org_models)

    # Stubs = orgs without a TR ID
    stubs = [o for o in org_models if not o.eu_transparency_register_id]

    # Hydrate dedup_status from Supabase — silver doesn't know about it
    # because it creates fresh orgs from bronze data
    if stubs:
        stub_ids = [s.id for s in stubs]
        context.log.info(f"Hydrating dedup_status for {len(stub_ids)} stubs from Supabase...")
        status_map: Dict[str, str] = {}
        for i in range(0, len(stub_ids), 500):
            batch = stub_ids[i:i + 500]
            resp = (
                client.table("organizations")
                .select("id,dedup_status")
                .in_("id", batch)
                .execute()
            )
            for row in (resp.data or []):
                if row.get("dedup_status"):
                    status_map[row["id"]] = row["dedup_status"]

        for stub in stubs:
            if stub.id in status_map:
                stub.dedup_status = status_map[stub.id]

        already = sum(1 for s in stubs if s.dedup_status)
        context.log.info(f"Stubs: {len(stubs)} total, {already} already classified, {len(stubs) - already} new")

    if not stubs and not config.backfill:
        return {"resolved": 0, "backfilled": 0}

    # Download TR dump + build lookup
    context.log.info("Downloading TR XML dump...")
    tr_orgs = download_tr_dump()
    tr_lookup = build_tr_lookup(tr_orgs)
    context.log.info(f"TR dump: {len(tr_orgs)} orgs, {len(tr_lookup['all_names'])} unique names")

    # Set up Anthropic client
    anthropic_client = None
    try:
        import anthropic
        from dotenv import dotenv_values
        env = dotenv_values(Path(__file__).parent.parent.parent.parent / ".env")
        api_key = env.get("ANTHROPIC_API_KEY")
        if api_key:
            anthropic_client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        context.log.warning("anthropic package not installed — skipping AI classification")

    # Phase 1: resolve stubs
    mappings = resolve_stubs(
        stubs, tr_lookup,
        supabase_client=client,
        logger=context.log,
        anthropic_client=anthropic_client,
        auto_accept_threshold=config.auto_accept_threshold,
        min_score=config.min_score,
        ai_batch_size=config.ai_batch_size,
        workers=config.workers,
        dry_run=config.dry_run,
    )

    # Phase 2: promote high-confidence matches
    # For each stub_id -> tr_id mapping:
    #   - If canonical org with that TR ID exists: relink all meetings to canonical
    #   - If no canonical exists: enrich the stub with TR ID (it becomes canonical)
    relinked = 0
    enriched = 0
    if not config.dry_run and mappings:
        for stub_id, tr_id in mappings.items():
            try:
                resp = (
                    client.table("organizations")
                    .select("id")
                    .eq("eu_transparency_register_id", tr_id)
                    .execute()
                )
                canonical_orgs = resp.data or []

                if canonical_orgs and canonical_orgs[0]["id"] != stub_id:
                    canonical_id = canonical_orgs[0]["id"]
                    # Relink lobbying meetings
                    client.table("lobbying_meetings").update(
                        {"organization_id": canonical_id}
                    ).eq("organization_id", stub_id).execute()
                    # Relink commission meeting organizations
                    client.table("commission_meeting_organizations").update(
                        {"organization_id": canonical_id}
                    ).eq("organization_id", stub_id).execute()
                    relinked += 1
                else:
                    # No canonical — enrich stub with TR ID
                    client.table("organizations").update(
                        {"eu_transparency_register_id": tr_id}
                    ).eq("id", stub_id).execute()
                    enriched += 1
            except Exception as e:
                context.log.warning(f"Failed to apply mapping {stub_id} -> {tr_id}: {e}")

        context.log.info(f"Phase 2: {relinked} relinked, {enriched} enriched")

    # Phase 3: backfill commission junction table
    backfilled = 0
    if config.backfill:
        backfilled = backfill_unmatched_junction_rows(
            client, resolver, tr_lookup,
            logger=context.log,
            anthropic_client=anthropic_client,
        )

    result = {
        "stubs_processed": len(stubs),
        "high_confidence_matches": len(mappings),
        "meetings_relinked": relinked,
        "stubs_enriched": enriched,
        "junction_rows_backfilled": backfilled,
        "dry_run": config.dry_run,
    }

    context.add_output_metadata(result)
    return result


# ---------------------------------------------------------------------------
# Asset list
# ---------------------------------------------------------------------------

organization_assets = [
    eu_organizations_silver,
    eu_organizations_diamond,
    eu_organizations_fuzzy,
]
