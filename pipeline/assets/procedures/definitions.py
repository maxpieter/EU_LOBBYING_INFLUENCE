"""Procedure matching Dagster assets.

Four assets:
- eu_procedures_catalog: scrape all ~4,900 EU procedure IDs (no real titles)
- eu_procedures_titles: backfill real titles via the v2 detail endpoint
- eu_procedure_aliases: generate aliases for top 100 procedures via Opus
- eu_meeting_procedure_matcher: 5-step cascade to link meetings → procedures
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from dagster import AssetExecutionContext, AssetIn, Config, asset

from pipeline.resources.supabase import SupabaseResource


# ---------------------------------------------------------------------------
# 1. Catalog
# ---------------------------------------------------------------------------

@asset(
    name="eu_procedures_catalog",
    group_name="eu_bronze",
    compute_kind="extraction",
    description=(
        "Scrape ALL EU legislative procedures (~4,900) from the EP Open Data Portal. "
        "Only upserts minimal fields (id, process_id, title, procedure_type) — "
        "existing detailed rows keep their events/actors/docs."
    ),
)
def eu_procedures_catalog(
    context: AssetExecutionContext,
    supabase: SupabaseResource,
) -> Dict[str, Any]:
    """Fetch and upsert all EU procedure titles.

    Two sources:
    1. EP Open Data Portal listing (~5,000 COD/CNS/APP procedures)
    2. Orphan refs from lobbying_meetings.related_procedure that aren't
       in the catalog yet (covers INI reports + recently tabled proposals
       the listing hasn't picked up).
    """
    import re as _re
    from .catalog import fetch_procedure_catalog

    procedures = fetch_procedure_catalog(logger=context.log)

    if not procedures:
        return {"procedures_fetched": 0}

    catalog_ids = {p["id"] for p in procedures}

    # --- Phase 2: backfill orphan refs from lobbying meetings ---
    client = supabase.get_client()

    meeting_refs: set[str] = set()
    offset = 0
    while True:
        resp = (
            client.table("lobbying_meetings")
            .select("related_procedure")
            .not_.is_("related_procedure", "null")
            .neq("related_procedure", "")
            .range(offset, offset + 999)
            .execute()
        )
        batch = resp.data or []
        meeting_refs.update(r["related_procedure"] for r in batch if r.get("related_procedure"))
        if len(batch) < 1000:
            break
        offset += 1000

    # Also check what's already in the DB
    existing_ids: set[str] = set()
    offset = 0
    while True:
        resp = client.table("procedures").select("id").range(offset, offset + 999).execute()
        batch = resp.data or []
        existing_ids.update(r["id"] for r in batch)
        if len(batch) < 1000:
            break
        offset += 1000

    orphan_refs = meeting_refs - catalog_ids - existing_ids
    ref_pattern = _re.compile(r"^(\d{4})/(\d+[A-Z]?)\((\w+)\)$")

    orphan_records = []
    for ref in orphan_refs:
        m = ref_pattern.match(ref)
        if not m:
            continue
        year, num, proc_type = m.groups()
        orphan_records.append({
            "id": ref,
            "process_id": f"{year}-{num}",
            "procedure_type": proc_type,
            "title": ref,  # placeholder — eu_procedures_titles will enrich later
        })

    if orphan_records:
        context.log.info(
            f"Found {len(orphan_records)} orphan procedure refs in lobbying meetings "
            f"not in catalog — adding with placeholder titles"
        )
        procedures.extend(orphan_records)
    else:
        context.log.info("No orphan procedure refs to backfill")

    # --- Preserve existing real titles ---
    existing_titles: Dict[str, str] = {}
    ids = [p["id"] for p in procedures]
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        try:
            resp = client.table("procedures").select("id,title").in_("id", batch).execute()
            for row in (resp.data or []):
                if row.get("title"):
                    existing_titles[row["id"]] = row["title"]
        except Exception as e:
            context.log.warning(f"Could not fetch existing titles: {e}")

    ref_id_pattern = _re.compile(r"^\d{4}/\d{4}")
    preserved = 0
    for p in procedures:
        existing = existing_titles.get(p["id"])
        if existing and not ref_id_pattern.match(existing) and ref_id_pattern.match(p["title"]):
            p["title"] = existing
            preserved += 1

    if preserved:
        context.log.info(f"Preserved {preserved} existing real titles (would have overwritten with reference ID)")

    # --- Upsert ---
    result = supabase.batch_upsert(
        table="procedures",
        data=procedures,
        batch_size=100,
        on_conflict="id",
        logger=context.log,
    )

    context.log.info(
        f"Catalog: {result['success']} upserted ({len(orphan_records)} from orphan backfill), "
        f"{result['failed']} failed"
    )
    context.add_output_metadata({
        "procedures_fetched": len(procedures),
        "orphan_refs_backfilled": len(orphan_records),
        "upserted": result["success"],
    })
    return result


# ---------------------------------------------------------------------------
# 2. Titles (backfill real titles from v2 detail endpoint)
# ---------------------------------------------------------------------------

class TitlesConfig(Config):
    """Configuration for title enrichment."""
    workers: int = 1
    request_interval: float = 0.5  # Min seconds between API calls (global)
    max_procedures: int = 0  # 0 = no cap, process all rows missing titles


@asset(
    name="eu_procedures_titles",
    group_name="eu_silver",
    compute_kind="enrichment",
    deps=["eu_procedures_catalog"],
    required_resource_keys={"supabase"},
    description=(
        "Backfill real procedure titles from the EP Open Data v2 detail endpoint. "
        "The /procedures listing only returns the reference ID as a label, so the "
        "catalog stores `title = id` as a placeholder. This asset queries rows "
        "where that placeholder is still in place, fetches the real title from "
        "the per-procedure detail endpoint, and updates them. Idempotent — subsequent "
        "runs are cheap because the filter yields no rows once everything is titled."
    ),
)
def eu_procedures_titles(
    context: AssetExecutionContext,
    config: TitlesConfig,
) -> Dict[str, Any]:
    """Enrich placeholder procedure titles with data from the v2 detail endpoint."""
    from .titles import fetch_titles_concurrent

    supabase: SupabaseResource = context.resources.supabase
    client = supabase.get_client()

    # PostgREST can't filter `title = id` server-side, so page through and
    # detect placeholder rows client-side. 5k rows is fine — minimal payload.
    rows: List[Dict[str, Any]] = []
    offset = 0
    page_size = 1000
    while True:
        resp = (
            client.table("procedures")
            .select("id,process_id,title")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    needs_title = [
        r for r in rows
        if r.get("process_id") and (not r.get("title") or r["title"] == r["id"])
    ]
    context.log.info(
        f"Procedures scanned: {len(rows)}, needing titles: {len(needs_title)}"
    )

    if config.max_procedures and len(needs_title) > config.max_procedures:
        context.log.info(
            f"Capping to {config.max_procedures} procedures this run (remainder picks up next run)"
        )
        needs_title = needs_title[: config.max_procedures]

    if not needs_title:
        context.add_output_metadata({"fetched": 0, "updated": 0, "remaining": 0})
        return {"fetched": 0, "updated": 0, "remaining": 0}

    titles = fetch_titles_concurrent(
        process_ids=[r["process_id"] for r in needs_title],
        workers=config.workers,
        request_interval=config.request_interval,
        logger=context.log,
    )

    # We know every row in `needs_title` already exists (we just queried them),
    # so use UPDATE per row instead of UPSERT. UPSERT would hit NOT NULL on the
    # INSERT path (checked before ON CONFLICT DO UPDATE fires), forcing us to
    # include every NOT NULL column. UPDATE only touches the title column.
    updates_to_apply = [
        (r["id"], titles[r["process_id"]])
        for r in needs_title
        if r["process_id"] in titles
    ]
    if not updates_to_apply:
        context.log.warning("No titles recovered — nothing to update")
        context.add_output_metadata({
            "fetched": 0,
            "updated": 0,
            "remaining": len(needs_title),
        })
        return {"fetched": 0, "updated": 0, "remaining": len(needs_title)}

    updated = 0
    failed = 0
    for i, (proc_id, title) in enumerate(updates_to_apply, 1):
        try:
            client.table("procedures").update({"title": title}).eq("id", proc_id).execute()
            updated += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                context.log.error(f"Update failed for {proc_id}: {e}")
        if i % 200 == 0:
            context.log.info(f"  updated {i}/{len(updates_to_apply)}")

    remaining = len(needs_title) - updated
    context.log.info(
        f"Titles updated: {updated}, failed: {failed}, still without real title: {remaining}"
    )
    context.add_output_metadata({
        "fetched": len(titles),
        "updated": updated,
        "failed_updates": failed,
        "remaining": remaining,
    })
    return {
        "fetched": len(titles),
        "updated": updated,
        "remaining": remaining,
    }


# ---------------------------------------------------------------------------
# 3. Aliases
# ---------------------------------------------------------------------------

class AliasConfig(Config):
    """Configuration for alias generation."""
    dry_run: bool = True
    batch_size: int = 10
    workers: int = 3


@asset(
    name="eu_procedure_aliases",
    group_name="eu_silver",
    compute_kind="ai_generation",
    deps=["eu_procedures_catalog", "eu_procedures_titles"],
    required_resource_keys={"supabase"},
    description=(
        "Generate aliases (acronyms, short names, informal names) for the top 100 "
        "detailed procedures using Claude Opus 4.6. Only processes procedures that "
        "don't already have aliases. ~$10 for first run."
    ),
)
def eu_procedure_aliases(
    context: AssetExecutionContext,
    config: AliasConfig,
) -> Dict[str, Any]:
    """Generate and upsert procedure aliases."""
    from .aliases import generate_aliases_for_procedures

    supabase: SupabaseResource = context.resources.supabase
    client = supabase.get_client()

    # Fetch top 100 detailed procedures (ones with events/actors populated)
    resp = (
        client.table("procedures")
        .select("id,title,procedure_type,subjects,proposal_date,events")
        .not_.is_("events", "null")
        .neq("events", "[]")
        .limit(200)
        .execute()
    )
    detailed = resp.data or []
    context.log.info(f"Detailed procedures (with events): {len(detailed)}")

    # Check which already have aliases
    alias_resp = (
        client.table("procedure_aliases")
        .select("procedure_id")
        .execute()
    )
    already_aliased = {r["procedure_id"] for r in (alias_resp.data or [])}

    to_process = [p for p in detailed if p["id"] not in already_aliased]
    context.log.info(f"Need aliases: {len(to_process)} (skipping {len(already_aliased)} already aliased)")

    if not to_process:
        return {"generated": 0}

    # Set up Anthropic client
    try:
        import anthropic
        from dotenv import dotenv_values
        env = dotenv_values(Path(__file__).parent.parent.parent.parent / ".env")
        api_key = env.get("ANTHROPIC_API_KEY")
        if not api_key:
            context.log.warning("No ANTHROPIC_API_KEY — cannot generate aliases")
            return {"generated": 0}
        anthropic_client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        context.log.warning("anthropic package not installed")
        return {"generated": 0}

    aliases = generate_aliases_for_procedures(
        to_process, anthropic_client,
        logger=context.log,
        batch_size=config.batch_size,
        workers=config.workers,
    )

    if config.dry_run:
        context.log.info(f"Dry run: would upsert {len(aliases)} aliases")
        for a in aliases[:20]:
            context.log.info(f"  {a['procedure_id']}: {a['alias']} ({a['alias_type']})")
        context.add_output_metadata({"generated": len(aliases), "dry_run": True})
        return {"generated": len(aliases), "dry_run": True}

    # Upsert aliases
    if aliases:
        result = supabase.batch_upsert(
            table="procedure_aliases",
            data=aliases,
            batch_size=100,
            on_conflict="alias,procedure_id",
            logger=context.log,
        )
        context.log.info(f"Aliases: {result['success']} upserted")
        context.add_output_metadata({"generated": len(aliases), "upserted": result["success"]})
        return result

    return {"generated": 0}


# ---------------------------------------------------------------------------
# 3. Matcher
# ---------------------------------------------------------------------------

class MatcherConfig(Config):
    """Configuration for procedure matching."""
    dry_run: bool = True
    ai_batch_size: int = 30
    workers: int = 5


@asset(
    name="eu_meeting_procedure_matcher",
    group_name="eu_silver",
    compute_kind="ai_matching",
    deps=["eu_procedures_catalog", "eu_procedures_titles", "eu_procedure_aliases"],
    required_resource_keys={"supabase"},
    description=(
        "Link meetings to legislative procedures via a 3-step cascade: "
        "1) exact ID from related_procedure field, 2) alias exact match, "
        "3) alias substring → AI confirmation (with org name context). "
        "Temporal filtering ensures meetings only match active procedures. "
        "Writes to meeting_procedure_links junction table."
    ),
)
def eu_meeting_procedure_matcher(
    context: AssetExecutionContext,
    config: MatcherConfig,
) -> Dict[str, Any]:
    """Run procedure matching on all unprocessed meetings."""
    from .matching import ProcedureMatcher, match_meetings

    supabase: SupabaseResource = context.resources.supabase
    client = supabase.get_client()

    # Load procedures
    procedures: list[dict] = []
    offset = 0
    while True:
        resp = (
            client.table("procedures")
            .select("id,title,procedure_type,proposal_date,decision_date,last_activity_date")
            .eq("is_deleted", False)
            .range(offset, offset + 999)
            .execute()
        )
        procedures.extend(resp.data or [])
        if len(resp.data or []) < 1000:
            break
        offset += 1000
    context.log.info(f"Loaded {len(procedures)} procedures")

    # Load aliases
    aliases: list[dict] = []
    offset = 0
    while True:
        resp = (
            client.table("procedure_aliases")
            .select("procedure_id,alias,alias_type")
            .range(offset, offset + 999)
            .execute()
        )
        aliases.extend(resp.data or [])
        if len(resp.data or []) < 1000:
            break
        offset += 1000
    context.log.info(f"Loaded {len(aliases)} aliases")

    # Build matcher
    matcher = ProcedureMatcher(procedures, aliases)
    context.log.info(f"Matcher: {matcher.procedure_count} procedures, {matcher.alias_count} alias entries")

    # Load unprocessed meetings (match_status IS NULL)
    # Lobbying
    lobbying: list[dict] = []
    offset = 0
    while True:
        resp = (
            client.table("lobbying_meetings")
            .select("id,title,meeting_date,related_procedure,match_status,organization_id")
            .is_("match_status", "null")
            .range(offset, offset + 999)
            .execute()
        )
        lobbying.extend(resp.data or [])
        if len(resp.data or []) < 1000:
            break
        offset += 1000

    # Commission
    commission: list[dict] = []
    offset = 0
    while True:
        resp = (
            client.table("commission_meetings")
            .select("id,subject,meeting_date,points_raised,match_status")
            .is_("match_status", "null")
            .range(offset, offset + 999)
            .execute()
        )
        commission.extend(resp.data or [])
        if len(resp.data or []) < 1000:
            break
        offset += 1000

    context.log.info(f"Unprocessed meetings: {len(lobbying)} lobbying, {len(commission)} commission")

    # Resolve org_id → org_name for lobbying meetings (context for AI prompt)
    org_ids = list({m["organization_id"] for m in lobbying if m.get("organization_id")})
    org_names: dict[str, str] = {}
    for start in range(0, len(org_ids), 500):
        batch = org_ids[start : start + 500]
        try:
            resp = client.table("organizations").select("id,name").in_("id", batch).execute()
            for row in (resp.data or []):
                org_names[row["id"]] = row["name"]
        except Exception:
            pass
    for m in lobbying:
        m["org_name"] = org_names.get(m.get("organization_id"))
    context.log.info(f"Resolved {len(org_names)} org names for AI context")

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
        context.log.warning("anthropic not installed — skipping AI step")

    # Run matching
    stats = match_meetings(
        lobbying, commission, matcher, client,
        logger=context.log,
        anthropic_client=anthropic_client,
        ai_batch_size=config.ai_batch_size,
        workers=config.workers,
        dry_run=config.dry_run,
    )

    context.log.info(f"Results: {stats}")
    context.add_output_metadata(stats)
    return stats


# ---------------------------------------------------------------------------
# Asset list
# ---------------------------------------------------------------------------

procedure_matching_assets = [
    eu_procedures_catalog,
    eu_procedures_titles,
    eu_procedure_aliases,
    eu_meeting_procedure_matcher,
]
