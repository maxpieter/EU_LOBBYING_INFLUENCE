"""Bronze layer: OEIL scraping + v2 API enrichment for EU legislation.

Discovery mode selection:
  - v2_feed  (default for recent partitions, ≤30 days old): Uses EP Open Data v2
              /procedures/feed which captures ANY procedural activity with
              millisecond-precision timestamps.
  - oeil_catalog (fallback for older partitions): Uses OEIL XML catalog filtered
              by lastpubdate.  Suitable for backfill when the 30-day feed window
              does not reach the target week.

The mode is chosen automatically based on partition recency.  It can be forced
via the ``LEGISLATION_DISCOVERY_MODE`` environment variable ("v2_feed" or
"oeil_catalog").
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from dagster import AssetExecutionContext, asset

from pipeline.models.legislation import Procedure
from pipeline.partitions.definitions import (
    get_week_range_from_partition,
    weekly_partitions,
)

# How far back (in days) the v2 feed reliably covers.
# The EP API caps the feed at ~30 days regardless of start-date parameter.
V2_FEED_LOOKBACK_DAYS = 28

DiscoveryMode = Literal["v2_feed", "oeil_catalog"]


# ---------------------------------------------------------------------------
# Discovery helpers (pure functions, testable without Dagster)
# ---------------------------------------------------------------------------


def select_discovery_mode(
    partition_start: datetime,
    env_override: Optional[str] = None,
) -> DiscoveryMode:
    """Choose the discovery mode for a given partition.

    Args:
        partition_start: Start datetime of the partition week.
        env_override: Value of LEGISLATION_DISCOVERY_MODE env var (or None).

    Returns:
        "v2_feed" if the partition is within the past 28 days (or forced),
        "oeil_catalog" otherwise.
    """
    if env_override:
        normalized = env_override.strip().lower()
        if normalized in ("v2_feed", "oeil_catalog"):
            return normalized  # type: ignore[return-value]

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=V2_FEED_LOOKBACK_DAYS)
    # Ensure comparison is timezone-aware
    if partition_start.tzinfo is None:
        partition_start = partition_start.replace(tzinfo=timezone.utc)

    return "v2_feed" if partition_start >= cutoff else "oeil_catalog"


def discover_via_v2_feed(
    start_date: datetime,
    end_date: datetime,
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Discover updated procedures using the EP v2 procedures feed.

    Returns a list of minimal procedure dicts with at minimum:
        process_id  : str
        oeil_ref    : str  -- OEIL reference format, e.g. "2023/0404(COD)"
        title       : str
        process_type: str
        updated_at  : datetime | None

    Args:
        start_date: Partition week start (used to filter feed entries).
        end_date:   Partition week end.
        logger:     Optional Dagster logger.
    """
    from .v2_feed import TRACKED_PROCEDURE_TYPES, fetch_procedures_feed_for_window

    entries = fetch_procedures_feed_for_window(
        start_dt=start_date,
        end_dt=end_date,
        process_types=TRACKED_PROCEDURE_TYPES,
        logger=logger,
    )

    # Convert feed entries to the shape expected downstream
    result = []
    for entry in entries:
        pid = entry["process_id"]
        pt = entry.get("process_type", "")
        # Reconstruct OEIL reference from process_id and type
        # "2023-0404" + "COD" -> "2023/0404(COD)"
        oeil_ref = _process_id_to_oeil_ref(pid, pt)
        result.append(
            {
                "process_id": pid,
                "oeil_ref": oeil_ref,
                "title": entry.get("title", ""),
                "process_type": pt,
                "updated_at": entry.get("updated_at"),
            }
        )

    return result


def discover_via_oeil_catalog(
    partition_year: int,
    start_date: datetime,
    end_date: datetime,
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Discover updated procedures using OEIL XML catalog (legacy path).

    Returns a list of minimal procedure dicts with at minimum:
        process_id  : str
        oeil_ref    : str
        title       : str
        process_type: str

    Args:
        partition_year: Year of the partition (used to scope OEIL catalog fetch).
        start_date:     Partition week start.
        end_date:       Partition week end.
        logger:         Optional Dagster logger.
    """
    from .oeil_catalog import (
        OEIL_PROCEDURE_TYPES,
        fetch_oeil_procedure_catalog,
        filter_procedures_by_date,
    )

    def _log(msg: str, level: str = "info") -> None:
        if logger:
            getattr(logger, level)(msg)

    all_metadata: List[Dict[str, Any]] = []

    for proc_name, oeil_type_code in OEIL_PROCEDURE_TYPES.items():
        _log(f"Fetching {proc_name} procedure metadata from OEIL XML for {partition_year}")
        proc_metadata = fetch_oeil_procedure_catalog(
            year=partition_year,
            procedure_type=oeil_type_code,
            logger=logger,
            use_cache=True,
            return_metadata=True,
        )
        all_metadata.extend(proc_metadata)
        _log(
            f"Found {len(proc_metadata)} {proc_name} procedures in OEIL catalog for {partition_year}"
        )

    _log(f"Total procedures from OEIL XML: {len(all_metadata)}")

    filtered_ids = filter_procedures_by_date(
        procedures_metadata=all_metadata,
        start_date=start_date,
        end_date=end_date,
        buffer_weeks=0,
        logger=logger,
    )

    # Build result with same shape as v2_feed discovery
    proc_id_to_meta = {m["process_id"]: m for m in all_metadata}
    result = []
    for pid in filtered_ids:
        meta = proc_id_to_meta.get(pid, {})
        result.append(
            {
                "process_id": pid,
                "oeil_ref": meta.get("oeil_ref", ""),
                "title": meta.get("title", ""),
                "process_type": _oeil_ref_to_type(meta.get("oeil_ref", "")),
                "updated_at": None,  # OEIL does not provide precise timestamps
            }
        )

    _log(f"OEIL catalog discovery: {len(result)} procedures to scrape")
    return result


def _process_id_to_oeil_ref(process_id: str, process_type: str) -> str:
    """Convert process_id + type to OEIL reference format.

    Examples:
        "2023-0404", "COD" -> "2023/0404(COD)"
        "2025-0058", "CNS" -> "2025/0058(CNS)"
    """
    if not process_id:
        return ""
    # "2023-0404" -> "2023/0404"
    slash = process_id.replace("-", "/", 1)
    if process_type:
        return f"{slash}({process_type})"
    return slash


def _oeil_ref_to_type(oeil_ref: str) -> str:
    """Extract procedure type from OEIL reference.

    Example: "2023/0404(COD)" -> "COD"
    """
    if not oeil_ref or "(" not in oeil_ref:
        return ""
    return oeil_ref.split("(")[-1].rstrip(")")


def build_procedures_from_oeil(
    scrape_results: List[Dict[str, Any]],
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Convert OEIL HTML scrape results into Procedure model dicts.

    This is the same conversion logic that was previously inlined in the
    asset function, extracted here for testability.

    Args:
        scrape_results: List of dicts returned by ``scrape_multiple_procedures``.
        logger:         Optional Dagster logger.

    Returns:
        List of dicts that conform to the ``Procedure`` model shape.
    """

    def _log(msg: str, level: str = "info") -> None:
        if logger:
            getattr(logger, level)(msg)

    procedures = []

    for proc in scrape_results:
        if proc.get("error"):
            _log(
                f"Skipping procedure {proc.get('id')} due to scraping error: {proc.get('error')}",
                "warning",
            )
            continue

        oeil_ref = proc.get("id", "")
        process_id = oeil_ref.split("(")[0].replace("/", "-") if "(" in oeil_ref else ""

        events_converted = _convert_oeil_events(process_id, proc.get("events", []))

        has_joint_committee = any(
            "joint committee" in e.get("title", "").lower() for e in proc.get("events", [])
        )
        actors_converted = _convert_oeil_actors(proc.get("actors", []), has_joint_committee)

        procedures.append(
            {
                "id": oeil_ref,
                "process_id": process_id,
                "title": proc.get("title"),
                "procedure_type": proc.get("procedure_type"),
                "status": proc.get("status"),
                "stage": None,
                "policy_area": proc.get("policy_area"),
                "subjects": proc.get("subjects", []),
                "legal_basis": proc.get("legal_basis", []),
                "events": events_converted,
                "actors": actors_converted,
                "oeil_url": proc.get("oeil_url"),
                "proposal_date": proc.get("proposal_date")
                or next(
                    (
                        e.get("event_date")
                        for e in proc.get("events", [])
                        if e.get("activity_type")
                        in ["PROPOSAL_PUBLICATION", "Legislative proposal published"]
                    ),
                    None,
                ),
                "decision_date": proc.get("decision_date"),
                "last_activity_date": max(
                    [e.get("event_date") for e in proc.get("events", []) if e.get("event_date")],
                    default=None,
                ),
                "commission_document": proc.get("commission_document"),
                "amending_acts": proc.get("amending_acts", []),
                "background_documents": proc.get("background_documents", []),
                "celex_number": proc.get("celex_number"),
                "eurlex_proposal_url": proc.get("eurlex_proposal_url"),
                "eurlex_final_act_url": proc.get("eurlex_final_act_url"),
            }
        )

    return procedures


def _convert_oeil_events(
    process_id: str,
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert raw OEIL scraper events into ProcedureEvent dicts."""
    converted = []
    for event in events:
        event_date_str = event.get("event_date")
        event_date = datetime.fromisoformat(event_date_str).date() if event_date_str else None

        summary_id = event.get("summary_id")
        description = f"Summary ID: {summary_id}" if summary_id else None

        event_type = event.get("event_type", "Activity")
        activity_type = event.get("activity_type", "")
        original_title = event.get("activity_type_original", activity_type)

        event_dict: Dict[str, Any] = {
            "event_id": f"{process_id}-{event_date_str}",
            "event_date": event_date,
            "event_type": event_type,
            "activity_type": activity_type,
            "title": original_title,
            "description": description,
            "documents": event.get("documents", []),
        }

        if event.get("summary_text"):
            event_dict["summary_text"] = event["summary_text"]
        if event.get("parliament_code"):
            event_dict["parliament_code"] = event["parliament_code"]

        converted.append(event_dict)

    return converted


def _convert_oeil_actors(
    actors: List[Dict[str, Any]],
    has_joint_committee: bool,
) -> List[Dict[str, Any]]:
    """Convert raw OEIL scraper actors into ProcedureActor dicts."""
    converted = []
    for actor in actors:
        role = actor.get("role")
        if has_joint_committee and role == "committee_for_opinion":
            committee_code = actor.get("committee_code", "")
            if committee_code in ["BUDG", "REGI", "AGRI", "PECH"]:
                role = "joint_committee_responsible"

        converted.append(
            {
                "actor_type": actor.get("actor_type"),
                "role": role,
                "mep_id": actor.get("mep_id"),
                "mep_name": actor.get("mep_name"),
                "committee_code": actor.get("committee_code"),
                "committee_name": actor.get("committee_name"),
                "institution_name": actor.get("institution_name"),
                "configuration": actor.get("configuration"),
                "meeting_number": actor.get("meeting_number"),
                "meeting_date": actor.get("meeting_date"),
                "commissioner_name": actor.get("commissioner_name"),
                "is_active": True,
            }
        )

    return converted


# ---------------------------------------------------------------------------
# Dagster asset
# ---------------------------------------------------------------------------


@asset(
    group_name="eu_bronze",
    compute_kind="scraper",
    partitions_def=weekly_partitions,
    description=(
        "Discover and scrape legislative procedures from OEIL and the EP Open Data v2 API. "
        "Recent partitions (<=28 days) use the v2 /procedures/feed for real-time discovery; "
        "older partitions fall back to the OEIL XML catalog filtered by lastpubdate. For each "
        "procedure: scrapes OEIL HTML for metadata (title, legal basis, policy area, timeline, "
        "rapporteurs, committees), then enriches with v2 API data (actors, events, documents). "
        "Outputs raw procedure dicts partitioned by week."
    ),
)
def eu_legislation_bronze(context: AssetExecutionContext) -> List[Dict[str, Any]]:
    """Bronze layer: v2 feed / OEIL XML discovery -> OEIL HTML scraping -> v2 enrichment.

    Discovery:
    - Recent partitions (<=28 days): Uses EP Open Data v2 /procedures/feed.
      Captures ANY procedural activity with millisecond-precision timestamps.
    - Older partitions (>28 days / backfill): Uses OEIL XML catalog filtered
      by lastpubdate (legacy path, unchanged).

    Force a specific mode via env var LEGISLATION_DISCOVERY_MODE=v2_feed|oeil_catalog.

    Content enrichment (always):
    - OEIL HTML scraping: subjects, legal basis, summary text, EUR-Lex refs
    - v2 API: structured events, participations, stage, foreseen activities
    """
    from .oeil_scraper import scrape_multiple_procedures
    from .v2_api import enrich_bronze_with_v2_events

    partition_key = context.partition_key
    start_date, end_date = get_week_range_from_partition(partition_key)
    partition_year = int(partition_key.split("-")[0])

    context.log.info(
        f"Fetching legislation for week {partition_key} ({start_date} to {end_date}), "
        f"year={partition_year}"
    )

    # --- STEP 1: Discover which procedures were updated this week ---
    env_override = os.environ.get("LEGISLATION_DISCOVERY_MODE")
    mode = select_discovery_mode(start_date, env_override=env_override)

    context.log.info(f"Discovery mode: {mode} (env_override={env_override!r})")

    if mode == "v2_feed":
        discovered = discover_via_v2_feed(start_date, end_date, logger=context.log)
    else:
        discovered = discover_via_oeil_catalog(
            partition_year, start_date, end_date, logger=context.log
        )

    context.log.info(f"Discovered {len(discovered)} procedures to scrape via {mode}")

    # --- STEP 2: Extract OEIL references for HTML scraping ---
    filtered_proc_refs = [d["oeil_ref"] for d in discovered if d.get("oeil_ref")]

    context.log.info(f"Scraping {len(filtered_proc_refs)} procedures from OEIL HTML")

    all_procedures = scrape_multiple_procedures(filtered_proc_refs, logger=context.log)

    context.log.info(f"Scraped {len(all_procedures)} procedures from OEIL HTML")

    # --- STEP 3: Convert OEIL scrape results to Procedure model format ---
    raw_procedures = build_procedures_from_oeil(all_procedures, logger=context.log)

    # Validate with Pydantic
    validated = [Procedure(**item).model_dump() for item in raw_procedures]

    # --- STEP 4: Enrich with v2 API events, participations, foreseen activities ---
    context.log.info(f"Enriching {len(validated)} procedures with v2 API data")
    enriched = []
    for proc in validated:
        try:
            enriched_proc = enrich_bronze_with_v2_events(proc, language="en", logger=context.log)
            enriched.append(enriched_proc)
        except Exception as exc:
            context.log.error(f"Failed to enrich procedure {proc.get('process_id')}: {exc}")
            proc["_v2_events"] = []
            proc["_v2_stats"] = {"error": str(exc)}
            enriched.append(proc)

    v2_enriched_count = sum(1 for p in enriched if p.get("_v2_events") and len(p["_v2_events"]) > 0)

    context.add_output_metadata(
        {
            "discovery_mode": mode,
            "year": partition_year,
            "discovered_count": len(discovered),
            "details_scraped": len(all_procedures),
            "validated_count": len(validated),
            "v2_enriched_count": v2_enriched_count,
            "week": partition_key,
        }
    )

    return enriched
