"""OEIL catalog discovery for EU legislation (backfill path).

Provides discovery of procedures for partitions older than the v2 feed's
~30-day window.  Uses the v2 /procedures paginated endpoint to enumerate
all known procedures, then filters by year derived from the process_id.

Limitation: the v2 paginated endpoint does not expose per-procedure
last-modified dates, so week-level filtering is not possible at discovery
time.  All procedures for the partition year are returned; the downstream
OEIL HTML scraper fetches full event timelines that can be used for
finer-grained deduplication in silver/diamond layers.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

# Procedure types tracked for legislation discovery.
# Maps display name -> EP Open Data process type code.
OEIL_PROCEDURE_TYPES: Dict[str, str] = {
    "COD": "COD",  # Ordinary legislative procedure (co-decision)
    "CNS": "CNS",  # Consultation procedure
    "APP": "APP",  # Consent (approval) procedure
}

# In-memory cache keyed by (year, procedure_type) to avoid re-fetching
# the full paginated listing for every partition of the same year.
_catalog_cache: Dict[tuple, List[Dict[str, Any]]] = {}


def fetch_oeil_procedure_catalog(
    year: int,
    procedure_type: str,
    logger: Optional[Any] = None,
    use_cache: bool = True,
    return_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """Fetch procedure catalog for a given year and type.

    Uses the v2 /procedures paginated endpoint, then filters by year prefix
    in the process_id (e.g. "2024-" for year=2024).

    Args:
        year: Year to scope discovery to.
        procedure_type: EP procedure type code ("COD", "CNS", "APP").
        logger: Optional Dagster logger.
        use_cache: Re-use cached results for the same (year, type) tuple.
        return_metadata: Included for call-site compatibility; always True.

    Returns:
        List of metadata dicts with keys:
            process_id  : str   e.g. "2024-0001"
            oeil_ref    : str   e.g. "2024/0001(COD)"
            title       : str
            process_type: str
    """
    from .v2_feed import fetch_all_procedure_ids

    cache_key = (year, procedure_type)

    if use_cache and cache_key in _catalog_cache:
        cached = _catalog_cache[cache_key]
        if logger:
            logger.info(
                f"OEIL catalog cache hit ({year}, {procedure_type}): {len(cached)} procedures"
            )
        return cached

    if logger:
        logger.info(
            f"Fetching all {procedure_type} procedures from v2 paginated endpoint"
        )

    all_procs = fetch_all_procedure_ids(
        process_types=[procedure_type],
        logger=logger,
    )

    year_prefix = f"{year}-"
    results: List[Dict[str, Any]] = []

    for p in all_procs:
        pid = p["process_id"]
        if not pid.startswith(year_prefix):
            continue
        pt = p.get("process_type", procedure_type)
        # "2024-0001" + "COD" -> "2024/0001(COD)"
        oeil_ref = pid.replace("-", "/", 1)
        if pt:
            oeil_ref = f"{oeil_ref}({pt})"
        results.append(
            {
                "process_id": pid,
                "oeil_ref": oeil_ref,
                "title": p.get("label", ""),
                "process_type": pt,
            }
        )

    if use_cache:
        _catalog_cache[cache_key] = results

    if logger:
        logger.info(f"OEIL catalog {year}/{procedure_type}: {len(results)} procedures")

    return results


def filter_procedures_by_date(
    procedures_metadata: List[Dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
    buffer_weeks: int = 0,
    logger: Optional[Any] = None,
) -> List[str]:
    """Return process_ids from the catalog that may have activity in the date range.

    The v2 paginated endpoint does not provide per-procedure activity dates,
    so this returns ALL procedures from the catalog.  Downstream layers
    (silver/diamond) handle deduplication.

    Args:
        procedures_metadata: Output of fetch_oeil_procedure_catalog.
        start_date: Partition week start.
        end_date: Partition week end.
        buffer_weeks: Extra weeks of buffer (unused — no dates to offset).
        logger: Optional Dagster logger.

    Returns:
        List of process_id strings.
    """
    process_ids = [p["process_id"] for p in procedures_metadata]

    if logger:
        logger.info(
            f"filter_procedures_by_date: returning all {len(process_ids)} procedures "
            f"for {start_date.date()} -> {end_date.date()} "
            f"(no per-procedure dates available from v2 paginated endpoint)"
        )

    return process_ids
