"""EP Open Data v2 procedures feed parser and discovery utilities.

Provides feed-based discovery of updated procedures using the EP Open Data API v2.
The feed at /procedures/feed returns Atom XML with millisecond-granular timestamps,
capturing ANY procedural activity (amendments, trilogues, votes, etc.) -- not just
the OEIL publication events.

Key difference vs OEIL XML catalog:
- OEIL `lastpubdate`: day-level granularity, only captures proposal publication/legislation
- v2 feed `<updated>`: millisecond ISO 8601, captures every activity type
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import requests

V2_API_BASE = "https://data.europarl.europa.eu/api/v2"
FEED_URL = f"{V2_API_BASE}/procedures/feed"
PROCEDURES_URL = f"{V2_API_BASE}/procedures"

# Atom namespace
ATOM_NS = "http://www.w3.org/2005/Atom"

# Procedure types tracked by the legislation pipeline
TRACKED_PROCEDURE_TYPES = ["COD", "CNS", "APP"]

# Limits on paginated backfill to avoid runaway loops
MAX_PAGINATE_OFFSET = 10_000
PAGE_SIZE = 50

# Rate-limit sleep between paginated requests (seconds)
PAGINATE_SLEEP_SECONDS = 0.2


def parse_feed_entries(
    xml_content: bytes | str,
) -> List[Dict[str, Any]]:
    """Parse Atom XML feed content into a list of procedure entry dicts.

    Each entry dict contains:
        process_id  : str  -- e.g. "2023-0404"
        title       : str  -- procedure title (English)
        process_type: str  -- e.g. "COD", "INI"
        updated_at  : datetime (UTC) -- millisecond-precision update timestamp
        api_url     : str  -- canonical REST URL for this procedure
        eli_url     : str  -- ELI/URI self link

    Args:
        xml_content: Raw bytes or str from the Atom feed response.

    Returns:
        List of entry dicts; empty list if content is empty or unparseable.
    """
    if not xml_content:
        return []

    if isinstance(xml_content, str):
        xml_content = xml_content.encode("utf-8")

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return []

    ns = {"atom": ATOM_NS}
    entries = root.findall("atom:entry", ns)
    results: List[Dict[str, Any]] = []

    for entry in entries:
        # <id>https://data.europarl.europa.eu/eli/dl/proc/2023-0404</id>
        id_elem = entry.find("atom:id", ns)
        if id_elem is None or not id_elem.text:
            continue
        eli_url = id_elem.text.strip()

        # Extract process_id from ELI URI: "...eli/dl/proc/2023-0404" -> "2023-0404"
        process_id = _extract_process_id_from_uri(eli_url)
        if not process_id:
            continue

        # <title type="text" xml:lang="en">...</title>
        title_elem = entry.find("atom:title", ns)
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

        # <category term="process-type" scheme="...ep-procedure-types/COD" label="..."/>
        process_type = ""
        category_elem = entry.find("atom:category", ns)
        if category_elem is not None:
            scheme = category_elem.get("scheme", "")
            # scheme ends with "/COD", "/INI", etc.
            if "/" in scheme:
                process_type = scheme.rsplit("/", 1)[-1]

        # <updated>2026-03-05T11:22:01.291Z</updated>
        updated_at: Optional[datetime] = None
        updated_elem = entry.find("atom:updated", ns)
        if updated_elem is not None and updated_elem.text:
            updated_at = _parse_iso_datetime(updated_elem.text.strip())

        # <link rel="alternate" href="...api/v2/procedures/2023-0404"/>
        api_url = ""
        for link_elem in entry.findall("atom:link", ns):
            if link_elem.get("rel") == "alternate":
                api_url = link_elem.get("href", "")
                break

        results.append(
            {
                "process_id": process_id,
                "title": title,
                "process_type": process_type,
                "updated_at": updated_at,
                "api_url": api_url,
                "eli_url": eli_url,
            }
        )

    return results


def fetch_procedures_feed(
    process_types: Optional[List[str]] = None,
    timeframe: str = "one-week",
    start_date: Optional[str] = None,
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Fetch and parse the EP v2 procedures feed.

    The feed covers at most the past ~30 days regardless of `start_date`. This
    makes it suitable for incremental weekly runs but NOT for backfilling
    historical partitions (use ``fetch_all_procedure_ids`` for that).

    Args:
        process_types: List of procedure type codes to filter (e.g. ["COD", "APP"]).
            If None, fetches all types.
        timeframe: "one-week" | "one-month" | "one-day" | "today" | "custom".
            Use "custom" together with ``start_date`` for exact range.
        start_date: ISO date string "YYYY-MM-DD", required when timeframe="custom".
        logger: Optional Dagster logger.

    Returns:
        List of entry dicts from ``parse_feed_entries``, possibly filtered by
        ``process_types``.
    """
    _log = _make_logger(logger)

    params: Dict[str, Any] = {"timeframe": timeframe}
    if start_date:
        params["start-date"] = start_date

    if process_types:
        # The API accepts a single process-type per request; fetch each and merge.
        all_entries: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for pt in process_types:
            pt_params = dict(params)
            pt_params["process-type"] = pt
            entries = _fetch_feed_once(pt_params, logger=logger)
            _log(f"v2 feed [{pt}] {timeframe}: {len(entries)} entries", "info")
            for e in entries:
                pid = e["process_id"]
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_entries.append(e)
        return all_entries

    # No filter -- fetch all procedure types in one request
    entries = _fetch_feed_once(params, logger=logger)
    _log(f"v2 feed (all types) {timeframe}: {len(entries)} entries", "info")
    return entries


def fetch_procedures_feed_for_window(
    start_dt: datetime,
    end_dt: datetime,
    process_types: Optional[List[str]] = None,
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Fetch feed entries whose ``updated_at`` falls within [start_dt, end_dt].

    Because the v2 feed is hard-capped at ~30 days, this function only works
    reliably for windows within the past 30 days.

    Args:
        start_dt: Window start (timezone-aware or naive UTC).
        end_dt:   Window end (timezone-aware or naive UTC).
        process_types: Optional list of procedure type codes to filter.
        logger: Optional Dagster logger.

    Returns:
        Filtered list of entry dicts.
    """
    # Always fetch one-month feed and filter in-memory; this ensures we get
    # the broadest possible data even for shorter windows within the 30-day cap.
    entries = fetch_procedures_feed(
        process_types=process_types,
        timeframe="one-month",
        logger=logger,
    )

    # Normalise start/end to aware UTC for comparison
    start_utc = _to_utc(start_dt)
    end_utc = _to_utc(end_dt)

    filtered = [
        e
        for e in entries
        if e["updated_at"] is not None and start_utc <= e["updated_at"] <= end_utc
    ]

    _make_logger(logger)(
        f"v2 feed window filter [{start_utc.date()} -> {end_utc.date()}]: "
        f"{len(filtered)}/{len(entries)} entries match",
        "info",
    )
    return filtered


def fetch_all_procedure_ids(
    process_types: Optional[List[str]] = None,
    logger: Optional[Any] = None,
    max_per_type: int = MAX_PAGINATE_OFFSET,
) -> List[Dict[str, Any]]:
    """Paginate through /procedures endpoint to discover ALL historical procedures.

    This is the backfill path. The paginated endpoint has no date-range filter,
    but returns procedures sorted by process_id chronologically. For COD alone
    there are ~2,900 procedures; all three types together are ~4,500-5,000.

    Each result dict contains: process_id, process_type, label.

    Args:
        process_types: List of procedure type codes (defaults to TRACKED_PROCEDURE_TYPES).
        logger: Optional Dagster logger.
        max_per_type: Safety cap on total results per type.

    Returns:
        List of minimal procedure dicts with process_id, process_type, label.
    """
    if process_types is None:
        process_types = TRACKED_PROCEDURE_TYPES

    _log = _make_logger(logger)
    all_results: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for pt in process_types:
        offset = 0
        type_count = 0
        _log(f"Paginating /procedures?process-type={pt}", "info")

        while offset < max_per_type:
            params = {
                "process-type": pt,
                "offset": offset,
                "limit": PAGE_SIZE,
                "format": "application/ld+json",
            }

            try:
                time.sleep(PAGINATE_SLEEP_SECONDS)
                resp = requests.get(
                    PROCEDURES_URL,
                    params=params,
                    timeout=30,
                    headers={"User-Agent": "Parl8/1.0"},
                )
                if resp.status_code == 204:
                    break
                if resp.status_code != 200:
                    _log(
                        f"Paginate /procedures offset={offset} type={pt}: "
                        f"HTTP {resp.status_code}",
                        "warning",
                    )
                    break

                data = resp.json()
                items = data.get("data") or []
                if not items:
                    break

                for item in items:
                    process_id = item.get("process_id", "")
                    if not process_id or process_id in seen_ids:
                        continue
                    seen_ids.add(process_id)
                    all_results.append(
                        {
                            "process_id": process_id,
                            "process_type": pt,
                            "label": item.get("label", ""),
                        }
                    )

                type_count += len(items)
                offset += PAGE_SIZE

                if len(items) < PAGE_SIZE:
                    # Last page
                    break

            except Exception as exc:
                _log(f"Error paginating /procedures type={pt} offset={offset}: {exc}", "error")
                break

        _log(f"Paginated {pt}: {type_count} total items", "info")

    _log(f"fetch_all_procedure_ids: {len(all_results)} total unique procedures", "info")
    return all_results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_process_id_from_uri(uri: str) -> str:
    """Extract process_id from an ELI URI.

    Examples:
        "https://data.europarl.europa.eu/eli/dl/proc/2023-0404" -> "2023-0404"
        "eli/dl/proc/2023-0404" -> "2023-0404"
    """
    if not uri:
        return ""
    # Strip trailing whitespace/newlines
    uri = uri.strip()
    # Everything after the last "/"
    last = uri.rsplit("/", 1)[-1]
    # Validate format: "YYYY-NNNN" (4 digits, dash, 1-4 digits)
    import re

    if re.match(r"^\d{4}-\d{1,4}[A-Z]?$", last):
        return last
    return ""


def _parse_iso_datetime(text: str) -> Optional[datetime]:
    """Parse ISO 8601 datetime with optional milliseconds and Z suffix."""
    if not text:
        return None
    # Normalise Z -> +00:00
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _to_utc(dt: datetime) -> datetime:
    """Ensure datetime is UTC-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fetch_feed_once(
    params: Dict[str, Any],
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Perform a single GET to the feed URL and parse the result."""
    _log = _make_logger(logger)
    try:
        resp = requests.get(
            FEED_URL,
            params=params,
            timeout=30,
            headers={"User-Agent": "Parl8/1.0"},
        )
        if resp.status_code == 204:
            return []
        if resp.status_code != 200:
            _log(f"v2 feed HTTP {resp.status_code} for params={params}", "warning")
            return []
        return parse_feed_entries(resp.content)
    except Exception as exc:
        _log(f"v2 feed request failed: {exc}", "error")
        return []


def _make_logger(logger: Optional[Any]):
    """Return a callable (msg, level) -> None that logs via Dagster or no-ops."""

    def _log(msg: str, level: str = "info") -> None:
        if logger:
            getattr(logger, level)(msg)

    return _log
