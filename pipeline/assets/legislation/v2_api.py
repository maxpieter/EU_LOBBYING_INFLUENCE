"""European Parliament v2 API integration for document enrichment.

Fetches detailed document metadata and file URLs from the EP Open Data API v2.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests

BASE = "https://data.europarl.europa.eu/"
API_BASE = "https://data.europarl.europa.eu/api/v2/"

DOC_REF_KEYS = (
    "based_on_a_realization_of",
    "recorded_in_a_realization_of",
    "decided_on_a_realization_of",
)

DEFAULT_TIMEOUT = 30


class EPV2ApiClient:
    """Client for European Parliament Open Data API v2."""

    def __init__(
        self,
        user_agent: str = "Parl8/1.0",
        max_retries: int = 4,
        backoff_seconds: float = 1.0,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout = timeout

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET JSON with retry/backoff for 429 / transient 5xx."""
        if params is None:
            params = {}

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 204:
                    return {"data": []}

                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = self.backoff_seconds * (2**attempt)
                    ra = resp.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        wait = max(wait, float(ra))
                    time.sleep(wait)
                    continue

                # For other errors, return empty and log
                return {"data": [], "_error": f"HTTP {resp.status_code}"}

            except Exception as e:
                if attempt < self.max_retries:
                    wait = self.backoff_seconds * (2**attempt)
                    time.sleep(wait)
                    continue
                return {"data": [], "_error": str(e)}

        return {"data": [], "_error": "Failed after retries"}

    def get_json_allow_404(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Like get_json, but returns (status_code, payload) and does NOT raise for 404/406."""
        if params is None:
            params = {}

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)

                if resp.status_code == 200:
                    return 200, resp.json()

                if resp.status_code == 204:
                    return 204, {"data": []}

                if resp.status_code in (404, 406):
                    try:
                        return resp.status_code, resp.json()
                    except Exception:
                        return resp.status_code, {"_error": resp.text}

                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = self.backoff_seconds * (2**attempt)
                    ra = resp.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        wait = max(wait, float(ra))
                    time.sleep(wait)
                    continue

                return resp.status_code, {"_error": f"HTTP {resp.status_code}"}

            except Exception as e:
                if attempt < self.max_retries:
                    wait = self.backoff_seconds * (2**attempt)
                    time.sleep(wait)
                    continue
                return 0, {"_error": str(e)}

        return 0, {"_error": "Failed after retries"}


def _strip_doc_id(doc_ref: str) -> str:
    """Extract document ID from various formats."""
    if not doc_ref:
        return doc_ref
    s = doc_ref.strip()

    # Remove base if full URL
    if s.startswith(BASE):
        s = s[len(BASE) :].lstrip("/")

    # Extract after eli/dl/doc/
    import re

    m = re.search(r"(?:^|/)eli/dl/doc/([^/?#]+)", s)
    if m:
        return m.group(1)

    return s


def _absolute_distribution_url(path_or_url: str) -> str:
    """Convert relative distribution path to absolute URL."""
    if not path_or_url:
        return path_or_url
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return urljoin(BASE, path_or_url.lstrip("/"))


def _doc_endpoint_v2(doc_id: str) -> str:
    """Best-effort mapping for v2 convenience endpoints."""
    if doc_id.startswith("TA-"):
        return f"adopted-texts/{doc_id}"
    if doc_id.startswith("PV-") or doc_id.startswith("CRE-"):
        return f"plenary-session-documents/{doc_id}"
    return f"documents/{doc_id}"


def _eli_doc_url(doc_ref_or_id: str) -> str:
    """Canonical dereference URL for documents."""
    doc_id = _strip_doc_id(doc_ref_or_id)
    return urljoin(BASE, f"eli/dl/doc/{doc_id}")


def _extract_files_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract file metadata from EP JSON-LD structure."""
    files: List[Dict[str, Any]] = []
    data = payload.get("data") or []
    if not data:
        return files

    work = data[0]
    realized = work.get("is_realized_by") or []
    if not isinstance(realized, list):
        realized = []

    for expr in realized:
        if not isinstance(expr, dict):
            continue
        embodied = expr.get("is_embodied_by") or []
        if not isinstance(embodied, list):
            embodied = []
        for man in embodied:
            if not isinstance(man, dict):
                continue
            url = man.get("is_exemplified_by")
            if not url:
                continue
            files.append(
                {
                    "manifestation_id": str(man.get("id", "")),
                    "media_type": man.get("media_type"),
                    "file_type": man.get("format"),
                    "issued": man.get("issued"),
                    "byte_size": man.get("byteSize"),
                    "url": _absolute_distribution_url(str(url)),
                }
            )

    # De-duplicate by URL
    uniq: Dict[str, Dict[str, Any]] = {}
    for f in files:
        uniq[f["url"]] = f
    return list(uniq.values())


def _pick_title(doc_obj: Dict[str, Any], language: str) -> Optional[str]:
    """Extract title from various EP JSON-LD fields."""
    for key in ("title_dcterms", "title", "title_alternative"):
        v = doc_obj.get(key)
        if isinstance(v, dict):
            if language in v:
                return v[language]
            if v:
                return next(iter(v.values()))
        if isinstance(v, str):
            return v
    return None


def resolve_document(
    client: EPV2ApiClient,
    doc_ref: str,
    language: str = "en",
    fmt: str = "application/ld+json",
) -> Optional[Dict[str, Any]]:
    """Resolve a document reference to metadata and file URLs."""
    doc_id = _strip_doc_id(doc_ref)
    if not doc_id:
        return None

    # Try v2 endpoint
    v2_endpoint = _doc_endpoint_v2(doc_id)
    v2_url = urljoin(API_BASE, v2_endpoint)
    status, payload = client.get_json_allow_404(
        v2_url, params={"format": fmt, "language": language}
    )

    source = "v2"
    fetched_from = v2_url

    # Fallback to dereference
    if status in (404, 406):
        eli_url = _eli_doc_url(doc_id)
        status2, payload2 = client.get_json_allow_404(
            eli_url, params={"format": fmt, "language": language}
        )
        if status2 not in (200, 204):
            return {
                "doc_id": doc_id,
                "source": "none",
                "fetched_from": f"v2:{v2_url} -> {status}; eli:{eli_url} -> {status2}",
                "work_type": None,
                "title": None,
                "language": language,
                "files": [],
                "raw_id": doc_ref,
            }
        source = "eli"
        fetched_from = eli_url
        payload = payload2

    data = payload.get("data") or []
    if not data:
        return {
            "doc_id": doc_id,
            "source": source,
            "fetched_from": fetched_from,
            "work_type": None,
            "title": None,
            "language": language,
            "files": [],
            "raw_id": doc_ref,
        }

    work = data[0] if isinstance(data[0], dict) else {}
    return {
        "doc_id": doc_id,
        "source": source,
        "fetched_from": fetched_from,
        "work_type": work.get("work_type"),
        "title": _pick_title(work, language=language),
        "language": language,
        "files": _extract_files_from_payload(payload),
        "raw_id": doc_ref,
    }


def _extract_foreseen_activities(proc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract foreseen (scheduled future) activities from a v2 procedure dict.

    The v2 API returns ``was_scheduled_in`` as a list of ForeseenActivity objects
    embedded in the procedure payload.  Each entry tells us when and where the
    procedure is *expected* to come before the plenary, including the agenda
    point and associated agenda documents.

    Args:
        proc: Raw procedure dict from the v2 API ``data[0]``.

    Returns:
        List of simplified foreseen activity dicts, each with:
            activity_id        : str
            activity_date      : str  (YYYY-MM-DD)
            had_activity_type  : str  (e.g. "def/ep-activities/PLENARY_VOTE")
            occured_at_stage   : str  (URI)
            notation_agendaPoint: str | None
            documented_by      : list[str]  (document reference URIs)
    """
    raw = proc.get("was_scheduled_in") or []
    if not isinstance(raw, list):
        return []

    results: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        doc_refs = item.get("documented_by_a_realization_of") or []
        if isinstance(doc_refs, str):
            doc_refs = [doc_refs]
        results.append(
            {
                "activity_id": item.get("activity_id") or item.get("id", ""),
                "activity_date": item.get("activity_date"),
                "had_activity_type": item.get("had_activity_type"),
                "occured_at_stage": item.get("occured_at_stage"),
                "notation_agendaPoint": item.get("notation_agendaPoint"),
                "documented_by": [str(r) for r in doc_refs if r],
            }
        )

    # Sort by activity_date ascending so the nearest event is first
    results.sort(key=lambda x: x.get("activity_date") or "")
    return results


def fetch_procedure_v2_events(
    process_id: str,
    language: str = "en",
    user_agent: str = "Parl8/1.0",
    logger: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str], List[Dict[str, Any]]]:
    """Fetch v2 API events, participations, foreseen activities for a procedure.

    Args:
        process_id: Procedure ID in format "2025-0021" or "2025/0021(COD)"

    Returns:
        Tuple of:
            events_list            : List[Dict]
            participations_list    : List[Dict]
            procedure_status_label : str | None
            foreseen_activities    : List[Dict]  -- future scheduled plenary events
    """
    client = EPV2ApiClient(user_agent=user_agent)

    # Normalize process_id format for v2 API
    # v2 API expects "YYYY-NNNN" format (e.g., "2025-0021")
    # We might receive "2025-0021" (good), "2025/0021" (OEIL), or "2025/0021(COD)" (OEIL with type)
    normalized_id = process_id

    # Remove procedure type suffix if present: "2025/0021(COD)" -> "2025/0021"
    if "(" in normalized_id:
        normalized_id = normalized_id.split("(")[0].strip()

    # Convert slash to dash: "2025/0021" -> "2025-0021"
    if "/" in normalized_id:
        normalized_id = normalized_id.replace("/", "-")

    # Fetch process data to extract participations and status
    process_url = urljoin(API_BASE, f"procedures/{normalized_id}")

    if logger:
        logger.debug(f"Fetching v2 procedure from {process_url} (original ID: {process_id})")

    process_payload = client.get_json(process_url, params={"format": "application/ld+json"})

    # Extract participations, foreseen activities, and procedure status from procedure data
    participations: List[Dict[str, Any]] = []
    foreseen_activities: List[Dict[str, Any]] = []
    procedure_status = None
    process_data = process_payload.get("data") or []
    if process_data and isinstance(process_data, list):
        proc = process_data[0]
        if isinstance(proc, dict):
            # Extract procedure-level status label
            procedure_status = proc.get("procedure_status_label")

            raw_participations = proc.get("had_participation") or []
            if isinstance(raw_participations, list):
                for part in raw_participations:
                    if isinstance(part, dict):
                        participations.append(
                            {
                                "id": part.get("id"),
                                "type": part.get("type"),
                                "activity_date": part.get("activity_date"),
                                "had_participant_person": part.get("had_participant_person"),
                                "had_participant_organization": part.get(
                                    "had_participant_organization"
                                ),
                                "occured_at_stage": part.get("occured_at_stage"),
                                "parliamentary_term": part.get("parliamentary_term"),
                                "participation_role": part.get("participation_role"),
                                "politicalGroup": part.get("politicalGroup"),
                                "participation_in_name_of": part.get("participation_in_name_of"),
                            }
                        )

            # Extract foreseen (upcoming) activities
            foreseen_activities = _extract_foreseen_activities(proc)

    if logger:
        logger.debug(
            f"Extracted {len(participations)} participations, {len(foreseen_activities)} foreseen activities from procedure data"
        )

    events_url = urljoin(API_BASE, f"procedures/{normalized_id}/events")

    if logger:
        logger.debug(f"Fetching v2 events from {events_url}")

    events_payload = client.get_json(events_url, params={"format": "application/ld+json"})

    events = events_payload.get("data") or []
    results: List[Dict[str, Any]] = []

    # Global de-dupe cache
    resolved_cache: Dict[str, Dict[str, Any]] = {}

    for ev in events:
        ev_id = ev.get("activity_id") or ev.get("id") or ""
        act_date = ev.get("activity_date")
        act_type = ev.get("had_activity_type")
        stage = ev.get("occured_at_stage")

        # Extract document references
        refs: List[str] = []
        for k in DOC_REF_KEYS:
            v = ev.get(k)
            if isinstance(v, list):
                refs.extend([x for x in v if isinstance(x, str)])
            elif isinstance(v, str):
                refs.append(v)

        # Normalize & de-dup
        seen: Set[str] = set()
        norm_refs: List[str] = []
        for r in refs:
            rid = _strip_doc_id(r) or r
            if rid not in seen:
                seen.add(rid)
                norm_refs.append(r)

        # Resolve documents
        resolved_docs: List[Dict[str, Any]] = []
        for r in norm_refs:
            doc_id = _strip_doc_id(r)
            if not doc_id:
                continue

            if doc_id in resolved_cache:
                resolved_docs.append(resolved_cache[doc_id])
                continue

            rd = resolve_document(client, r, language=language)
            if rd:
                resolved_cache[doc_id] = rd
                resolved_docs.append(rd)

        results.append(
            {
                "event_id": ev_id,
                "activity_date": act_date,
                "had_activity_type": act_type,
                "stage": stage,
                "doc_refs": norm_refs,
                "resolved_docs": resolved_docs,
            }
        )

    if logger:
        logger.info(
            f"Fetched {len(events)} v2 events with {len(resolved_cache)} unique documents, "
            f"{len(participations)} participations, and {len(foreseen_activities)} foreseen activities "
            f"for {process_id}"
        )

    return results, participations, procedure_status, foreseen_activities


def enrich_bronze_with_v2_events(
    procedure: Dict[str, Any],
    language: str = "en",
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Enrich Bronze procedure data with v2 API event and participation data.

    Adds _v2_events, _v2_participations, and _v2_status fields with detailed metadata.
    Also populates the stage field with v2 procedure status.
    """
    # Try procedure_id first (format: "2025-0021"), fallback to id (format: "2025/0021(COD)")
    procedure_id = procedure.get("procedure_id") or procedure.get("id")
    if not procedure_id:
        if logger:
            logger.warning("Procedure has no ID, skipping v2 enrichment")
        return procedure

    try:
        v2_events, v2_participations, v2_status, v2_foreseen = fetch_procedure_v2_events(
            process_id=procedure_id, language=language, logger=logger
        )
        procedure["_v2_events"] = v2_events
        procedure["_v2_participations"] = v2_participations
        procedure["_v2_status"] = v2_status
        procedure["foreseen_activities"] = v2_foreseen

        # Use v2 status for the stage field (OEIL's current_stage is not useful)
        if v2_status:
            procedure["stage"] = v2_status

        # Add summary statistics
        total_docs = sum(len(e.get("resolved_docs", [])) for e in v2_events)
        procedure["_v2_stats"] = {
            "events_count": len(v2_events),
            "unique_docs_count": total_docs,
            "participations_count": len(v2_participations),
            "foreseen_activities_count": len(v2_foreseen),
        }

    except Exception as e:
        if logger:
            logger.error(f"Failed to fetch v2 data for {procedure_id}: {e}")
        procedure["_v2_events"] = []
        procedure["_v2_participations"] = []
        procedure["_v2_status"] = None
        procedure["foreseen_activities"] = []
        procedure["_v2_stats"] = {"error": str(e)}

    return procedure
