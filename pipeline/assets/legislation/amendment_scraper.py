"""Core logic for scraping EU legislative amendments from OEIL.

Downloads DOCX files from doceo.europarl.europa.eu, converts to HTML via
mammoth, then parses the structured two-column tables using the existing
amendment_parser module.

EP amendment documents (-AM-) don't have HTML versions on doceo — only PDF
and DOCX. The DOCX format preserves XML-like tags (<NumAm>, <Members>,
<DocAmend>, <Article>) and clean table structures that mammoth converts
to standard HTML, which amendment_parser.parse_amendment_document() already
handles.

Functions here are drop-in replacements for the old PDF-based counterparts.
They accept a ``requests.Session`` and a raw Supabase client as parameters so
they work identically in both standalone and Dagster-asset contexts.
"""

from __future__ import annotations

import io
import re
import time
import uuid
from typing import Any

import mammoth
from bs4 import BeautifulSoup

from .amendment_parser import parse_amendment_document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OEIL_BASE = "https://oeil.europarl.europa.eu/oeil/en/procedure-file"
DOCEO_BASE = "https://www.europarl.europa.eu/doceo/document"
RATE_LIMIT_SLEEP = 0.75
DOWNLOAD_TIMEOUT = 60
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def generate_amendment_id(procedure_id: str, document_id: str, amendment_number: int) -> str:
    """Generate a deterministic UUID v5 for a (procedure, document, amendment) triple."""
    name_string = "::".join([procedure_id, document_id, str(amendment_number)])
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name_string))


def extract_committee(document_id: str) -> str:
    """Extract committee code from a document ID like 'ECON-AM-781235' -> 'ECON'."""
    parts = document_id.split("-")
    return parts[0] if parts else ""


def _classify_target_type(location: str) -> str:
    """Return the primary structural type of a target-element string."""
    if not location:
        return ""
    primary = re.split(r"[\s\u2013\u2014\-\u2013\u2014()./]+", location.strip(), maxsplit=1)[0].lower()
    for keyword in ("recital", "article", "annex", "citation", "paragraph", "title", "rule"):
        if primary == keyword:
            return keyword
    # Fallback: scan full string
    low = location.lower()
    for keyword in ("recital", "article", "annex", "citation", "rule", "paragraph", "title"):
        if keyword in low:
            return keyword
    return "other"


# ---------------------------------------------------------------------------
# DOCX download + parse
# ---------------------------------------------------------------------------


def fetch_docx(
    doc_id: str,
    session: Any,
    logger: Any = None,
) -> bytes | None:
    """Download the EN DOCX for a document ID.

    Returns raw bytes on success, or None on failure.
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    url = f"{DOCEO_BASE}/{doc_id}_EN.docx"
    _log(f"  Downloading {url}")

    try:
        resp = session.get(url, timeout=DOWNLOAD_TIMEOUT)
    except Exception as exc:
        _err(f"  HTTP error fetching {doc_id}: {exc}")
        time.sleep(RATE_LIMIT_SLEEP)
        return None

    time.sleep(RATE_LIMIT_SLEEP)

    if resp.status_code == 404:
        _err(f"  404 - no DOCX available for {doc_id}")
        return None

    try:
        resp.raise_for_status()
    except Exception as exc:
        _err(f"  HTTP {resp.status_code} for {doc_id}: {exc}")
        return None

    # Verify it's actually a DOCX (not an HTML error page)
    ct = resp.headers.get("content-type", "")
    if "html" in ct.lower() and len(resp.content) < 50000:
        _err(f"  {doc_id}: got HTML instead of DOCX (likely error page)")
        return None

    _log(f"  OK ({len(resp.content) // 1024} KB)")
    return resp.content


def parse_docx_to_amendments(
    docx_bytes: bytes,
    document_id: str,
    procedure_id: str,
    event_date: str | None = None,
    logger: Any = None,
) -> list[dict[str, Any]]:
    """Convert DOCX to HTML via mammoth, parse amendments, return Supabase rows.

    Uses the existing amendment_parser.parse_amendment_document() which handles
    mammoth's XML-tag-preserving HTML output.
    """
    _log = logger.info if logger else print

    # Convert DOCX to HTML
    result = mammoth.convert_to_html(io.BytesIO(docx_bytes))
    html = result.value

    if not html or len(html) < 100:
        _log(f"  {document_id}: mammoth produced empty HTML")
        return []

    # Parse using existing amendment_parser
    parsed = parse_amendment_document(html)
    raw_amendments = parsed.get("all_amendments", [])

    if not raw_amendments:
        return []

    committee = extract_committee(document_id)

    # Convert to Supabase row format
    rows: list[dict[str, Any]] = []
    for amend in raw_amendments:
        num = amend.get("amendment_number", 0)
        if not num:
            continue

        target = amend.get("target_article") or ""
        original = amend.get("original") or ""
        amended = amend.get("amended") or ""
        justification = amend.get("justification") or ""
        authors = amend.get("submitted_by") or []

        row = {
            "id": generate_amendment_id(procedure_id, document_id, num),
            "procedure_id": procedure_id,
            "document_id": document_id,
            "amendment_number": num,
            "target_element": target,
            "target_type": _classify_target_type(target),
            "original_text": original.strip() or None,
            "amended_text": amended.strip() or None,
            "justification": justification.strip() or None,
            "committee": committee,
            "submitted_by": authors,
            "event_date": event_date,
            "work_type": "AMENDMENT_LIST",
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# OEIL discovery
# ---------------------------------------------------------------------------


def scrape_amendment_doc_ids(
    procedure_id: str,
    session: Any,
    logger: Any = None,
) -> list[dict[str, str]]:
    """Fetch the OEIL procedure page and return amendment document metadata.

    Returns list of dicts with keys ``doc_id`` and ``date_str``.
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    url = f"{OEIL_BASE}?reference={procedure_id}"
    _log(f"Fetching OEIL page: {url}")

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        _err(f"Could not fetch OEIL page: {exc}")
        return []

    time.sleep(RATE_LIMIT_SLEEP)

    soup = BeautifulSoup(resp.text, "html.parser")
    found: dict[str, str] = {}

    for tag in soup.find_all("a", href=True):
        href: str = tag["href"]
        am_match = re.search(r"([\w]+-AM-\d+)_EN\.html", href, re.IGNORECASE)
        if not am_match:
            am_match = re.search(r"([\w]+-AM-\d+)", href, re.IGNORECASE)
        if not am_match:
            continue

        doc_id = am_match.group(1).upper()
        if doc_id in found:
            continue

        date_str = ""
        parent = tag.parent
        if parent:
            text_context = parent.get_text(" ", strip=True)
            date_match = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b", text_context)
            if date_match:
                raw_date = date_match.group(1)
                if "/" in raw_date:
                    parts = raw_date.split("/")
                    try:
                        date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
                    except IndexError:
                        date_str = ""
                else:
                    date_str = raw_date

        found[doc_id] = date_str
        _log(f"  Found amendment document: {doc_id} (date={date_str or 'unknown'})")

    return [{"doc_id": k, "date_str": v} for k, v in found.items()]


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------


def upsert_amendments(
    rows: list[dict[str, Any]],
    client: Any,
    batch_size: int = BATCH_SIZE,
    logger: Any = None,
) -> int:
    """Upsert rows into ``procedure_amendments`` in batches."""
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    if not rows:
        return 0

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        try:
            client.table("procedure_amendments").upsert(batch, on_conflict="id").execute()
            total += len(batch)
            _log(f"  Upserted batch {i // batch_size + 1}: {len(batch)} rows (total: {total})")
        except Exception as exc:
            _err(f"  Batch upsert failed at offset {i}: {exc}")

    return total


# ---------------------------------------------------------------------------
# Batch helpers (used by the Dagster asset)
# ---------------------------------------------------------------------------


def fetch_all_cod_procedures(client: Any) -> list[str]:
    """Return all COD procedure IDs from Supabase."""
    rows: list[dict] = []
    offset = 0
    batch = 1000
    while True:
        resp = (
            client.table("procedures")
            .select("id")
            .like("id", "%COD%")
            .range(offset, offset + batch - 1)
            .execute()
        )
        rows.extend(resp.data)
        if len(resp.data) < batch:
            break
        offset += batch
    return [r["id"] for r in rows]


def fetch_procedures_with_amendments(client: Any) -> set[str]:
    """Return procedure IDs that already have amendments in Supabase."""
    rows: list[dict] = []
    offset = 0
    batch = 1000
    while True:
        resp = (
            client.table("procedure_amendments")
            .select("procedure_id")
            .range(offset, offset + batch - 1)
            .execute()
        )
        rows.extend(resp.data)
        if len(resp.data) < batch:
            break
        offset += batch
    return {r["procedure_id"] for r in rows}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def scrape_procedure_amendments(
    procedure_id: str,
    client: Any,
    session: Any,
    logger: Any = None,
) -> int:
    """Discover, download, parse, and upload amendments for one procedure.

    Downloads the DOCX for each amendment document found on the OEIL
    procedure page, converts via mammoth, parses the tables, and upserts.

    Returns the number of amendment rows upserted (0 if nothing found).
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    _log(f"Starting amendment scrape for {procedure_id}")

    # Step 1: discover amendment document IDs from OEIL
    doc_entries = scrape_amendment_doc_ids(procedure_id, session, logger=logger)
    if not doc_entries:
        _err(f"No amendment documents found on OEIL for {procedure_id}")
        return 0

    _log(f"Found {len(doc_entries)} amendment document(s) for {procedure_id}")

    # Step 2: fetch DOCX for each document and parse
    all_rows: list[dict[str, Any]] = []

    for entry in doc_entries:
        doc_id = entry["doc_id"]
        date_str = entry["date_str"]

        docx_bytes = fetch_docx(doc_id, session, logger=logger)
        if docx_bytes is None:
            _err(f"  Skipping {doc_id}: download failed")
            continue

        rows = parse_docx_to_amendments(
            docx_bytes=docx_bytes,
            document_id=doc_id,
            procedure_id=procedure_id,
            event_date=date_str or None,
            logger=logger,
        )
        _log(f"  {doc_id}: {len(rows)} amendments parsed")
        all_rows.extend(rows)

    _log(f"Total amendments parsed for {procedure_id}: {len(all_rows)}")

    if not all_rows:
        return 0

    # Step 3: upload to Supabase
    uploaded = upsert_amendments(all_rows, client, logger=logger)
    _log(f"Uploaded {uploaded} amendments for {procedure_id}")
    return uploaded
