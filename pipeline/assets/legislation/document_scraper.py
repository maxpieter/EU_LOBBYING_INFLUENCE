"""Core logic for scraping non-amendment EU legislative documents from OEIL.

Functions here are adapted from scripts/scrape_documents_to_supabase.py.
They accept a ``requests.Session`` and a raw Supabase client as parameters so
they work identically in both standalone and Dagster-asset contexts.

Document types handled:
  - Draft reports      (*-PR-*)
  - Committee opinions (*-AD-*)
  - Committee reports  (A-{leg}-{year}-{num})
  - Texts adopted      (TA-{leg}-{year}-{num})
  - Commission proposals (COM({year}){num})

Amendment documents (*-AM-*) are explicitly excluded.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PDFTOTEXT = "/opt/homebrew/bin/pdftotext"
OEIL_BASE = "https://oeil.europarl.europa.eu/oeil/en/procedure-file"
DOCEO_PDF_BASE = "https://www.europarl.europa.eu/doceo/document"
COM_PDF_TEMPLATE = (
    "https://www.europarl.europa.eu/RegData/docs_autres_institutions"
    "/commission_europeenne/com/{year}/{num_padded}/COM_COM({year}){num_padded}_EN.pdf"
)
EURLEX_PDF_TEMPLATE = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:5{year}PC{num_padded}"
)

RATE_LIMIT_OEIL = 0.75
RATE_LIMIT_PDF = 0.5
DOWNLOAD_TIMEOUT = 60
BATCH_SIZE = 50

# Document type constants (stored in the DB)
DOCTYPE_DRAFT_REPORT = "draft_report"
DOCTYPE_OPINION = "opinion"
DOCTYPE_COMMITTEE_REPORT = "committee_report"
DOCTYPE_TEXT_ADOPTED = "text_adopted"
DOCTYPE_COMMISSION_PROPOSAL = "commission_proposal"

# ---------------------------------------------------------------------------
# UUID helper
# ---------------------------------------------------------------------------


def generate_document_id(procedure_id: str, document_id: str) -> str:
    """Return a deterministic UUID v5 for a (procedure_id, document_id) pair."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{procedure_id}::{document_id}"))


# ---------------------------------------------------------------------------
# Document ID / URL helpers
# ---------------------------------------------------------------------------


def classify_document(doc_id: str) -> str | None:
    """Return the document type string for a document ID, or None to skip.

    Returns None for amendment documents (*-AM-*) and unknown formats.
    """
    upper = doc_id.upper()
    if "-AM-" in upper:
        return None
    if "-PR-" in upper:
        return DOCTYPE_DRAFT_REPORT
    if "-AD-" in upper:
        return DOCTYPE_OPINION
    if re.match(r"^A-\d+-\d{4}-\d+$", upper):
        return DOCTYPE_COMMITTEE_REPORT
    if re.match(r"^TA-\d+-\d{4}-\d+$", upper):
        return DOCTYPE_TEXT_ADOPTED
    if re.match(r"^COM\(\d{4}\)\d+$", upper):
        return DOCTYPE_COMMISSION_PROPOSAL
    return None


def extract_committee(doc_id: str) -> str | None:
    """Extract committee code from a document ID, e.g. 'ECON-PR-778136' -> 'ECON'."""
    match = re.match(r"^([A-Z0-9]+)-(?:PR|AD)-\d+$", doc_id.upper())
    if match:
        return match.group(1)
    return None


def build_pdf_url(doc_id: str) -> str | None:
    """Construct the primary PDF download URL for a given document ID."""
    upper = doc_id.upper()

    if re.match(r"^[\w]+-(?:PR|AD)-\d+$", upper):
        return f"{DOCEO_PDF_BASE}/{upper}_EN.pdf"

    if re.match(r"^A-\d+-\d{4}-\d+$", upper):
        return f"{DOCEO_PDF_BASE}/{upper}_EN.pdf"

    if re.match(r"^TA-\d+-\d{4}-\d+$", upper):
        return f"{DOCEO_PDF_BASE}/{upper}_EN.pdf"

    com_match = re.match(r"^COM\((\d{4})\)(\d+)$", upper)
    if com_match:
        year = com_match.group(1)
        num = com_match.group(2).zfill(4)
        return COM_PDF_TEMPLATE.format(year=year, num_padded=num)

    return None


def build_eurlex_fallback_url(doc_id: str) -> str | None:
    """Return the EUR-Lex fallback URL for a COM proposal, or None for other types."""
    com_match = re.match(r"^COM\((\d{4})\)(\d+)$", doc_id.upper())
    if com_match:
        year = com_match.group(1)
        num = com_match.group(2).zfill(4)
        return EURLEX_PDF_TEMPLATE.format(year=year, num_padded=num)
    return None


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _normalise_date(raw: str) -> str | None:
    raw = raw.strip()
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", raw)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw
    return None


def _find_nearby_date(tag: Any) -> str | None:
    """Walk up to three ancestor levels to find a date string near a tag."""
    node = tag
    for _ in range(3):
        if node is None:
            break
        text = node.get_text(" ", strip=True)
        m = re.search(r"\b(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})\b", text)
        if m:
            return _normalise_date(m.group(1))
        node = getattr(node, "parent", None)
    return None


# ---------------------------------------------------------------------------
# OEIL page scraping — regex patterns
# ---------------------------------------------------------------------------

_DOCEO_HREF_RE = re.compile(
    r"/doceo/document/([\w]+-(?:PR|AD|AM)-\d+)_EN\.(?:html|pdf)",
    re.IGNORECASE,
)
_A_HREF_RE = re.compile(
    r"/doceo/document/(A-\d+-\d{4}-\d+)_EN\.(?:html|pdf)",
    re.IGNORECASE,
)
_TA_HREF_RE = re.compile(
    r"/doceo/document/(TA-\d+-\d{4}-\d+)_EN\.(?:html|pdf)",
    re.IGNORECASE,
)
_COM_REGDATA_RE = re.compile(
    r"commission_europeenne/com/(\d{4})/(\d+)/COM_COM\((\d{4})\)(\d+)_EN\.pdf",
    re.IGNORECASE,
)
_COM_TEXT_RE = re.compile(r"\bCOM\s*\((\d{4})\)\s*(\d{3,4})\b", re.IGNORECASE)


def scrape_document_entries(
    procedure_id: str,
    session: Any,
    logger: Any = None,
) -> list[dict[str, Any]]:
    """Fetch the OEIL procedure page and return non-amendment document metadata.

    Parameters
    ----------
    procedure_id:
        EU procedure reference.
    session:
        A ``requests.Session`` (or compatible object).
    logger:
        Optional logger.

    Returns
    -------
    List of dicts with keys: ``doc_id``, ``doc_type``, ``date_str``, ``url``.
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    oeil_url = f"{OEIL_BASE}?reference={procedure_id}"
    _log(f"Fetching OEIL page: {oeil_url}")

    try:
        resp = session.get(oeil_url, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        _err(f"Could not fetch OEIL page: {exc}")
        return []

    time.sleep(RATE_LIMIT_OEIL)

    soup = BeautifulSoup(resp.text, "html.parser")
    found: dict[str, dict[str, Any]] = {}

    def _add(doc_id: str, doc_type: str, date_str: str | None, url: str | None) -> None:
        key = doc_id.upper()
        if key not in found:
            found[key] = {
                "doc_id": key,
                "doc_type": doc_type,
                "date_str": date_str,
                "url": url,
            }
            _log(f"  Found [{doc_type}]: {key}" + (f" (date={date_str})" if date_str else ""))

    for tag in soup.find_all("a", href=True):
        href: str = tag["href"]

        m = _DOCEO_HREF_RE.search(href)
        if m:
            doc_id = m.group(1).upper()
            doc_type = classify_document(doc_id)
            if doc_type is not None:
                date = _find_nearby_date(tag)
                _add(doc_id, doc_type, date, href if href.startswith("http") else None)
            continue

        m = _A_HREF_RE.search(href)
        if m:
            doc_id = m.group(1).upper()
            date = _find_nearby_date(tag)
            _add(doc_id, DOCTYPE_COMMITTEE_REPORT, date, href if href.startswith("http") else None)
            continue

        m = _TA_HREF_RE.search(href)
        if m:
            doc_id = m.group(1).upper()
            date = _find_nearby_date(tag)
            _add(doc_id, DOCTYPE_TEXT_ADOPTED, date, href if href.startswith("http") else None)
            continue

        m = _COM_REGDATA_RE.search(href)
        if m:
            year = m.group(1)
            num = m.group(2).lstrip("0") or "0"
            doc_id = f"COM({year}){num.zfill(4)}"
            date = _find_nearby_date(tag)
            _add(doc_id, DOCTYPE_COMMISSION_PROPOSAL, date, None)
            continue

    # Second pass: COM references in link text
    for tag in soup.find_all("a", href=True):
        link_text = tag.get_text(" ", strip=True)
        for m in _COM_TEXT_RE.finditer(link_text):
            year = m.group(1)
            num = m.group(2).zfill(4)
            doc_id = f"COM({year}){num}"
            if doc_id not in found:
                date = _find_nearby_date(tag)
                _add(doc_id, DOCTYPE_COMMISSION_PROPOSAL, date, None)

    # Third pass: raw page text for unlinked COM references
    page_text = soup.get_text(" ")
    for m in _COM_TEXT_RE.finditer(page_text):
        year = m.group(1)
        num = m.group(2).zfill(4)
        doc_id = f"COM({year}){num}"
        if doc_id not in found:
            _add(doc_id, DOCTYPE_COMMISSION_PROPOSAL, None, None)

    return list(found.values())


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------


def _download_with_fallback(
    doc_id: str,
    dest: Path,
    session: Any,
    logger: Any = None,
) -> tuple[Path | None, str | None]:
    """Try primary URL, then EUR-Lex fallback for COM proposals.

    Returns ``(local_path, final_url)`` on success, ``(None, None)`` on failure.
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    primary_url = build_pdf_url(doc_id)
    if primary_url is None:
        _err(f"No PDF URL pattern for {doc_id}")
        return None, None

    for attempt, url in enumerate(
        filter(None, [primary_url, build_eurlex_fallback_url(doc_id)])
    ):
        label = "primary" if attempt == 0 else "EUR-Lex fallback"
        _log(f"Downloading {doc_id} ({label}): {url}")
        try:
            resp = session.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            if resp.status_code == 404:
                _log(f"{doc_id}: 404 not available in EN ({label})")
                time.sleep(RATE_LIMIT_PDF)
                continue
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                _log(f"{doc_id}: HTML response received (skipping {label})")
                time.sleep(RATE_LIMIT_PDF)
                continue
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
            size_kb = dest.stat().st_size // 1024
            _log(f"Downloaded {dest.name} ({size_kb} KB)")
            time.sleep(RATE_LIMIT_PDF)
            return dest, url
        except Exception as exc:
            _err(f"Download failed for {doc_id} ({label}): {exc}")
            if dest.exists():
                dest.unlink()
            time.sleep(RATE_LIMIT_PDF)

    return None, None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def _extract_pdf_text(pdf_path: Path, logger: Any = None) -> str:
    """Run pdftotext -layout and return stdout. Returns empty string on error."""
    _err = logger.warning if logger else print
    try:
        result = subprocess.run(
            [PDFTOTEXT, "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return result.stdout
    except subprocess.CalledProcessError as exc:
        _err(f"pdftotext failed for {pdf_path.name}: {exc}")
        return ""
    except subprocess.TimeoutExpired:
        _err(f"pdftotext timed out for {pdf_path.name}")
        return ""
    except FileNotFoundError:
        _err(f"pdftotext not found at {PDFTOTEXT}. Install poppler.")
        return ""


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def build_row(
    procedure_id: str,
    entry: dict[str, Any],
    pdf_path: Path | None,
    pdf_url: str | None,
    content_text: str,
) -> dict[str, Any]:
    """Build a Supabase row dict from a scraped document entry."""
    doc_id = entry["doc_id"]
    file_size = pdf_path.stat().st_size if pdf_path and pdf_path.exists() else None

    page_count: int | None = None
    if content_text:
        page_count = content_text.count("\x0c") + 1

    return {
        "id": generate_document_id(procedure_id, doc_id),
        "procedure_id": procedure_id,
        "document_id": doc_id,
        "document_type": entry["doc_type"],
        "committee": extract_committee(doc_id),
        "rapporteur": None,
        "title": None,
        "url": entry.get("url"),
        "pdf_url": pdf_url or build_pdf_url(doc_id),
        "content_text": content_text or None,
        "event_date": entry.get("date_str"),
        "page_count": page_count,
        "file_size_bytes": file_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------


def upsert_documents(
    rows: list[dict[str, Any]],
    client: Any,
    batch_size: int = BATCH_SIZE,
    logger: Any = None,
) -> int:
    """Upsert document rows to ``procedure_documents`` in batches.

    Parameters
    ----------
    rows:
        Document rows ready for Supabase.
    client:
        Raw Supabase client (from ``SupabaseResource.get_client()``).
    batch_size:
        Number of rows per upsert call.
    logger:
        Optional logger.

    Returns
    -------
    Count of rows upserted.
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    if not rows:
        return 0

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        try:
            client.table("procedure_documents").upsert(batch, on_conflict="id").execute()
            total += len(batch)
            _log(
                f"Upserted batch {i // batch_size + 1}: {len(batch)} row(s) "
                f"(running total: {total})"
            )
        except Exception as exc:
            _err(f"Batch upsert failed at offset {i}: {exc}")

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


def fetch_procedures_with_documents(client: Any) -> set[str]:
    """Return procedure IDs that already have rows in procedure_documents."""
    rows: list[dict] = []
    offset = 0
    batch = 1000
    while True:
        resp = (
            client.table("procedure_documents")
            .select("procedure_id")
            .range(offset, offset + batch - 1)
            .execute()
        )
        rows.extend(resp.data)
        if len(resp.data) < batch:
            break
        offset += batch
    return {r["procedure_id"] for r in rows}


def scrape_procedure_documents(
    procedure_id: str,
    client: Any,
    session: Any,
    logger: Any = None,
) -> dict[str, int]:
    """Discover, download, parse, and upload documents for one procedure.

    Parameters
    ----------
    procedure_id:
        EU procedure reference, e.g. ``2021/0106(COD)``.
    client:
        Raw Supabase client (from ``SupabaseResource.get_client()``).
    session:
        A ``requests.Session`` (or compatible object).
    logger:
        Optional logger.

    Returns
    -------
    Dict mapping document type -> count uploaded, plus a ``"total"`` key.
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    _log(f"Starting document scrape for {procedure_id}")

    # Step 1: discover documents from OEIL
    entries = scrape_document_entries(procedure_id, session, logger=logger)
    if not entries:
        _err(f"No non-amendment documents found on the OEIL page for {procedure_id}")
        return {"total": 0}

    # Belt-and-suspenders: drop any amendments that slipped through
    entries = [e for e in entries if e.get("doc_type") is not None]
    _log(f"Found {len(entries)} document(s) for {procedure_id}")

    # Step 2 + 3: download PDFs and extract text
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="eu_documents_") as tmpdir:
        tmp_path = Path(tmpdir)

        for entry in entries:
            doc_id = entry["doc_id"]
            safe_name = re.sub(r"[^\w\-]", "_", doc_id)
            dest = tmp_path / f"{safe_name}_EN.pdf"

            pdf_path, pdf_url = _download_with_fallback(doc_id, dest, session, logger=logger)

            content_text = ""
            if pdf_path:
                content_text = _extract_pdf_text(pdf_path, logger=logger)
                word_count = len(content_text.split()) if content_text else 0
                _log(f"  {pdf_path.name}: {word_count} words extracted")

            rows.append(
                build_row(
                    procedure_id=procedure_id,
                    entry=entry,
                    pdf_path=pdf_path,
                    pdf_url=pdf_url,
                    content_text=content_text,
                )
            )

    if not rows:
        _log(f"Nothing to upload for {procedure_id}")
        return {"total": 0}

    # Step 4: upsert to Supabase
    uploaded = upsert_documents(rows, client, logger=logger)

    # Build per-type counts from the rows we attempted (including those without text)
    type_counts: dict[str, int] = {}
    for entry in entries:
        doc_type = entry.get("doc_type", "unknown")
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

    type_counts["total"] = uploaded
    _log(
        f"Uploaded {uploaded} document(s) for {procedure_id}: "
        + ", ".join(f"{k}={v}" for k, v in type_counts.items() if k != "total")
    )
    return type_counts
