"""Core logic for scraping EU legislative amendment PDFs from OEIL.

Functions here are adapted from scripts/scrape_amendments_to_supabase.py.
They accept a ``requests.Session`` and a raw Supabase client as parameters so
they work identically in both standalone and Dagster-asset contexts.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PDFTOTEXT = "/opt/homebrew/bin/pdftotext"
OEIL_BASE = "https://oeil.europarl.europa.eu/oeil/en/procedure-file"
DOCEO_PDF_BASE = "https://www.europarl.europa.eu/doceo/document"
RATE_LIMIT_SLEEP = 0.75
DOWNLOAD_TIMEOUT = 60
BATCH_SIZE = 100

# ---------------------------------------------------------------------------
# UUID / filename helpers
# ---------------------------------------------------------------------------


def generate_amendment_id(procedure_id: str, document_id: str, amendment_number: int) -> str:
    """Generate a deterministic UUID v5 for a (procedure, document, amendment) triple."""
    name_string = "::".join([procedure_id, document_id, str(amendment_number)])
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name_string))


def safe_id(procedure_id: str) -> str:
    """Convert '2023/0212(COD)' -> '2023_0212_COD' for safe filesystem use."""
    return re.sub(r"[/()]+", "_", procedure_id).strip("_")


def extract_committee(document_id: str) -> str:
    """Extract committee code from a document ID like 'ECON-AM-781235' -> 'ECON'."""
    parts = document_id.split("-")
    return parts[0] if parts else ""


def doc_id_from_filename(filename: str) -> str:
    """Derive canonical document ID from a PDF filename.

    Examples:
        ECON-AM-781235_EN.pdf            -> ECON-AM-781235
        ECON-AM-781235_amendments_1-100.pdf -> ECON-AM-781235
    """
    stem = Path(filename).stem
    stem = re.sub(r"_EN$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_amendments_[\d\-]+$", "", stem, flags=re.IGNORECASE)
    return stem


# ---------------------------------------------------------------------------
# PDF parsing helpers
# ---------------------------------------------------------------------------

_NOISE_LINES: list[re.Pattern] = [
    re.compile(r"^PE\d+\.\d+v\d+-\d+$"),
    re.compile(r"^AM\\[\w]+\.docx$"),
    re.compile(r"^PR\\[\w]+\.docx$"),
    re.compile(r"^\d+/\d+$"),
    re.compile(r"^EN$"),
    re.compile(r"^United in diversity$"),
    re.compile(r"^AM_Com_LegReport$"),
    re.compile(r"^PR_COD_1amCom$"),
]
_AMENDMENT_RE = re.compile(r"^Amendment\s+(\d+)\s*$")
_LOCATION_RE = re.compile(
    r"^((?:Recital|Article|Citation|Paragraph|Annex|Title)\s+[\w\s\-–()]+)$",
    re.IGNORECASE,
)
_HEADER_SEPARATOR = re.compile(
    r"^(?:Text proposed by the Commission|Draft legislative resolution)$",
    re.IGNORECASE,
)


def _is_noise(line: str) -> bool:
    return any(p.match(line.strip()) for p in _NOISE_LINES)


def _extract_pdf_text(pdf_path: Path) -> str:
    """Run pdftotext -layout and return stdout as a string."""
    result = subprocess.run(
        [PDFTOTEXT, "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _split_amendment_blocks(lines: list[str]) -> list[tuple[int, list[str]]]:
    """Split raw lines into (amendment_number, block_lines) pairs."""
    blocks: list[tuple[int, list[str]]] = []
    current_num: int | None = None
    current_lines: list[str] = []
    for line in lines:
        m = _AMENDMENT_RE.match(line.strip())
        if m:
            if current_num is not None:
                blocks.append((current_num, current_lines))
            current_num = int(m.group(1))
            current_lines = []
        elif current_num is not None:
            current_lines.append(line)
    if current_num is not None:
        blocks.append((current_num, current_lines))
    return blocks


def _extract_location_from_block(lines: list[str]) -> str:
    parts: list[str] = []
    for line in lines[:20]:
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADER_SEPARATOR.match(stripped):
            break
        if _LOCATION_RE.match(stripped):
            parts.append(stripped)
        elif re.match(r"^Proposal for a regulation$", stripped, re.IGNORECASE):
            continue
    return " / ".join(parts) if parts else ""


def _extract_authors_from_block(lines: list[str]) -> list[str]:
    authors: list[str] = []
    for line in lines[:15]:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(?:Proposal for a regulation|on behalf of)", stripped, re.IGNORECASE):
            break
        if re.match(r"^(?:Text proposed|Draft legislative)", stripped, re.IGNORECASE):
            break
        if _LOCATION_RE.match(stripped):
            break
        authors.append(stripped)
    return authors


def _extract_justification_from_block(lines: list[str]) -> str:
    start: int | None = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "justification":
            start = i + 1
            break
    if start is None:
        return ""
    parts: list[str] = []
    for line in lines[start:]:
        if re.match(r"^Or\.\s+\w{2}$", line.strip()):
            break
        parts.append(line.strip())
    return " ".join(filter(None, parts)).strip()


def _body_text(lines: list[str]) -> str:
    return " ".join(
        line.strip()
        for line in lines
        if line.strip()
        and not _is_noise(line)
        and not re.match(r"^Or\.\s+\w{2}$", line.strip())
    )


def _classify_target_type(location: str) -> str:
    if not location:
        return ""
    low = location.lower()
    if "recital" in low:
        return "recital"
    if "article" in low:
        return "article"
    if "annex" in low:
        return "annex"
    if "citation" in low:
        return "citation"
    if "paragraph" in low:
        return "paragraph"
    if "title" in low:
        return "title"
    return "other"


def parse_pdf_to_amendments(
    pdf_path: Path,
    document_id: str,
    procedure_id: str,
    event_date: str | None = None,
    logger: Any = None,
) -> list[dict[str, Any]]:
    """Parse a single amendment PDF and return a list of row dicts for Supabase.

    Each dict maps directly to the ``procedure_amendments`` table schema.

    Parameters
    ----------
    pdf_path:
        Local path to the PDF file.
    document_id:
        Canonical document ID, e.g. ``ECON-AM-781235``.
    procedure_id:
        EU procedure reference, e.g. ``2021/0106(COD)``.
    event_date:
        Optional ISO date string (``YYYY-MM-DD``) for the document event.
    logger:
        Optional logger (Dagster context.log or stdlib logging). Falls back to
        ``print`` if None.
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    try:
        text = _extract_pdf_text(pdf_path)
    except subprocess.CalledProcessError as exc:
        _err(f"pdftotext failed for {pdf_path.name}: {exc}")
        return []

    lines = [line for line in text.splitlines() if not _is_noise(line)]
    blocks = _split_amendment_blocks(lines)

    committee = extract_committee(document_id)
    rows: list[dict[str, Any]] = []

    for num, block_lines in blocks:
        authors = _extract_authors_from_block(block_lines)
        location = _extract_location_from_block(block_lines)
        justification = _extract_justification_from_block(block_lines)
        body = _body_text(block_lines)
        target_type = _classify_target_type(location)

        rows.append(
            {
                "id": generate_amendment_id(procedure_id, document_id, num),
                "procedure_id": procedure_id,
                "document_id": document_id,
                "amendment_number": num,
                "committee": committee,
                "target_element": location or None,
                "target_type": target_type or None,
                "original_text": None,
                "amended_text": body or None,
                "justification": justification or None,
                "submitted_by": authors if authors else None,
                "rapporteur_mep_id": None,
                "adopted": None,
                "work_type": None,
                "event_date": event_date or None,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# OEIL scraping
# ---------------------------------------------------------------------------


def scrape_amendment_doc_ids(
    procedure_id: str,
    session: Any,
    logger: Any = None,
) -> list[dict[str, str]]:
    """Fetch the OEIL procedure page and return amendment document metadata.

    Parameters
    ----------
    procedure_id:
        EU procedure reference.
    session:
        A ``requests.Session`` (or any object with a compatible ``.get()`` method).
    logger:
        Optional logger.

    Returns
    -------
    list of dicts with keys ``doc_id`` and ``date_str``.
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


def download_pdf(
    doc_id: str,
    dest_dir: Path,
    session: Any,
    logger: Any = None,
) -> Path | None:
    """Download the EN PDF for a document ID to dest_dir.

    Parameters
    ----------
    doc_id:
        Canonical document ID, e.g. ``ECON-AM-781235``.
    dest_dir:
        Local directory to save the PDF into.
    session:
        A ``requests.Session`` (or compatible object).
    logger:
        Optional logger.

    Returns
    -------
    Local ``Path`` on success, ``None`` on failure.
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    url = f"{DOCEO_PDF_BASE}/{doc_id}_EN.pdf"
    dest = dest_dir / f"{doc_id}_EN.pdf"

    if dest.exists():
        _log(f"Cached: {dest.name}")
        return dest

    _log(f"Downloading {url} ...")
    try:
        resp = session.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
        size_kb = dest.stat().st_size // 1024
        _log(f"Downloaded {dest.name} ({size_kb} KB)")
        return dest
    except Exception as exc:
        _err(f"Download failed for {doc_id}: {exc}")
        if dest.exists():
            dest.unlink()
        return None
    finally:
        time.sleep(RATE_LIMIT_SLEEP)


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------


def upsert_amendments(
    rows: list[dict[str, Any]],
    client: Any,
    batch_size: int = BATCH_SIZE,
    logger: Any = None,
) -> int:
    """Upsert rows into ``procedure_amendments`` in batches.

    Parameters
    ----------
    rows:
        Amendment rows ready for Supabase.
    client:
        Raw Supabase client (from ``SupabaseResource.get_client()``).
    batch_size:
        Number of rows per upsert call.
    logger:
        Optional logger.

    Returns
    -------
    Count of rows successfully upserted.
    """
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
            _log(
                f"Upserted batch {i // batch_size + 1}: {len(batch)} rows "
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


def scrape_procedure_amendments(
    procedure_id: str,
    client: Any,
    session: Any,
    logger: Any = None,
) -> int:
    """Discover, download, parse, and upload amendments for one procedure.

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
    Number of amendment rows upserted (0 if nothing found).
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    _log(f"Starting amendment scrape for {procedure_id}")

    # Step 1: discover amendment document IDs from OEIL
    doc_entries = scrape_amendment_doc_ids(procedure_id, session, logger=logger)
    if not doc_entries:
        _err(f"No amendment documents found on the OEIL page for {procedure_id}")
        return 0

    _log(f"Found {len(doc_entries)} unique amendment document(s) for {procedure_id}")

    # Step 2: download PDFs to a temp directory
    all_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="eu_amendments_") as tmpdir:
        tmp_path = Path(tmpdir)
        downloaded: list[tuple[Path, str, str]] = []

        for entry in doc_entries:
            doc_id = entry["doc_id"]
            date_str = entry["date_str"]
            pdf_path = download_pdf(doc_id, tmp_path, session, logger=logger)
            if pdf_path:
                downloaded.append((pdf_path, doc_id, date_str))

        if not downloaded:
            _err(f"No PDFs could be downloaded for {procedure_id}")
            return 0

        # Step 3: parse PDFs
        _log(f"Parsing {len(downloaded)} PDF(s) for {procedure_id} ...")
        for pdf_path, doc_id, date_str in downloaded:
            rows = parse_pdf_to_amendments(
                pdf_path=pdf_path,
                document_id=doc_id,
                procedure_id=procedure_id,
                event_date=date_str or None,
                logger=logger,
            )
            _log(f"  {pdf_path.name}: {len(rows)} amendments parsed")
            all_rows.extend(rows)

    _log(f"Total amendments parsed for {procedure_id}: {len(all_rows)}")

    if not all_rows:
        return 0

    # Step 4: upload to Supabase
    uploaded = upsert_amendments(all_rows, client, logger=logger)
    _log(f"Uploaded {uploaded} amendments for {procedure_id}")
    return uploaded
