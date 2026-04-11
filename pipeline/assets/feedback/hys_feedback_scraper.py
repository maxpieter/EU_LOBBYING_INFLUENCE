"""Bronze scraper: Have Your Say (HYS) feedback for EU legislation procedures.

Workflow:
1. Load procedures from Supabase that have a commission_document (COM number).
2. For each COM number, search the HYS REST API to find matching initiative(s).
3. For each matched initiative, paginate through all feedback submissions.
4. Filter out EU_CITIZEN respondents; keep organisations, businesses, NGOs, etc.
5. For long feedback text, chunk into overlapping windows for keyword search (RAG).
6. Upsert raw rows to `hys_feedback_bronze` and chunks to `hys_feedback_chunks`.

Key fields captured per feedback:
- feedback_id        : HYS internal ID (primary key)
- initiative_id      : HYS initiative numeric ID
- procedure_id       : Supabase procedure ID (OEIL reference) matched via COM number
- com_number         : Normalised COM number used for matching
- user_type          : Respondent category (ORGANISATION, BUSINESS_ASSOCIATION, etc.)
- transparency_reg_id: EU Transparency Register number (critical for org linkage)
- organisation_name  : Name of the respondent organisation
- country            : Country code
- language           : Submission language
- feedback_text      : Extracted plain text (from attachment or inline)
- date_feedback      : Submission date
- publication_status : Published/Awaiting/etc.
- raw_json           : Full API response row stored as-is for future re-parsing

HYS API endpoints used:
  Search initiatives by COM reference:
    GET /api/search?keyword=COM(2025)836&documentTypes=INITIATIVE&size=10
  Feedback list (paginated):
    GET /api/initiatives/{id}/feedbacks?page=0&size=25&language=ALL&sort=dateFeedback,DESC
  Full feedback detail (has attachment text):
    GET /api/feedback/{feedbackId}

COM number normalisation:
  Supabase stores: "COM(2025)0836" (with leading zeros)
  HYS search needs: "COM(2025)836" (no leading zeros)
  This module normalises both ways for robust matching.
"""

from __future__ import annotations

import io
import json
import re
import time
from typing import Any

import pdfplumber
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EC_BASE = "https://ec.europa.eu/info/law/better-regulation"
SEARCH_URL = f"{EC_BASE}/brpapi/searchInitiatives"
INITIATIVE_URL = f"{EC_BASE}/brpapi/groupInitiatives"
FEEDBACK_URL = f"{EC_BASE}/api/allFeedback"
HYS_DOWNLOAD_BASE = f"{EC_BASE}/api/download"

# Respondent types to EXCLUDE (we want organisations, not individual citizens)
EXCLUDED_USER_TYPES = {"EU_CITIZEN", "CITIZEN"}

# Pagination page size (HYS API supports up to 25)
PAGE_SIZE = 25

# Rate limiting: be polite to the EC servers
RATE_LIMIT_SLEEP = 0.8  # seconds between API calls

# Max chars per text chunk for keyword search / simplified RAG
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Upsert batch size
BATCH_SIZE = 50

DEFAULT_TIMEOUT = 20
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _make_session() -> requests.Session:
    """Create an HTTP session with browser-like headers for the EC portal."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{EC_BASE}/have-your-say/initiatives",
        }
    )
    return session


def _get_json(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    logger: Any = None,
    retries: int = MAX_RETRIES,
) -> dict | list | None:
    """GET a JSON endpoint with retry + rate limiting.

    Returns parsed JSON on success, None on permanent failure.
    """
    _log = logger.info if logger else print
    _warn = logger.warning if logger else print

    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT_SLEEP)
            resp = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            if attempt < retries - 1:
                wait = (attempt + 1) * 2
                _warn(f"  HTTP error {url}: {exc}. Retry in {wait}s...")
                time.sleep(wait)
            else:
                _warn(f"  Permanent failure {url}: {exc}")
                return None
        except json.JSONDecodeError as exc:
            _warn(f"  JSON parse error {url}: {exc}")
            return None

    return None


# ---------------------------------------------------------------------------
# COM number normalisation
# ---------------------------------------------------------------------------


def normalise_com_number(raw: str) -> str | None:
    """Normalise a COM number to the format HYS search expects.

    Supabase stores:  "COM(2025)0836"  (with leading zeros)
    HYS search needs: "COM(2025)836"   (no leading zeros)

    Returns None if the input doesn't look like a COM number.
    """
    if not raw:
        return None
    m = re.match(r"COM\s*\((\d{4})\)\s*0*(\d+)", raw, re.IGNORECASE)
    if not m:
        return None
    return f"COM({m.group(1)}){m.group(2)}"


def com_variants(com_number: str) -> list[str]:
    """Return search-friendly variants of a COM number.

    HYS search is a free-text keyword search, so we try a few forms.
    """
    normalised = normalise_com_number(com_number)
    if not normalised:
        return []
    # e.g. "COM(2025)836" and "COM/2025/836" (alternate notation seen in some titles)
    m = re.match(r"COM\((\d{4})\)(\d+)", normalised)
    if not m:
        return [normalised]
    year, num = m.group(1), m.group(2)
    return [
        f"COM({year}){num}",           # canonical
        f"COM/{year}/{num}",           # alternate slash notation
        f"COM(2025){num.zfill(4)}",    # with leading zeros (matches Supabase)
    ]


# ---------------------------------------------------------------------------
# HYS API: initiative search
# ---------------------------------------------------------------------------


def search_initiatives_by_com(
    com_number: str,
    session: requests.Session,
    logger: Any = None,
) -> list[dict]:
    """Search HYS for initiatives whose publications reference the COM number.

    Uses brpapi/searchInitiatives (keyword search), then verifies each
    candidate by fetching its detail and checking publication references.

    Returns a list of matching initiative dicts (usually 0 or 1),
    each with keys: id (int), shortTitle, detail.
    """
    _log = logger.info if logger else print

    normalised = normalise_com_number(com_number)
    if not normalised:
        return []

    data = _get_json(session, SEARCH_URL, params={
        "text": normalised,
        "size": 20,
        "language": "EN",
    }, logger=logger)

    if not data:
        return []

    # Response: {"initiativeResultDtoPage": {"content": [...], "totalElements": N}}
    items = data.get("initiativeResultDtoPage", {}).get("content", [])
    if not items:
        return []

    # Verify each candidate: fetch detail and check publication references
    variants = set(v.upper() for v in com_variants(com_number))
    matched = []
    for item in items:
        init_id = int(item["id"])
        detail = _get_json(session, f"{INITIATIVE_URL}/{init_id}", logger=logger)
        if not detail:
            continue
        for pub in detail.get("publications", []):
            pub_ref = (pub.get("reference") or "").upper()
            if any(v in pub_ref for v in variants):
                matched.append({**item, "id": init_id, "detail": detail})
                _log(f"  Matched initiative {init_id}: {item.get('shortTitle', '')[:60]}")
                break

    return matched


# ---------------------------------------------------------------------------
# HYS API: feedback pagination
# ---------------------------------------------------------------------------


def iter_feedbacks_for_initiative(
    initiative_id: int,
    session: requests.Session,
    logger: Any = None,
):
    """Yield non-citizen feedback dicts one at a time across all publication rounds.

    Yields items page-by-page so the caller never holds the full list in memory.
    Each yielded dict has `trNumber` remapped to `transparencyRegisterId`.
    """
    _log = logger.info if logger else print

    # Step 1: get publications for this initiative
    detail = _get_json(session, f"{INITIATIVE_URL}/{initiative_id}", logger=logger)
    if not detail:
        _log(f"  Initiative {initiative_id}: could not fetch detail")
        return

    publications = detail.get("publications", [])
    if not publications:
        _log(f"  Initiative {initiative_id}: no publications found")
        return

    seen_ids: set[int] = set()

    # Step 2: paginate feedback per publication
    for pub in publications:
        pub_id = pub.get("id")
        if not pub_id:
            continue

        page = 0
        while True:
            data = _get_json(session, FEEDBACK_URL, params={
                "publicationId": pub_id,
                "language": "EN",
                "page": page,
                "size": PAGE_SIZE,
                "sort": "dateFeedback,DESC",
            }, logger=logger)

            if not data:
                break

            # Response: {"_embedded": {"feedbackList": [...]}, "totalElements": N}
            # or: {"content": [...], "totalElements": N}
            items = (
                (data.get("_embedded") or {}).get("feedbackList")
                or data.get("content")
                or []
            )

            if not items:
                break

            total = data.get("totalElements", 0)

            if page == 0:
                _log(
                    f"  Publication {pub_id} ({pub.get('type', '')}): "
                    f"{total} total feedback entries"
                )

            for item in items:
                if item.get("userType", "") in EXCLUDED_USER_TYPES:
                    continue
                # Skip submissions with no inline text and no attachments
                has_text = bool(item.get("feedback", "").strip())
                has_attachments = bool(item.get("attachments"))
                if not has_text and not has_attachments:
                    continue
                fb_id = item.get("id")
                if fb_id in seen_ids:
                    continue
                seen_ids.add(fb_id)
                # Remap trNumber → transparencyRegisterId for consistent field naming
                if "trNumber" in item and "transparencyRegisterId" not in item:
                    item = dict(item)
                    item["transparencyRegisterId"] = item.pop("trNumber") or None
                yield item

            page += 1
            collected = page * PAGE_SIZE
            if collected >= total:
                break


def fetch_all_feedbacks_for_initiative(
    initiative_id: int,
    session: requests.Session,
    logger: Any = None,
) -> list[dict]:
    """Backwards-compatible wrapper — collects iter_feedbacks_for_initiative into a list."""
    return list(iter_feedbacks_for_initiative(initiative_id, session, logger=logger))



# ---------------------------------------------------------------------------
# PDF download + text extraction
# ---------------------------------------------------------------------------


def _extract_pdf_inline(raw_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber (inline, no subprocess).

    pdfplumber preserves visual line breaks (one \n per PDF line). We normalise
    these hard wraps into spaces so that sentences read as continuous text, while
    keeping double newlines (page boundaries / real blank lines) as paragraph
    separators for the downstream chunker.
    """
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        pages: list[str] = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Collapse single newlines (visual line wraps) into spaces
                normalised = re.sub(r"(?<!\n)\n(?!\n)", " ", text.strip())
                pages.append(normalised)
    return ("\n\n".join(pages)).strip()


def download_and_extract_pdf(
    document_id: str,
    session: requests.Session,
    logger: Any = None,
) -> str | None:
    """Download a feedback attachment PDF and extract its text.

    URL pattern confirmed from browser devtools:
      GET https://ec.europa.eu/info/law/better-regulation/api/download/{documentId}

    Returns extracted text string, or None if download/extraction fails.
    Skips scanned PDFs (no extractable text layer) silently.
    """
    _warn = logger.warning if logger else print

    url = f"{HYS_DOWNLOAD_BASE}/{document_id}"
    try:
        time.sleep(RATE_LIMIT_SLEEP)
        with session.get(
            url,
            headers={"Accept": "*/*"},
            timeout=60,
            stream=True,
        ) as resp:
            if resp.status_code == 404:
                _warn(f"  PDF not found: {document_id}")
                return None
            resp.raise_for_status()
            raw_chunks = []
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    raw_chunks.append(chunk)
        raw_bytes = b"".join(raw_chunks)
    except requests.exceptions.RequestException as exc:
        _warn(f"  Failed to download PDF {document_id}: {exc}")
        return None

    # Verify it's actually a PDF
    content_type = resp.headers.get("content-type", "")
    if "pdf" not in content_type.lower() and not raw_bytes[:4] == b"%PDF":
        _warn(f"  {document_id}: unexpected content-type {content_type!r}, skipping")
        return None

    try:
        full_text = _extract_pdf_inline(raw_bytes)
    except Exception as exc:
        _warn(f"  pdfplumber failed on {document_id}: {exc}")
        return None

    if not full_text:
        # Scanned PDF — no text layer, skip silently
        return None

    return full_text


def extract_text_from_attachments(
    attachments: list[dict],
    session: requests.Session,
    logger: Any = None,
) -> str | None:
    """Download and extract text from all attachments, concatenating results.

    Each attachment dict from the HYS API looks like:
      {"id": "090166e52a9ad0f1", "fileName": "position.pdf", ...}

    The `id` field is the documentId used in the download URL.
    Returns combined text from all PDFs, or None if nothing extracted.
    """
    _log = logger.info if logger else print

    all_text: list[str] = []
    for attachment in attachments:
        doc_id = attachment.get("documentId") or attachment.get("id")
        if not doc_id:
            continue
        filename = attachment.get("fileName", "")
        _log(f"    Downloading attachment: {filename} ({doc_id})")
        text = download_and_extract_pdf(doc_id, session, logger=logger)
        if text:
            all_text.append(text)
            _log(f"    Extracted {len(text):,} chars from {filename}")

    return "\n\n---\n\n".join(all_text) if all_text else None

# ---------------------------------------------------------------------------
# Text chunking for keyword search / simplified RAG
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks using recursive paragraph/sentence/word boundaries.

    Uses LangChain's RecursiveCharacterTextSplitter, which tries to split on
    paragraph breaks, then newlines, then sentences, then words — never mid-word.
    """
    if not text:
        return []
    # Collapse single newlines (PDF visual line wraps) into spaces, preserving
    # double newlines (real paragraph breaks) for the splitter to use.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def build_chunk_records(
    feedback_id: int,
    initiative_id: int,
    procedure_id: str,
    com_number: str,
    text: str,
    organisation_name: str | None,
    transparency_reg_id: str | None,
    date_feedback: str | None,
) -> list[dict]:
    """Build chunk records for `hys_feedback_chunks` table."""
    chunks = chunk_text(text)
    records = []
    for i, chunk_text_content in enumerate(chunks):
        records.append(
            {
                "feedback_id": feedback_id,
                "initiative_id": initiative_id,
                "procedure_id": procedure_id,
                "com_number": com_number,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "chunk_text": chunk_text_content,
                "organisation_name": organisation_name,
                "transparency_reg_id": transparency_reg_id,
                "date_feedback": date_feedback,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Transform raw API feedback row -> Supabase bronze row
# ---------------------------------------------------------------------------


def transform_feedback_row(
    raw: dict,
    initiative_id: int,
    procedure_id: str,
    com_number: str,
    pdf_text: str | None = None,
) -> dict:
    """Map a raw HYS API feedback dict to our `hys_feedback_bronze` schema.

    pdf_text: pre-extracted text from attachments (pass in after calling
              extract_text_from_attachments). Takes priority over inline feedback.
    """
    attachments = raw.get("attachments") or []

    # Priority: PDF attachment text > inline feedback field
    feedback_text = pdf_text or raw.get("feedback") or None

    org_name = raw.get("organization") or (
        f"{raw.get('firstName', '')} {raw.get('lastName', '')}".strip() or None
    )

    return {
        "feedback_id": raw["id"],
        "initiative_id": initiative_id,
        "procedure_id": procedure_id,
        "com_number": com_number,
        "user_type": raw.get("userType"),
        "transparency_reg_id": raw.get("transparencyRegisterId"),
        "organisation_name": org_name,
        "country": raw.get("country"),
        "language": raw.get("language"),
        "feedback_text": feedback_text,
        "attachment_count": len(attachments),
        "pdf_extracted": pdf_text is not None,
        "date_feedback": raw.get("dateFeedback"),
        "publication_status": raw.get("publicationStatus") or raw.get("status"),
        "raw_json": json.dumps(raw, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Supabase upsert helpers
# ---------------------------------------------------------------------------


def upsert_feedback_rows(
    rows: list[dict],
    client: Any,
    logger: Any = None,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Upsert rows into `hys_feedback_bronze` in batches."""
    _log = logger.info if logger else print
    _warn = logger.warning if logger else print

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        try:
            client.table("hys_feedback_bronze").upsert(
                batch, on_conflict="feedback_id"
            ).execute()
            total += len(batch)
            _log(f"  Upserted feedback batch {i // batch_size + 1}: {len(batch)} rows")
        except Exception as exc:
            _warn(f"  Feedback batch upsert failed at offset {i}: {exc}")
    return total


def upsert_chunk_rows(
    rows: list[dict],
    client: Any,
    logger: Any = None,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Upsert rows into `hys_feedback_chunks` in batches."""
    _log = logger.info if logger else print
    _warn = logger.warning if logger else print

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        _log(f"  Upserting chunk batch offset={i} size={len(batch)} feedback_id={batch[0].get('feedback_id')}")
        try:
            client.table("hys_feedback_chunks").upsert(
                batch, on_conflict="feedback_id,chunk_index"
            ).execute()
            total += len(batch)
            _log(f"  Chunk batch ok: {len(batch)} rows")
        except Exception as exc:
            _warn(f"  Chunk batch upsert FAILED at offset {i}: {type(exc).__name__}: {exc}")
    return total


# ---------------------------------------------------------------------------
# Supabase: load procedures
# ---------------------------------------------------------------------------


def fetch_procedures_with_com_numbers(
    client: Any,
    procedure_ids: list[str] | None = None,
) -> list[dict]:
    """Fetch procedures that have a commission_document (COM number).

    Returns list of dicts with keys: id (procedure_id), commission_document.
    """
    query = (
        client.table("procedures")
        .select("id, commission_document")
        .not_.is_("commission_document", "null")
    )
    if procedure_ids:
        query = query.in_("id", procedure_ids)

    resp = query.execute()
    return resp.data or []


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------


def scrape_hys_feedback_for_procedure(
    procedure_id: str,
    com_number: str,
    session: requests.Session,
    client: Any,
    logger: Any = None,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Scrape all non-citizen HYS feedback for a single procedure.

    Steps:
      1. Find matching HYS initiative(s) via COM number search.
      2. Paginate all feedback, filtering out citizens.
      3. Transform + upsert feedback rows.
      4. Build + upsert text chunks.

    Returns counts: {feedback_upserted, chunks_upserted, initiatives_found}.
    """
    _log = logger.info if logger else print
    _warn = logger.warning if logger else print

    normalised_com = normalise_com_number(com_number)
    if not normalised_com:
        _warn(f"  Cannot normalise COM number: {com_number!r}")
        return {"feedback_upserted": 0, "chunks_upserted": 0, "initiatives_found": 0}

    _log(f"Searching HYS for {procedure_id} ({normalised_com})")

    # Step 1: find initiative(s)
    initiatives = search_initiatives_by_com(normalised_com, session, logger=logger)
    if not initiatives:
        _log(f"  No HYS initiative found for {normalised_com}")
        return {"feedback_upserted": 0, "chunks_upserted": 0, "initiatives_found": 0}

    total_feedback = 0
    total_chunks = 0

    for initiative in initiatives:
        initiative_id = initiative["id"]

        # Optional: skip if we already have rows for this initiative+procedure
        if skip_existing:
            existing = (
                client.table("hys_feedback_bronze")
                .select("feedback_id", count="exact")
                .eq("initiative_id", initiative_id)
                .eq("procedure_id", procedure_id)
                .limit(1)
                .execute()
            )
            if existing.count and existing.count > 0:
                _log(
                    f"  Skipping initiative {initiative_id}: "
                    f"{existing.count} rows already exist"
                )
                continue

        # Steps 2–3: paginate feedback, download PDFs, upsert bronze rows
        # in rolling BATCH_SIZE windows. Chunking is handled by the
        # downstream hys_feedback_chunks asset.
        batch: list[dict] = []
        items_seen = 0

        def _flush(rows: list[dict]) -> None:
            nonlocal total_feedback
            total_feedback += upsert_feedback_rows(rows, client, logger=logger)

        for raw in iter_feedbacks_for_initiative(initiative_id, session, logger=logger):
            items_seen += 1
            attachments = raw.get("attachments") or []
            pdf_text = None
            if attachments:
                pdf_text = extract_text_from_attachments(
                    attachments, session, logger=logger
                )
            batch.append(
                transform_feedback_row(
                    raw, initiative_id, procedure_id, normalised_com,
                    pdf_text=pdf_text,
                )
            )

            if len(batch) >= BATCH_SIZE:
                _flush(batch)
                batch.clear()

        # Flush remainder
        if batch:
            _flush(batch)
            batch.clear()

        if items_seen == 0:
            _log(f"  No org-level feedback for initiative {initiative_id}")
            continue

    return {
        "feedback_upserted": total_feedback,
        "chunks_upserted": total_chunks,
        "initiatives_found": len(initiatives),
    }

