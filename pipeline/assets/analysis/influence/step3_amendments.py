"""Step 3: Amendment parsing — from local PDFs or Supabase."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from ._config import DATA_DIR, PDFTOTEXT
from ._helpers import safe_id
from ._supabase import fetch_all

# ---------------------------------------------------------------------------
# PDF parsing constants
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
    result = subprocess.run(
        [PDFTOTEXT, "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _split_amendment_blocks(lines: list[str]) -> list[tuple[int, list[str]]]:
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


def _parse_pdf_amendments(pdf_path: Path, source_label: str) -> list[dict[str, Any]]:
    text = _extract_pdf_text(pdf_path)
    lines = [line for line in text.splitlines() if not _is_noise(line)]
    blocks = _split_amendment_blocks(lines)
    amendments: list[dict[str, Any]] = []
    for num, block_lines in blocks:
        authors = _extract_authors_from_block(block_lines)
        location = _extract_location_from_block(block_lines)
        justification = _extract_justification_from_block(block_lines)
        body = _body_text(block_lines)
        summary = body[:200].replace("  ", " ").strip() + ("..." if len(body) > 200 else "")
        amendments.append(
            {
                "number": num,
                "source": source_label,
                "authors": authors,
                "location": location,
                "body": body,
                "justification": justification,
                "summary": summary,
                "themes": [],
            }
        )
    return amendments


def step3_parse_amendments(
    procedure_id: str,
    client: Any,
    logger: Any = None,
) -> list[dict[str, Any]]:
    """Parse amendments from PDFs or Supabase.

    Search order:
        1. PDF files in data/{procedure_id_safe}_documents/
        2. procedure_amendments table in Supabase
        3. Warn and return empty list if neither found
    """
    _log = logger.info if logger else print
    _warn = logger.warning if logger else print

    pid_safe = safe_id(procedure_id)
    docs_dir = DATA_DIR / f"{pid_safe}_documents"

    if docs_dir.exists():
        pdf_files = sorted(docs_dir.glob("*.pdf"))
        if pdf_files:
            _log(f"Found {len(pdf_files)} PDF(s) in {docs_dir}")
            all_amendments: list[dict[str, Any]] = []
            for pdf_path in pdf_files:
                label = pdf_path.stem
                try:
                    parsed = _parse_pdf_amendments(pdf_path, label)
                    _log(f"  {pdf_path.name}: {len(parsed)} amendments")
                    all_amendments.extend(parsed)
                except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                    _warn(f"  {pdf_path.name}: FAILED ({exc})")
            _log(f"Total from PDFs: {len(all_amendments)} amendments")
            return all_amendments

    _log(f"No PDFs in {docs_dir} — checking Supabase procedure_amendments ...")
    try:
        db_amendments = fetch_all(client, "procedure_amendments", "*", {"procedure_id": procedure_id})
        if db_amendments:
            _log(f"Found {len(db_amendments)} amendments in Supabase.")
            amendments: list[dict[str, Any]] = []
            for row in db_amendments:
                orig = row.get("original_text") or ""
                amend = row.get("amended_text") or ""
                body = f"{orig} {amend}".strip() if (orig or amend) else ""
                authors_raw = row.get("submitted_by") or []
                authors = [str(a) for a in authors_raw] if isinstance(authors_raw, list) else []
                amendments.append(
                    {
                        "number": row.get("amendment_number") or 0,
                        "source": row.get("document_id") or "supabase",
                        "authors": authors,
                        "location": row.get("target_element") or "",
                        "original_text": orig,
                        "amended_text": amend,
                        "body": body,
                        "justification": row.get("justification") or "",
                        "summary": body[:200] + ("..." if len(body) > 200 else ""),
                        "themes": [],
                    }
                )
            return amendments
    except Exception as exc:
        _warn(f"Supabase amendment query failed: {exc}")

    _warn("No amendments found — continuing with meetings-only analysis.")
    return []
