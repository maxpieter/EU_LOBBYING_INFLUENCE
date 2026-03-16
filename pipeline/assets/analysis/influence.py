"""Core functions for the 8-step EU lobbying influence analysis pipeline.

Adapted from scripts/influence_pipeline.py. The primary change is that every
function that previously called ``make_supabase_client()`` now accepts a
``client`` parameter (a raw Supabase client from
``SupabaseResource.get_client()``).

The AI provider configuration still reads from environment variables exactly
as the standalone script does (GEMINI_API_KEY, ANTHROPIC_API_KEY,
OPENAI_API_KEY), so no Dagster resource is required for AI access.

Steps
-----
1. step1_collect_data          — fetch procedure data from Supabase
2. step2_generate_taxonomy     — AI-assisted theme taxonomy (cached to disk)
3. step3_parse_amendments      — parse amendments from Supabase or local PDFs
4. step4_classify_amendments   — AI + regex theme classification
5. step5_extract_positions     — AI + regex position extraction from meetings
6. step6_quantitative_analysis — LEI / ALAS / ICI / Fisher's exact
7. step7_directional_alignment — AI-scored amendment-to-lobby alignment
8. step8_generate_report       — assemble JSON report and print summary
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional scipy
# ---------------------------------------------------------------------------

try:
    from scipy import stats as scipy_stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).parent
PROJECT_ROOT = _MODULE_DIR.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "analysis"
TAXONOMY_CACHE_DIR = DATA_DIR / "theme_taxonomies"
PDFTOTEXT = "/opt/homebrew/bin/pdftotext"

# ---------------------------------------------------------------------------
# AI provider configuration (global mutable state, mirrors standalone script)
# ---------------------------------------------------------------------------

AI_PROVIDER: str | None = None
AI_RATE_SLEEP: float = 0.5
AI_MAX_WORKERS: int = 3  # Conservative for CLI subprocess calls

GROQ_MODEL = "llama-3.3-70b-versatile"


def configure_ai_provider() -> None:
    """Detect which AI provider is available and configure the module global.

    Priority: claude-cli > groq > anthropic > openai
    """
    global AI_PROVIDER, AI_MAX_WORKERS, AI_RATE_SLEEP
    import shutil

    # Prefer claude CLI (uses OAuth, no API key needed, best model quality)
    if shutil.which("claude"):
        AI_PROVIDER = "claude-cli"
        AI_MAX_WORKERS = 3   # Subprocess overhead, keep low
        AI_RATE_SLEEP = 0.5
        return

    if os.getenv("GROQ_API_KEY"):
        try:
            from openai import OpenAI  # noqa: F401

            AI_PROVIDER = "groq"
            AI_MAX_WORKERS = 5
            AI_RATE_SLEEP = 2.0
            return
        except ImportError:
            pass

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401

            AI_PROVIDER = "anthropic"
            AI_MAX_WORKERS = 10
            AI_RATE_SLEEP = 0.5
            return
        except ImportError:
            pass

    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai  # noqa: F401

            AI_PROVIDER = "openai"
            AI_MAX_WORKERS = 10
            AI_RATE_SLEEP = 0.5
            return
        except ImportError:
            pass

    AI_PROVIDER = None


def ai_complete(prompt: str, system: str = "", json_mode: bool = False) -> str:
    """Send a prompt to the configured AI provider and return the text response."""
    if AI_PROVIDER is None:
        return ""

    if json_mode:
        prompt = (
            prompt
            + "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "Do not include any prose, markdown fences, or explanations outside the JSON."
        )

    try:
        if AI_PROVIDER == "claude-cli":
            return _ai_claude_cli(prompt, system)
        if AI_PROVIDER == "groq":
            return _ai_groq(prompt, system)
        if AI_PROVIDER == "anthropic":
            return _ai_anthropic(prompt, system)
        if AI_PROVIDER == "openai":
            return _ai_openai(prompt, system)
    except Exception:
        pass

    return ""


def _ai_claude_cli(prompt: str, system: str) -> str:
    """Call the claude CLI in pipe mode (uses OAuth, no API key needed)."""
    import subprocess

    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    result = subprocess.run(
        ["claude", "-p", "--model", "sonnet"],
        input=full_prompt,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr[:200]}")
    return result.stdout.strip()


def _ai_groq(prompt: str, system: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def _ai_anthropic(prompt: str, system: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    kwargs: dict[str, Any] = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    message = client.messages.create(**kwargs)
    return message.content[0].text if message.content else ""


def _ai_openai(prompt: str, system: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"} if "valid JSON" in prompt else None,
    )
    return response.choices[0].message.content or ""


def parse_json_response(raw: str, retry_prompt: str = "") -> dict | list | None:
    """Parse an AI JSON response, retrying once if parsing fails."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if not retry_prompt:
        return None

    nudge = (
        f"{retry_prompt}\n\nYour previous response was not valid JSON. "
        "Respond ONLY with valid JSON, nothing else."
    )
    retry_raw = ai_complete(nudge, json_mode=True)
    if not retry_raw:
        return None

    retry_cleaned = re.sub(r"^```(?:json)?\s*", "", retry_raw.strip(), flags=re.IGNORECASE)
    retry_cleaned = re.sub(r"\s*```$", "", retry_cleaned.strip())
    try:
        return json.loads(retry_cleaned)
    except json.JSONDecodeError:
        return None


def ai_complete_parallel(
    prompts: list[str],
    *,
    system: str = "",
    json_mode: bool = False,
    label: str = "AI",
    logger: Any = None,
) -> list[str]:
    """Execute multiple AI prompts concurrently using ThreadPoolExecutor."""
    _log = logger.info if logger else print

    if AI_PROVIDER is None or not prompts:
        return [""] * len(prompts)

    results: list[str] = [""] * len(prompts)
    completed = 0

    def _call(idx: int, prompt: str) -> tuple[int, str]:
        return idx, ai_complete(prompt, system=system, json_mode=json_mode)

    with ThreadPoolExecutor(max_workers=AI_MAX_WORKERS) as executor:
        futures = {executor.submit(_call, i, p): i for i, p in enumerate(prompts)}
        for future in as_completed(futures):
            try:
                idx, response = future.result()
                results[idx] = response
            except Exception as exc:
                idx = futures[future]
                if logger:
                    logger.warning(f"[{label}] Call {idx} failed: {exc}")
            completed += 1
            if completed % 20 == 0 or completed == len(prompts):
                _log(f"[{label}] {completed}/{len(prompts)} done")

    return results


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------


def fetch_all(
    client: Any,
    table: str,
    select: str = "*",
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Paginated fetch from any Supabase table (1000 rows per page)."""
    rows: list[dict[str, Any]] = []
    offset = 0
    batch = 1000
    while True:
        q = client.table(table).select(select).range(offset, offset + batch - 1)
        if filters:
            for col, val in filters.items():
                q = q.eq(col, val)
        resp = q.execute()
        rows.extend(resp.data)
        if len(resp.data) < batch:
            break
        offset += batch
    return rows


# ---------------------------------------------------------------------------
# Filename safety
# ---------------------------------------------------------------------------


def safe_id(procedure_id: str) -> str:
    """Convert '2023/0212(COD)' to a safe filename stem."""
    return re.sub(r"[/()]+", "_", procedure_id).strip("_")


# ---------------------------------------------------------------------------
# STEP 1: Data Collection (Deterministic)
# ---------------------------------------------------------------------------


def step1_collect_data(
    procedure_id: str,
    client: Any,
    logger: Any = None,
) -> dict[str, Any]:
    """Fetch all data for the procedure from Supabase.

    Returns a dict with keys: ``procedure``, ``articles``, ``lobbying``,
    ``commission``.
    """
    _log = logger.info if logger else print

    _log(f"STEP 1: Fetching procedure {procedure_id!r} ...")

    proc_resp = (
        client.table("procedures")
        .select("id,title,description,events,actors")
        .eq("id", procedure_id)
        .limit(1)
        .execute()
    )
    if not proc_resp.data:
        raise ValueError(f"Procedure {procedure_id!r} not found in database.")
    procedure = proc_resp.data[0]
    _log(f"Title: {procedure.get('title', 'N/A')}")

    # Fetch commission proposal text from procedure_documents (richer than procedure_articles)
    proposal_docs = fetch_all(
        client,
        "procedure_documents",
        "procedure_id,document_id,document_type,content_text",
        {"procedure_id": procedure_id, "document_type": "commission_proposal"},
    )
    proposal_text = ""
    if proposal_docs:
        proposal_text = (proposal_docs[0].get("content_text") or "")[:5000]
        _log(f"Commission proposal text: {len(proposal_text)} chars")

    commission_meetings = _fetch_commission_meetings(client, procedure_id)
    commission_meetings = _enrich_commission_meetings(client, commission_meetings)
    with_notes = sum(1 for m in commission_meetings if m.get("points_raised"))
    _log(
        f"Commission meetings: {len(commission_meetings)} ({with_notes} with points_raised)"
    )

    lobbying_meetings = _fetch_lobbying_meetings(client, procedure_id)
    if lobbying_meetings:
        lobbying_meetings = _enrich_lobbying_meetings(client, lobbying_meetings)
    _log(f"EP lobbying meetings: {len(lobbying_meetings)}")

    unique_orgs = {m.get("org_name", "") for m in lobbying_meetings if m.get("org_name")}
    unique_meps = {m.get("mep_name", "") for m in lobbying_meetings if m.get("mep_name")}
    _log(f"Unique lobbying organisations: {len(unique_orgs)}")
    _log(f"Unique MEPs with meetings: {len(unique_meps)}")

    return {
        "procedure": procedure,
        "proposal_text": proposal_text,
        "lobbying": lobbying_meetings,
        "commission": commission_meetings,
    }


def _fetch_commission_meetings(client: Any, procedure_id: str) -> list[dict[str, Any]]:
    direct = fetch_all(
        client,
        "commission_meetings",
        "id,commissioner_name,meeting_date,subject,organizations_raw,points_raised,conclusions",
        {"matched_procedure_id": procedure_id},
    )
    link_rows = fetch_all(
        client,
        "meeting_procedure_links",
        "commission_meeting_id",
        {"procedure_id": procedure_id},
    )
    linked_ids = {r["commission_meeting_id"] for r in link_rows if r.get("commission_meeting_id")}
    direct_ids = {m["id"] for m in direct}
    for mid in linked_ids - direct_ids:
        resp = (
            client.table("commission_meetings")
            .select(
                "id,commissioner_name,meeting_date,subject,organizations_raw,"
                "points_raised,conclusions"
            )
            .eq("id", mid)
            .limit(1)
            .execute()
        )
        if resp.data:
            direct.extend(resp.data)
    return direct


def _fetch_lobbying_meetings(client: Any, procedure_id: str) -> list[dict[str, Any]]:
    link_rows = fetch_all(
        client,
        "meeting_procedure_links",
        "lobbying_meeting_id",
        {"procedure_id": procedure_id},
    )
    linked_ids = [r["lobbying_meeting_id"] for r in link_rows if r.get("lobbying_meeting_id")]
    direct = fetch_all(
        client,
        "lobbying_meetings",
        "id,mep_id,organization_id,meeting_date,title,capacity,related_procedure",
        {"related_procedure": procedure_id},
    )
    direct_ids = {m["id"] for m in direct}
    for mid in linked_ids:
        if mid not in direct_ids:
            resp = (
                client.table("lobbying_meetings")
                .select("id,mep_id,organization_id,meeting_date,title,capacity,related_procedure")
                .eq("id", mid)
                .limit(1)
                .execute()
            )
            if resp.data:
                direct.extend(resp.data)
                direct_ids.add(mid)
    return direct


def _enrich_commission_meetings(
    client: Any,
    meetings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not meetings:
        return meetings
    meeting_ids = [m["id"] for m in meetings]
    all_links: list[dict[str, Any]] = []
    for i in range(0, len(meeting_ids), 200):
        chunk = meeting_ids[i : i + 200]
        resp = (
            client.table("commission_meeting_organizations")
            .select("meeting_id,organization_id,organization_name")
            .in_("meeting_id", chunk)
            .limit(10000)
            .execute()
        )
        if resp.data:
            all_links.extend(resp.data)

    org_ids = list({lnk["organization_id"] for lnk in all_links if lnk.get("organization_id")})
    org_ir_map: dict[str, str] = {}
    for i in range(0, len(org_ids), 200):
        chunk = org_ids[i : i + 200]
        resp = (
            client.table("organizations")
            .select("id,interests_represented")
            .in_("id", chunk)
            .limit(10000)
            .execute()
        )
        for row in resp.data or []:
            if row.get("interests_represented"):
                org_ir_map[row["id"]] = row["interests_represented"]

    meeting_orgs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lnk in all_links:
        meeting_orgs[lnk["meeting_id"]].append(
            {
                "name": lnk.get("organization_name", ""),
                "interests_represented": org_ir_map.get(lnk.get("organization_id", ""), "Unknown"),
            }
        )
    for m in meetings:
        m["resolved_orgs"] = meeting_orgs.get(m["id"], [])
    return meetings


def _enrich_lobbying_meetings(
    client: Any,
    meetings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    org_ids = list({m["organization_id"] for m in meetings if m.get("organization_id")})
    org_map: dict[str, dict[str, Any]] = {}
    for i in range(0, len(org_ids), 200):
        chunk = org_ids[i : i + 200]
        resp = (
            client.table("organizations")
            .select("id,name,interests_represented")
            .in_("id", chunk)
            .limit(10000)
            .execute()
        )
        for row in resp.data or []:
            org_map[row["id"]] = row

    mep_ids = list({m["mep_id"] for m in meetings if m.get("mep_id")})
    mep_map: dict[int, str] = {}
    for i in range(0, len(mep_ids), 200):
        chunk = mep_ids[i : i + 200]
        resp = (
            client.table("meps")
            .select('id,"fullName"')
            .in_("id", chunk)
            .limit(10000)
            .execute()
        )
        for row in resp.data or []:
            mep_map[row["id"]] = row.get("fullName", f"MEP {row['id']}")

    for m in meetings:
        org_data = org_map.get(m.get("organization_id", ""), {})
        m["org_name"] = org_data.get("name", "")
        m["interests_represented"] = org_data.get("interests_represented") or "Unknown"
        m["mep_name"] = mep_map.get(m.get("mep_id"), "Unknown MEP")
    return meetings


# ---------------------------------------------------------------------------
# STEP 2: Theme Taxonomy Generation (AI-Assisted, cached)
# ---------------------------------------------------------------------------


def step2_generate_taxonomy(
    procedure_id: str,
    data: dict[str, Any],
    no_ai: bool = False,
    regen: bool = False,
    logger: Any = None,
) -> dict[str, Any]:
    """Generate or load a policy-theme taxonomy for the procedure.

    Parameters
    ----------
    procedure_id:
        EU procedure reference.
    data:
        Output of ``step1_collect_data``.
    no_ai:
        When True, skip AI calls and return empty taxonomy.
    regen:
        When True, delete the cached taxonomy and regenerate via AI.
    logger:
        Optional logger.
    """
    _log = logger.info if logger else print

    TAXONOMY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TAXONOMY_CACHE_DIR / f"{safe_id(procedure_id)}.json"

    if regen and cache_path.exists():
        cache_path.unlink()
        _log(f"Cleared taxonomy cache: {cache_path}")

    if cache_path.exists():
        _log(f"Loading cached taxonomy from {cache_path}")
        with cache_path.open(encoding="utf-8") as fh:
            taxonomy = json.load(fh)
        _log(f"Loaded {len(taxonomy)} themes: {', '.join(taxonomy.keys())}")
        return taxonomy

    if no_ai or AI_PROVIDER is None:
        _log("No AI provider available — returning empty taxonomy (regex-only mode).")
        return {}

    procedure = data["procedure"]
    title = procedure.get("title", "Unknown procedure")
    description = (procedure.get("description") or "")[:2000]
    proposal_text = data.get("proposal_text", "")

    system_prompt = (
        "You are an expert EU policy analyst specialising in legislative analysis "
        "and lobbying research. You identify contested political dimensions in "
        "EU legislative proposals by studying the text and understanding the "
        "stakeholder landscape."
    )

    user_prompt = f"""Analyse the following EU legislative procedure and identify 5-12 distinct, contested policy themes (political dimensions) that different stakeholders are likely to lobby on.

PROCEDURE: {procedure_id}
TITLE: {title}
DESCRIPTION: {description}

{"COMMISSION PROPOSAL EXCERPT:" + chr(10) + proposal_text if proposal_text else ""}

For each theme, respond with a JSON object structured as follows:

{{
  "themes": [
    {{
      "key": "snake_case_theme_key",
      "description": "One-sentence human-readable description of the policy dimension",
      "articles": ["Article 3", "Recital 5"],
      "keywords": [
        "regex_pattern_1",
        "regex_pattern_2",
        "regex_pattern_3",
        "regex_pattern_4",
        "regex_pattern_5"
      ],
      "salience": "Why this dimension is politically contested and which stakeholders care"
    }}
  ]
}}

Requirements:
- 5-12 themes, each genuinely distinct from the others
- keywords must be valid Python regex patterns (use \\b for word boundaries, \\s+ for spaces)
- articles should reference specific Articles or Recitals from the proposal
- themes should reflect real stakeholder conflicts (industry vs. civil society, member state vs. Commission, etc.)

Respond ONLY with the JSON object above."""

    _log("Calling AI to generate theme taxonomy ...")
    raw = ai_complete(user_prompt, system=system_prompt, json_mode=True)
    time.sleep(AI_RATE_SLEEP)

    parsed = parse_json_response(raw, retry_prompt=user_prompt)
    if not parsed or not isinstance(parsed, dict):
        _log("[WARN] Could not parse AI taxonomy response — using empty taxonomy.")
        return {}

    themes_list = parsed.get("themes", [])
    if not themes_list:
        _log("[WARN] AI returned no themes — using empty taxonomy.")
        return {}

    taxonomy: dict[str, Any] = {}
    for theme in themes_list:
        key = theme.get("key", "").strip()
        if not key:
            continue
        taxonomy[key] = {
            "description": theme.get("description", ""),
            "articles": theme.get("articles", []),
            "keywords": theme.get("keywords", []),
            "salience": theme.get("salience", ""),
        }

    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(taxonomy, fh, indent=2, ensure_ascii=False)
    _log(f"Generated {len(taxonomy)} themes; cached to {cache_path}")
    _log(f"Themes: {', '.join(taxonomy.keys())}")
    return taxonomy


# ---------------------------------------------------------------------------
# STEP 3: Amendment Parsing (Deterministic)
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


# ---------------------------------------------------------------------------
# Regex classification helpers
# ---------------------------------------------------------------------------


def compile_taxonomy_patterns(taxonomy: dict[str, Any]) -> dict[str, list[re.Pattern]]:
    compiled: dict[str, list[re.Pattern]] = {}
    for key, cfg in taxonomy.items():
        patterns: list[re.Pattern] = []
        for raw_pattern in cfg.get("keywords", []):
            try:
                patterns.append(re.compile(raw_pattern, re.IGNORECASE))
            except re.error:
                pass
        compiled[key] = patterns
    return compiled


def _classify_by_regex(text: str, patterns: dict[str, list[re.Pattern]]) -> list[str]:
    if not text:
        return []
    matched: list[str] = []
    for theme, pats in patterns.items():
        for pat in pats:
            if pat.search(text):
                matched.append(theme)
                break
    return matched


def _meeting_text(meeting: dict[str, Any]) -> str:
    return " ".join(
        filter(
            None,
            [
                meeting.get("subject"),
                meeting.get("points_raised"),
                meeting.get("conclusions"),
                meeting.get("title"),
            ],
        )
    )


def _taxonomy_summary_for_prompt(taxonomy: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, cfg in taxonomy.items():
        desc = cfg.get("description", "")
        kw_sample = ", ".join(cfg.get("keywords", [])[:4])
        lines.append(f'  "{key}": {desc}  [keywords: {kw_sample}]')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# STEP 4: Theme Classification (AI-Assisted with regex fallback)
# ---------------------------------------------------------------------------


def step4_classify_amendments(
    amendments: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    no_ai: bool = False,
    logger: Any = None,
) -> list[dict[str, Any]]:
    """Classify each amendment against the theme taxonomy."""
    _log = logger.info if logger else print

    if not amendments:
        _log("No amendments to classify.")
        return amendments

    compiled = compile_taxonomy_patterns(taxonomy)
    taxonomy_summary = _taxonomy_summary_for_prompt(taxonomy)

    if no_ai or AI_PROVIDER is None or not taxonomy:
        _log(f"Classifying {len(amendments)} amendments via regex only ...")
        for am in amendments:
            text = (
                am.get("body", "")
                + " "
                + am.get("justification", "")
                + " "
                + am.get("location", "")
            )
            am["themes"] = _classify_by_regex(text, compiled)
        classified = sum(1 for a in amendments if a["themes"])
        _log(f"Classified: {classified}/{len(amendments)} matched at least one theme")
        return amendments

    batch_size = 15
    batches = [amendments[i : i + batch_size] for i in range(0, len(amendments), batch_size)]
    _log(
        f"Classifying {len(amendments)} amendments in {len(batches)} batch(es) "
        f"of up to {batch_size} (parallel, {AI_MAX_WORKERS} workers) ..."
    )

    batch_prompts: list[str] = []
    for batch in batches:
        items_text = "\n\n".join(
            f"[AM-{am['number']}] {am['location']}\n{am['body'][:600]}" for am in batch
        )
        batch_prompts.append(
            f"""Classify each amendment excerpt by policy theme.

TAXONOMY:
{taxonomy_summary}

AMENDMENTS TO CLASSIFY:
{items_text}

For each amendment, return a JSON array where each entry has:
  "number": the amendment number (integer)
  "themes": list of theme keys from the taxonomy (may be empty list)

Example: [{{"number": 42, "themes": ["holding_limits", "privacy_data_protection"]}}, ...]

Only use theme keys from the taxonomy. An amendment may match 0, 1, or multiple themes.
Respond ONLY with the JSON array."""
        )

    raw_responses = ai_complete_parallel(
        batch_prompts, json_mode=True, label="classify", logger=logger
    )

    for batch, raw in zip(batches, raw_responses):
        parsed = parse_json_response(raw) if raw else None
        if parsed and isinstance(parsed, list):
            result_map = {entry.get("number"): entry.get("themes", []) for entry in parsed}
            for am in batch:
                ai_themes = result_map.get(am["number"])
                if ai_themes is not None:
                    am["themes"] = [t for t in ai_themes if t in taxonomy]
                else:
                    text = am.get("body", "") + " " + am.get("justification", "") + " " + am.get("location", "")
                    am["themes"] = _classify_by_regex(text, compiled)
        else:
            for am in batch:
                text = am.get("body", "") + " " + am.get("justification", "") + " " + am.get("location", "")
                am["themes"] = _classify_by_regex(text, compiled)

    classified = sum(1 for a in amendments if a["themes"])
    _log(f"Classified: {classified}/{len(amendments)} matched at least one theme")
    return amendments


# ---------------------------------------------------------------------------
# STEP 5: Position Extraction (AI-Assisted)
# ---------------------------------------------------------------------------


def step5_extract_positions(
    commission_meetings: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    compiled_patterns: dict[str, list[re.Pattern]],
    no_ai: bool = False,
    logger: Any = None,
) -> list[dict[str, Any]]:
    """Extract structured positions from commission meeting texts."""
    _log = logger.info if logger else print

    substantive = [
        m for m in commission_meetings
        if m.get("points_raised") and len(m.get("points_raised", "")) > 50
    ]
    _log(f"Commission meetings with substantive text: {len(substantive)}")

    if not substantive:
        _log("Nothing to extract.")
        return []

    positions: list[dict[str, Any]] = []
    for m in substantive:
        text = _meeting_text(m)
        themes = _classify_by_regex(text, compiled_patterns)
        resolved = m.get("resolved_orgs") or []
        org_names = [o["name"] for o in resolved if o.get("name")]
        if not org_names:
            raw = (m.get("organizations_raw") or "").strip()
            org_names = [raw.split("|")[0].strip()] if raw else ["Unknown"]
        positions.append(
            {
                "meeting_id": m.get("id"),
                "date": str(m.get("meeting_date", ""))[:10],
                "commissioner": m.get("commissioner_name", ""),
                "orgs": org_names,
                "themes": themes,
                "direction": "unknown",
                "summary": (m.get("points_raised") or "")[:200],
                "ai_enhanced": False,
            }
        )

    if no_ai or AI_PROVIDER is None or not taxonomy:
        _log(f"Extracted {len(positions)} positions (regex only).")
        return positions

    taxonomy_summary = _taxonomy_summary_for_prompt(taxonomy)
    batch_size = 4
    batches = [substantive[i : i + batch_size] for i in range(0, len(substantive), batch_size)]
    _log(
        f"Enhancing {len(substantive)} meetings in {len(batches)} batch(es) via AI ..."
    )

    meeting_id_to_position = {p["meeting_id"]: p for p in positions}

    batch_prompts: list[str] = []
    for batch in batches:
        items = []
        for m in batch:
            resolved = m.get("resolved_orgs") or []
            org_names_str = ", ".join(o["name"] for o in resolved if o.get("name")) or (
                m.get("organizations_raw") or "Unknown"
            )
            items.append(
                f'Meeting ID: {m["id"]}\n'
                f'Date: {m.get("meeting_date", "")}\n'
                f'Organisations: {org_names_str}\n'
                f'Points raised: {(m.get("points_raised") or "")[:800]}'
            )

        batch_prompts.append(
            f"""Extract structured lobbying positions from these Commission meeting records.

TAXONOMY:
{taxonomy_summary}

MEETINGS:
{"---".join(items)}

For each meeting, return a JSON array where each entry has:
  "meeting_id": the meeting ID string
  "themes": list of relevant theme keys from the taxonomy
  "direction": one of "supports", "opposes", "modifies", or "unclear"
               (relative to the original Commission proposal)
  "summary": one sentence capturing the core position taken

Respond ONLY with the JSON array."""
        )

    raw_responses = ai_complete_parallel(
        batch_prompts, json_mode=True, label="positions", logger=logger
    )

    for raw in raw_responses:
        parsed = parse_json_response(raw) if raw else None
        if parsed and isinstance(parsed, list):
            for entry in parsed:
                mid = entry.get("meeting_id")
                if mid and mid in meeting_id_to_position:
                    pos = meeting_id_to_position[mid]
                    ai_themes = [t for t in (entry.get("themes") or []) if t in taxonomy]
                    if ai_themes:
                        pos["themes"] = ai_themes
                    direction = entry.get("direction", "").lower()
                    if direction in ("supports", "opposes", "modifies", "unclear"):
                        pos["direction"] = direction
                    if entry.get("summary"):
                        pos["summary"] = entry["summary"][:500]
                    pos["ai_enhanced"] = True

    enhanced = sum(1 for p in positions if p.get("ai_enhanced"))
    _log(f"Positions extracted: {len(positions)} ({enhanced} AI-enhanced)")
    return positions


# ---------------------------------------------------------------------------
# STEP 6: Quantitative Analysis (Deterministic)
# ---------------------------------------------------------------------------

_ORG_INTEREST_WEIGHTS: dict[str, float] = {
    "Promotes their own interests or the collective interests of their members": 1.0,
    "Advances interests of their clients": 0.8,
    "Does not represent commercial interests": 0.3,
    "Unknown": 0.5,
}


def _interest_weight(interests_represented: str) -> float:
    return _ORG_INTEREST_WEIGHTS.get(interests_represented, 0.5)


def _extract_rapporteurs(procedure: dict[str, Any]) -> dict[str, dict[str, str]]:
    actors_raw = procedure.get("actors")
    if not actors_raw:
        return {}
    if isinstance(actors_raw, str):
        try:
            actors_raw = json.loads(actors_raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(actors_raw, list):
        return {}

    responsible_committee = None
    for actor in actors_raw:
        if isinstance(actor, dict) and actor.get("role") == "committee_responsible":
            responsible_committee = actor.get("committee_code")
            break

    result: dict[str, dict[str, str]] = {}
    for actor in actors_raw:
        if not isinstance(actor, dict):
            continue
        if actor.get("actor_type") != "mep":
            continue
        role = actor.get("role", "")
        if role not in ("rapporteur", "shadow_rapporteur", "opinion_rapporteur"):
            continue
        raw_name = (actor.get("mep_name") or "").strip()
        if not raw_name:
            continue

        party_match = re.search(r"\(([^)]+)\)\s*$", raw_name)
        party = party_match.group(1) if party_match else ""
        name_part = re.sub(r"\s*\([^)]+\)\s*$", "", raw_name).strip()

        parts = name_part.split()
        if len(parts) >= 2:
            first_idx = len(parts)
            for i, p in enumerate(parts):
                if p != p.upper() and i > 0:
                    first_idx = i
                    break
            first_name = " ".join(parts[first_idx:])
            last_name = " ".join(parts[:first_idx])
            canonical = f"{first_name} {last_name}".strip()
        else:
            canonical = name_part

        committee_code = actor.get("committee_code")
        if role == "rapporteur" and committee_code == responsible_committee:
            result[canonical] = {"party": party, "role": "Rapporteur"}
        elif role in ("rapporteur", "shadow_rapporteur"):
            result[canonical] = {"party": party, "role": "Shadow"}
        elif role == "opinion_rapporteur":
            result[canonical] = {"party": party, "role": "Opinion Rapporteur"}

    return result


def _build_mep_crossref(
    amendments: list[dict[str, Any]],
    lobbying_meetings: list[dict[str, Any]],
    commission_meetings: list[dict[str, Any]],
    key_meps: dict[str, dict[str, str]],
    compiled_patterns: dict[str, list[re.Pattern]],
) -> dict[str, Any]:
    org_known_themes: dict[str, set[str]] = defaultdict(set)
    for m in commission_meetings:
        text = _meeting_text(m)
        themes = _classify_by_regex(text, compiled_patterns)
        for org_info in (m.get("resolved_orgs") or []):
            name = (org_info.get("name") or "").strip().lower()
            if name:
                org_known_themes[name].update(themes)
        raw = (m.get("organizations_raw") or "").strip()
        if raw and not m.get("resolved_orgs"):
            org_known_themes[raw.lower()].update(themes)

    alias_map: dict[str, str] = {}
    for canonical in key_meps:
        parts = canonical.lower().split()
        alias_map[canonical.lower()] = canonical
        if parts:
            alias_map[parts[-1]] = canonical
        if len(parts) >= 2:
            alias_map[f"{parts[0]} {parts[-1]}"] = canonical

    def _resolve_mep_name(raw_name: str) -> str | None:
        low = raw_name.strip().lower()
        if low in alias_map:
            return alias_map[low]
        for fragment, canonical in alias_map.items():
            if fragment in low:
                return canonical
        return None

    mep_am_themes: dict[str, Counter] = defaultdict(Counter)
    mep_am_total: dict[str, int] = defaultdict(int)
    mep_am_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for am in amendments:
        authors = am.get("authors") or []
        if isinstance(authors, str):
            authors = [authors]
        for raw_author in authors:
            canonical = _resolve_mep_name(raw_author)
            if not canonical:
                continue
            mep_am_total[canonical] += 1
            for t in am.get("themes", []):
                mep_am_themes[canonical][t] += 1
                key_ex = f"{canonical}:{t}"
                if len(mep_am_examples[key_ex]) < 5:
                    mep_am_examples[key_ex].append(
                        {
                            "number": am.get("number"),
                            "source": am.get("source"),
                            "location": am.get("location", ""),
                            "body_excerpt": (am.get("body") or "")[:300],
                        }
                    )

    mep_mtg_themes: dict[str, Counter] = defaultdict(Counter)
    mep_mtg_orgs: dict[str, Counter] = defaultdict(Counter)
    mep_mtg_total: dict[str, int] = defaultdict(int)

    for m in lobbying_meetings:
        mep_name = m.get("mep_name", "")
        canonical = _resolve_mep_name(mep_name)
        if not canonical:
            continue
        org = m.get("org_name") or "Unknown"
        text = _meeting_text(m)
        themes = set(_classify_by_regex(text, compiled_patterns))
        themes.update(org_known_themes.get(org.lower(), set()))
        mep_mtg_total[canonical] += 1
        mep_mtg_orgs[canonical][org] += 1
        for t in themes:
            mep_mtg_themes[canonical][t] += 1

    all_meps = set(mep_am_total.keys()) | set(mep_mtg_total.keys()) | set(key_meps.keys())
    result: dict[str, Any] = {}
    for mep in sorted(all_meps):
        am_themes = dict(mep_am_themes.get(mep, Counter()).most_common())
        mtg_themes = dict(mep_mtg_themes.get(mep, Counter()).most_common())
        overlapping = sorted(set(am_themes) & set(mtg_themes))
        top_orgs = [
            {"org": o, "count": c}
            for o, c in mep_mtg_orgs.get(mep, Counter()).most_common(10)
        ]
        theme_details: dict[str, Any] = {}
        for t in overlapping:
            key_ex = f"{mep}:{t}"
            theme_details[t] = {
                "amendments_on_theme": am_themes.get(t, 0),
                "meetings_on_theme": mtg_themes.get(t, 0),
                "amendment_examples": mep_am_examples.get(key_ex, []),
            }
        result[mep] = {
            "role": key_meps.get(mep, {}).get("role", "Member"),
            "party": key_meps.get(mep, {}).get("party", ""),
            "total_amendments": mep_am_total.get(mep, 0),
            "total_meetings": mep_mtg_total.get(mep, 0),
            "amendment_themes": am_themes,
            "meeting_themes": mtg_themes,
            "overlapping_themes": overlapping,
            "theme_details": theme_details,
            "top_orgs_met": top_orgs,
        }
    return result


def _compute_lei(
    mep_crossref: dict[str, Any],
    org_influence: dict[str, Any],
    total_procedure_meetings: int,
) -> float:
    if total_procedure_meetings == 0:
        return 0.0
    top_orgs = mep_crossref.get("top_orgs_met", [])
    weighted_sum = sum(
        entry.get("count", 0)
        * _interest_weight(org_influence.get(entry.get("org", ""), {}).get("interests_represented", "Unknown"))
        for entry in top_orgs
    )
    return weighted_sum / total_procedure_meetings


def _compute_alas(mep_crossref: dict[str, Any]) -> float:
    total_am = mep_crossref.get("total_amendments", 0)
    total_mtg = mep_crossref.get("total_meetings", 0)
    am_themes = mep_crossref.get("amendment_themes", {})
    mtg_themes = mep_crossref.get("meeting_themes", {})
    overlapping = mep_crossref.get("overlapping_themes", [])
    if total_am == 0 or total_mtg == 0 or not overlapping:
        return 0.0
    raw_sum = sum(
        (am_themes.get(t, 0) / total_am) * (mtg_themes.get(t, 0) / total_mtg)
        for t in overlapping
    )
    return math.sqrt(min(1.0, raw_sum))


def _compute_ici(mep_crossref: dict[str, Any]) -> float:
    top_orgs = mep_crossref.get("top_orgs_met", [])
    total_mtg = mep_crossref.get("total_meetings", 0)
    if total_mtg == 0 or not top_orgs:
        return 0.0
    top_total = sum(e.get("count", 0) for e in top_orgs)
    remainder = total_mtg - top_total
    hhi = sum((e.get("count", 0) / total_mtg) ** 2 for e in top_orgs)
    for _ in range(remainder):
        hhi += (1 / total_mtg) ** 2
    return hhi


def _build_org_influence(
    commission_meetings: list[dict[str, Any]],
    lobbying_meetings: list[dict[str, Any]],
    compiled_patterns: dict[str, list[re.Pattern]],
) -> dict[str, Any]:
    org_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"meetings_count": 0, "themes_lobbied": set(), "interests_represented": "Unknown"}
    )
    for m in commission_meetings:
        text = _meeting_text(m)
        themes = _classify_by_regex(text, compiled_patterns)
        for org_info in (m.get("resolved_orgs") or []):
            name = org_info.get("name") or "Unknown"
            org_data[name]["meetings_count"] += 1
            org_data[name]["themes_lobbied"].update(themes)
            ir = org_info.get("interests_represented") or ""
            if ir and ir != "Unknown":
                org_data[name]["interests_represented"] = ir
        if not m.get("resolved_orgs"):
            raw = (m.get("organizations_raw") or "").strip() or "Unknown"
            name = raw.split("|")[0].strip() if "|" in raw else raw
            org_data[name]["meetings_count"] += 1
            org_data[name]["themes_lobbied"].update(themes)

    for m in lobbying_meetings:
        name = m.get("org_name") or "Unknown"
        text = _meeting_text(m)
        themes = _classify_by_regex(text, compiled_patterns)
        org_data[name]["meetings_count"] += 1
        org_data[name]["themes_lobbied"].update(themes)
        ir = m.get("interests_represented") or ""
        if ir and ir != "Unknown":
            org_data[name]["interests_represented"] = ir

    return {
        name: {
            "meetings_count": d["meetings_count"],
            "themes_lobbied": sorted(d["themes_lobbied"]),
            "interests_represented": d["interests_represented"],
        }
        for name, d in sorted(
            org_data.items(), key=lambda x: x[1]["meetings_count"], reverse=True
        )
    }


def _build_theme_indicators(
    amendments: list[dict[str, Any]],
    commission_meetings: list[dict[str, Any]],
    lobbying_meetings: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    compiled_patterns: dict[str, list[re.Pattern]],
) -> dict[str, Any]:
    indicators: dict[str, Any] = {}
    for theme_key, cfg in taxonomy.items():
        theme_amendments = [a for a in amendments if theme_key in (a.get("themes") or [])]

        comm_entries: list[dict[str, Any]] = []
        for m in commission_meetings:
            text = _meeting_text(m)
            pats = compiled_patterns.get(theme_key, [])
            if not any(p.search(text) for p in pats):
                continue
            resolved = m.get("resolved_orgs") or []
            orgs = [o["name"] for o in resolved] if resolved else [
                (m.get("organizations_raw") or "Unknown").split("|")[0].strip()
            ]
            comm_entries.append(
                {
                    "date": str(m.get("meeting_date", ""))[:10],
                    "orgs": orgs,
                    "subject_preview": (m.get("subject") or "")[:120],
                }
            )

        ep_entries: list[dict[str, Any]] = []
        for m in lobbying_meetings:
            text = _meeting_text(m)
            pats = compiled_patterns.get(theme_key, [])
            if not any(p.search(text) for p in pats):
                continue
            ep_entries.append(
                {
                    "date": str(m.get("meeting_date", ""))[:10],
                    "mep": m.get("mep_name", ""),
                    "org": m.get("org_name", ""),
                }
            )

        all_orgs: set[str] = set()
        for entry in comm_entries:
            all_orgs.update(o for o in entry.get("orgs", []) if o)
        for entry in ep_entries:
            if entry.get("org"):
                all_orgs.add(entry["org"])

        indicators[theme_key] = {
            "description": cfg.get("description", ""),
            "amendment_count": len(theme_amendments),
            "commission_meeting_count": len(comm_entries),
            "ep_meeting_count": len(ep_entries),
            "total_meeting_count": len(comm_entries) + len(ep_entries),
            "active_orgs": sorted(all_orgs),
            "active_org_count": len(all_orgs),
        }

    return indicators


def _run_statistical_tests(
    rows: list[dict[str, Any]],
    crossref_all: dict[str, Any],
    theme_indicators: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if not SCIPY_AVAILABLE:
        results["error"] = "scipy not available"
        return results
    if len(rows) < 4:
        results["error"] = "Insufficient MEP count for meaningful statistical tests"
        return results

    lei_vals = [r["lei"] for r in rows]
    amd_vals = [r["amendments"] for r in rows]
    alas_vals = [r["alas"] for r in rows]
    overlap_vals = [r["overlapping_themes"] for r in rows]

    commercial_themes = {
        t for t, ind in theme_indicators.items() if ind.get("total_meeting_count", 0) > 0
    }
    comm_amd_vals = [
        sum(
            v
            for k, v in crossref_all.get(r["mep"], {}).get("amendment_themes", {}).items()
            if k in commercial_themes
        )
        for r in rows
    ]

    if len(set(lei_vals)) > 1 and len(set(amd_vals)) > 1:
        r1, p1 = scipy_stats.pearsonr(lei_vals, amd_vals)
        results["test1_lei_vs_amendments"] = {
            "test": "Pearson correlation",
            "variables": "LEI vs. total_amendments",
            "n": len(rows),
            "statistic": round(float(r1), 4),
            "p_value": round(float(p1), 4),
            "interpretation": (
                "Significant positive correlation: higher-exposed MEPs table more amendments."
                if p1 < 0.05 and r1 > 0
                else "No statistically significant relationship."
            ),
        }
    else:
        results["test1_lei_vs_amendments"] = {"error": "Insufficient variance"}

    if len(set(alas_vals)) > 1 and len(set(comm_amd_vals)) > 1:
        r2, p2 = scipy_stats.pearsonr(alas_vals, comm_amd_vals)
        results["test2_alas_vs_commercial_amendments"] = {
            "test": "Pearson correlation",
            "variables": "ALAS vs. amendments_on_commercial_themes",
            "n": len(rows),
            "statistic": round(float(r2), 4),
            "p_value": round(float(p2), 4),
            "interpretation": (
                "Significant positive correlation."
                if p2 < 0.05 and r2 > 0
                else "No statistically significant relationship."
            ),
        }
    else:
        results["test2_alas_vs_commercial_amendments"] = {"error": "Insufficient variance"}

    median_lei = sorted(lei_vals)[len(lei_vals) // 2]
    median_overlap = sorted(overlap_vals)[len(overlap_vals) // 2]
    a = b = c = d = 0
    for r in rows:
        high_lei = r["lei"] > median_lei
        high_ov = r["overlapping_themes"] > median_overlap
        if high_lei and high_ov:
            a += 1
        elif high_lei and not high_ov:
            b += 1
        elif not high_lei and high_ov:
            c += 1
        else:
            d += 1
    try:
        odds_ratio, p3 = scipy_stats.fisher_exact([[a, b], [c, d]])
        results["test3_lei_group_vs_overlap_group"] = {
            "test": "Fisher's exact test",
            "variables": "LEI group (high/low) x thematic overlap group (high/low)",
            "contingency": {"a": a, "b": b, "c": c, "d": d},
            "odds_ratio": round(float(odds_ratio), 4),
            "p_value": round(float(p3), 4),
            "interpretation": (
                "Significant: higher-exposure MEPs have broader thematic overlap."
                if p3 < 0.05
                else "No statistically significant association."
            ),
        }
    except Exception as exc:
        results["test3_lei_group_vs_overlap_group"] = {"error": str(exc)}

    return results


def step6_quantitative_analysis(
    data: dict[str, Any],
    amendments: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    compiled_patterns: dict[str, list[re.Pattern]],
    logger: Any = None,
) -> dict[str, Any]:
    """Compute LEI, ALAS, ICI, theme indicators, and statistical tests."""
    _log = logger.info if logger else print
    _warn = logger.warning if logger else print

    procedure = data["procedure"]
    lobbying_meetings = data["lobbying"]
    commission_meetings = data["commission"]

    key_meps = _extract_rapporteurs(procedure)
    if key_meps:
        _log(f"Key MEPs from procedure actors: {len(key_meps)}")
        for name, meta in key_meps.items():
            _log(f"  {meta['role']:10s} {name} ({meta['party']})")
    else:
        _warn("No rapporteur/shadow data in procedure actors field.")

    total_procedure_meetings = len(lobbying_meetings)
    org_influence = _build_org_influence(commission_meetings, lobbying_meetings, compiled_patterns)
    mep_crossref = _build_mep_crossref(
        amendments, lobbying_meetings, commission_meetings, key_meps, compiled_patterns
    )

    mep_indices: dict[str, dict[str, float]] = {}
    for mep, crossref in mep_crossref.items():
        mep_indices[mep] = {
            "lei": _compute_lei(crossref, org_influence, total_procedure_meetings),
            "alas": _compute_alas(crossref),
            "ici": _compute_ici(crossref),
        }

    theme_indicators = _build_theme_indicators(
        amendments, commission_meetings, lobbying_meetings, taxonomy, compiled_patterns
    )

    comparison_rows: list[dict[str, Any]] = []
    for mep_name, meta in {**key_meps, **{m: {} for m in mep_crossref}}.items():
        crossref = mep_crossref.get(mep_name, {})
        indices = mep_indices.get(mep_name, {"lei": 0.0, "alas": 0.0, "ici": 0.0})
        am_themes = crossref.get("amendment_themes", {})
        top_themes = [t for t, _ in sorted(am_themes.items(), key=lambda x: x[1], reverse=True)[:3]]
        comparison_rows.append(
            {
                "mep": mep_name,
                "party": meta.get("party", crossref.get("party", "")),
                "role": meta.get("role", crossref.get("role", "Member")),
                "amendments": crossref.get("total_amendments", 0),
                "meetings": crossref.get("total_meetings", 0),
                "overlapping_themes": len(crossref.get("overlapping_themes", [])),
                "lei": round(indices["lei"], 4),
                "alas": round(indices["alas"], 4),
                "ici": round(indices["ici"], 4),
                "top_themes": top_themes,
                "top_orgs": [e.get("org", "") for e in crossref.get("top_orgs_met", [])[:3]],
            }
        )

    seen_meps: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in comparison_rows:
        if row["mep"] not in seen_meps:
            seen_meps.add(row["mep"])
            unique_rows.append(row)
    unique_rows.sort(key=lambda r: r["lei"], reverse=True)

    stat_tests = _run_statistical_tests(unique_rows, mep_crossref, theme_indicators)

    _log(f"MEPs analysed: {len(unique_rows)}")
    _log(f"Top MEP by LEI: {unique_rows[0]['mep'] if unique_rows else 'N/A'}")

    return {
        "key_meps": key_meps,
        "mep_crossref": mep_crossref,
        "mep_indices": mep_indices,
        "org_influence": org_influence,
        "theme_indicators": theme_indicators,
        "comparison_rows": unique_rows,
        "statistical_tests": stat_tests,
        "total_procedure_meetings": total_procedure_meetings,
    }


# ---------------------------------------------------------------------------
# STEP 7: Directional Alignment (AI-Assisted)
# ---------------------------------------------------------------------------


def _amendment_mentions_mep(amendment: dict[str, Any], mep_name: str) -> bool:
    authors = amendment.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    name_low = mep_name.lower()
    parts = name_low.split()
    surname = parts[-1] if parts else ""
    for author in authors:
        author_low = author.lower()
        if surname and surname in author_low:
            return True
        if name_low in author_low:
            return True
    return not authors


def step7_directional_alignment(
    quant: dict[str, Any],
    positions: list[dict[str, Any]],
    amendments: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    no_ai: bool = False,
    logger: Any = None,
) -> dict[str, Any]:
    """Score directional alignment between org positions and MEP amendments."""
    _log = logger.info if logger else print

    mep_crossref = quant.get("mep_crossref", {})
    ranked = sorted(
        mep_crossref.items(), key=lambda x: x[1].get("total_meetings", 0), reverse=True
    )
    top_meps = ranked[:10]

    if not top_meps:
        _log("No MEP data available.")
        return {}

    positions_by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pos in positions:
        for t in pos.get("themes", []):
            positions_by_theme[t].append(pos)

    mep_alignment: dict[str, Any] = {}
    all_tasks: list[tuple[str, str, list[dict], str]] = []

    for mep_name, crossref in top_meps:
        overlapping_themes = crossref.get("overlapping_themes", [])
        if not overlapping_themes:
            mep_alignment[mep_name] = {
                "total_pairs": 0, "toward": 0, "away": 0, "neutral": 0,
                "alignment_fraction": None, "theme_scores": {},
            }
            continue

        if no_ai or AI_PROVIDER is None:
            mep_alignment[mep_name] = {
                "total_pairs": 0, "toward": 0, "away": 0, "neutral": 0,
                "alignment_fraction": None,
                "theme_scores": {t: {"skipped": "no_ai_provider"} for t in overlapping_themes},
            }
            continue

        for theme in overlapping_themes:
            theme_positions = positions_by_theme.get(theme, [])
            theme_amendments = [
                a for a in amendments
                if theme in (a.get("themes") or []) and _amendment_mentions_mep(a, mep_name)
            ]
            if not theme_positions or not theme_amendments:
                continue

            pairs: list[dict[str, str]] = []
            for pos in theme_positions[:5]:
                for am in theme_amendments[:5]:
                    pairs.append(
                        {
                            "position_org": (pos.get("orgs") or ["Unknown"])[0],
                            "position_summary": pos.get("summary", "")[:300],
                            "amendment_number": str(am.get("number", "")),
                            "amendment_excerpt": (am.get("body") or "")[:400],
                        }
                    )
            if not pairs:
                continue

            prompt = f"""Assess whether each amendment moves regulation TOWARD or AWAY FROM the organisation's stated position.

THEME: {theme}
TAXONOMY CONTEXT: {taxonomy.get(theme, {}).get("description", "")}

PAIRS TO ASSESS:
{json.dumps(pairs, indent=2)}

For each pair, return a JSON array where each entry has:
  "position_org": organisation name (unchanged from input)
  "amendment_number": amendment number (unchanged from input)
  "score": 1 (amendment moves TOWARD the org's position), 0 (neutral/unclear), or -1 (AWAY from the org's position)
  "reasoning": one short sentence explaining the score

Respond ONLY with the JSON array."""
            all_tasks.append((mep_name, theme, pairs, prompt))

    if all_tasks:
        _log(f"Firing {len(all_tasks)} alignment prompts in parallel ({AI_MAX_WORKERS} workers) ...")
        prompts_only = [t[3] for t in all_tasks]
        raw_responses = ai_complete_parallel(
            prompts_only, json_mode=True, label="alignment", logger=logger
        )

        mep_scores: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"theme_scores": {}, "total_toward": 0, "total_away": 0, "total_neutral": 0}
        )

        for (mep_name, theme, pairs, _prompt), raw in zip(all_tasks, raw_responses):
            scored_pairs = parse_json_response(raw) if raw else None
            t_toward = t_away = t_neutral = 0
            if scored_pairs and isinstance(scored_pairs, list):
                for entry in scored_pairs:
                    score = entry.get("score", 0)
                    if score == 1:
                        t_toward += 1
                    elif score == -1:
                        t_away += 1
                    else:
                        t_neutral += 1
            else:
                t_neutral = len(pairs)

            mep_scores[mep_name]["total_toward"] += t_toward
            mep_scores[mep_name]["total_away"] += t_away
            mep_scores[mep_name]["total_neutral"] += t_neutral
            mep_scores[mep_name]["theme_scores"][theme] = {
                "pairs_evaluated": len(pairs),
                "toward": t_toward,
                "away": t_away,
                "neutral": t_neutral,
                "pair_details": scored_pairs or [],
            }

        for mep_name, scores_data in mep_scores.items():
            total_toward = scores_data["total_toward"]
            total_away = scores_data["total_away"]
            total_neutral = scores_data["total_neutral"]
            total_pairs = total_toward + total_away + total_neutral
            alignment_fraction = total_toward / total_pairs if total_pairs > 0 else None

            mep_alignment[mep_name] = {
                "total_pairs": total_pairs,
                "toward": total_toward,
                "away": total_away,
                "neutral": total_neutral,
                "alignment_fraction": (
                    round(alignment_fraction, 3) if alignment_fraction is not None else None
                ),
                "theme_scores": scores_data["theme_scores"],
            }
            _log(
                f"  {mep_name}: {total_pairs} pairs | "
                f"toward={total_toward} away={total_away} neutral={total_neutral}"
            )

    for mep_name, _ in top_meps:
        if mep_name not in mep_alignment:
            mep_alignment[mep_name] = {
                "total_pairs": 0, "toward": 0, "away": 0, "neutral": 0,
                "alignment_fraction": None, "theme_scores": {},
            }

    return mep_alignment


# ---------------------------------------------------------------------------
# STEP 8: Report Generation (Deterministic)
# ---------------------------------------------------------------------------


def step8_generate_report(
    procedure_id: str,
    data: dict[str, Any],
    taxonomy: dict[str, Any],
    amendments: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    quant: dict[str, Any],
    alignment: dict[str, Any],
    output_dir: Path | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Assemble the full structured report and write to disk.

    Parameters
    ----------
    output_dir:
        Directory to write the JSON report. Defaults to ``SCRIPTS_DIR``.
    """
    _log = logger.info if logger else print

    procedure = data["procedure"]
    lobbying_meetings = data["lobbying"]
    commission_meetings = data["commission"]

    with_points = sum(1 for m in commission_meetings if m.get("points_raised"))
    source_counts: dict[str, int] = Counter(a.get("source", "unknown") for a in amendments)

    report: dict[str, Any] = {
        "procedure": procedure_id,
        "title": procedure.get("title", ""),
        "analysis_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ai_provider": AI_PROVIDER,
        "summary_stats": {
            "total_amendments_parsed": len(amendments),
            "amendments_by_source": dict(source_counts),
            "total_lobbying_meetings": len(lobbying_meetings),
            "total_commission_meetings": len(commission_meetings),
            "commission_meetings_with_notes": with_points,
            "total_organisations": len(quant.get("org_influence", {})),
            "themes_with_lobbying_activity": sum(
                1
                for ind in quant.get("theme_indicators", {}).values()
                if ind.get("total_meeting_count", 0) > 0
            ),
        },
        "taxonomy": taxonomy,
        "theme_indicators": quant.get("theme_indicators", {}),
        "org_influence": quant.get("org_influence", {}),
        "mep_exposure": {
            mep: {
                "total_meetings": cr.get("total_meetings", 0),
                "total_amendments": cr.get("total_amendments", 0),
                "top_orgs": cr.get("top_orgs_met", [])[:5],
            }
            for mep, cr in quant.get("mep_crossref", {}).items()
        },
        "mep_amendment_crossref": quant.get("mep_crossref", {}),
        "mep_indices": quant.get("mep_indices", {}),
        "comparison_table": quant.get("comparison_rows", []),
        "statistical_tests": quant.get("statistical_tests", {}),
        "positions": positions,
        "directional_alignment": alignment,
    }

    # Write to analysis/{procedure_id}/influence_report.json
    # Replace / with : for macOS-safe directory names (Finder shows : as /)
    pid_dir_name = procedure_id.replace("/", ":")
    proc_dir = (output_dir or ANALYSIS_OUTPUT_DIR) / pid_dir_name
    proc_dir.mkdir(parents=True, exist_ok=True)
    output_path = proc_dir / "influence_report.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    _log(f"JSON report written to: {output_path}")

    # Generate one-pager (md + pdf) if AI is available
    if AI_PROVIDER is not None:
        try:
            _generate_one_pager(report, proc_dir, logger=logger)
        except Exception as exc:
            _log(f"One-pager generation failed (non-fatal): {exc}")

    return report


def _generate_one_pager(
    report: dict[str, Any],
    proc_dir: Path,
    logger: Any = None,
) -> None:
    """Generate a think-tank style one-pager from the report JSON."""
    _log = logger.info if logger else print

    prompt_path = Path(__file__).parent / "one_pager_prompt.md"
    if not prompt_path.exists():
        _log("one_pager_prompt.md not found, skipping one-pager generation")
        return

    prompt_template = prompt_path.read_text(encoding="utf-8")

    # Trim report for token efficiency
    report_trimmed = {k: v for k, v in report.items() if k != "org_influence"}
    if "org_influence" in report:
        top_orgs = dict(
            sorted(
                report["org_influence"].items(),
                key=lambda x: x[1].get("meetings_count", 0),
                reverse=True,
            )[:30]
        )
        report_trimmed["org_influence_top30"] = top_orgs
    if "directional_alignment" in report_trimmed:
        for mep, data in report_trimmed["directional_alignment"].items():
            if "theme_scores" in data:
                for theme, scores in data["theme_scores"].items():
                    if "pair_details" in scores:
                        scores["pair_details"] = scores["pair_details"][:5]

    report_json = json.dumps(report_trimmed, indent=2, ensure_ascii=False, default=str)
    user_prompt = prompt_template.split("```json")[0] + "```json\n" + report_json + "\n```"

    system = (
        "You are a policy analyst at a European transparency think tank. "
        "You write concise, evidence-based briefings about lobbying influence on EU legislation. "
        "Your tone is factual and measured — you present data patterns without sensationalising them."
    )

    _log("Generating one-pager via AI ...")
    md = ai_complete(f"{system}\n\n{user_prompt}")

    # Strip markdown code fences if the model wrapped it
    if md.startswith("```markdown"):
        md = md[len("```markdown"):].strip()
    if md.startswith("```"):
        md = md[3:].strip()
    if md.endswith("```"):
        md = md[:-3].strip()

    if not md or len(md) < 200:
        _log("One-pager generation returned insufficient content, skipping")
        return

    md_path = proc_dir / "one_pager.md"
    md_path.write_text(md, encoding="utf-8")
    _log(f"One-pager markdown written to: {md_path}")

    # Convert to PDF via pandoc if available
    import subprocess
    import shutil

    pandoc = shutil.which("pandoc")
    if not pandoc:
        _log("pandoc not found, skipping PDF generation")
        return

    pdflatex = shutil.which("pdflatex") or "/Library/TeX/texbin/pdflatex"
    pdf_path = proc_dir / "one_pager.pdf"
    try:
        subprocess.run(
            [
                pandoc, str(md_path), "-o", str(pdf_path),
                f"--pdf-engine={pdflatex}",
                "-V", "geometry:margin=1in",
                "-V", "fontsize=11pt",
            ],
            capture_output=True, text=True, check=True, timeout=30,
        )
        _log(f"One-pager PDF written to: {pdf_path}")
    except Exception as exc:
        _log(f"PDF generation failed: {exc}")


# ---------------------------------------------------------------------------
# High-level runner (called by Dagster asset)
# ---------------------------------------------------------------------------


def run_influence_pipeline(
    procedure_id: str,
    client: Any,
    no_ai: bool = False,
    regen_taxonomy: bool = False,
    output_dir: Path | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Run all 8 pipeline steps and return the report dict.

    This is the single entry-point called from the Dagster asset. It mirrors
    the ``main()`` function in the standalone script but accepts an injected
    Supabase client instead of creating one from environment variables.

    Parameters
    ----------
    procedure_id:
        EU procedure reference, e.g. ``2023/0212(COD)``.
    client:
        Raw Supabase client (from ``SupabaseResource.get_client()``).
    no_ai:
        When True, all AI calls are skipped (regex-only mode).
    regen_taxonomy:
        When True, the cached taxonomy file is deleted and regenerated.
    output_dir:
        Directory for the JSON report. Defaults to ``SCRIPTS_DIR``.
    logger:
        Optional logger (e.g. ``context.log`` from Dagster).
    """
    _log = logger.info if logger else print

    _log(f"Starting influence pipeline for {procedure_id}")
    _log(f"Mode: {'regex-only (no_ai=True)' if no_ai else 'AI-assisted'}")

    if not no_ai:
        configure_ai_provider()
        _log(f"AI provider: {AI_PROVIDER or 'None (regex-only fallback)'}")
    else:
        _log("AI disabled (no_ai=True).")

    from concurrent.futures import ThreadPoolExecutor, Future

    # Step 1 — collect all data from Supabase
    data = step1_collect_data(procedure_id, client, logger=logger)

    # Steps 2 + 3 in parallel (taxonomy generation + amendment fetching)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_taxonomy: Future = pool.submit(
            step2_generate_taxonomy,
            procedure_id, data, no_ai, regen_taxonomy, logger,
        )
        future_amendments: Future = pool.submit(
            step3_parse_amendments,
            procedure_id, client, logger,
        )
        taxonomy = future_taxonomy.result()
        amendments = future_amendments.result()

    compiled_patterns = compile_taxonomy_patterns(taxonomy)

    # Steps 4 + 5 in parallel (classify amendments + extract positions)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_classified: Future = pool.submit(
            step4_classify_amendments,
            amendments, taxonomy, no_ai, logger,
        )
        future_positions: Future = pool.submit(
            step5_extract_positions,
            data["commission"], taxonomy, compiled_patterns, no_ai, logger,
        )
        amendments = future_classified.result()
        positions = future_positions.result()

    # Step 6 — quantitative analysis (needs classified amendments + positions)
    quant = step6_quantitative_analysis(
        data, amendments, taxonomy, compiled_patterns, logger=logger
    )

    # Step 7 — directional alignment (needs quant + positions + amendments)
    alignment = step7_directional_alignment(
        quant, positions, amendments, taxonomy, no_ai=no_ai, logger=logger
    )

    # Step 8
    report = step8_generate_report(
        procedure_id, data, taxonomy, amendments, positions, quant, alignment,
        output_dir=output_dir, logger=logger,
    )

    _log(f"Influence pipeline complete for {procedure_id}")
    return report
