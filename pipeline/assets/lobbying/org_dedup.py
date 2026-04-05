"""Organisation deduplication: resolve stub orgs to canonical TR entries.

Four-pass strategy:
1. TR ID extraction — stub names containing a Transparency Register ID
   (e.g. "Bundesverband deutscher Banken e.V. (Bankenverband) 0764199368-97")
   are matched deterministically to the canonical org with that TR ID.
2. Case-insensitive name matching — exact match after lowercasing and
   stripping legal suffixes (Ltd, GmbH, SA, etc.).
3. Acronym matching — stub name matches a canonical org's acronym field.
4. EU Transparency Register web search — remaining unmatched stubs are
   looked up via the TR public search API; an AI model (haiku) confirms
   whether each result is a genuine match.  High-confidence matches are
   applied directly; medium-confidence matches are flagged for review in
   analysis/org_dedup_report.csv.

Passes 1-3 are fully deterministic and never produce false positives.
Pass 4 uses AI confirmation to guard against false positives from the
looser string matching inherent in a web search.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Legal suffixes to strip for fuzzy matching
_LEGAL_SUFFIXES = re.compile(
    r"\s+(ltd|gmbh|sa|ag|bv|nv|plc|inc|e\.v\.|aisbl|asbl|eeig|se|s\.a\.|s\.p\.a\.|s\.r\.l\.)\.?\s*$",
    re.IGNORECASE,
)

# TR ID pattern: 10+ digits followed by dash and 2 digits
_TR_ID_RE = re.compile(r"(\d{10,}-\d{2})")

# Output directory for pass-4 reports (four levels up from this file)
_ANALYSIS_DIR = Path(__file__).parent.parent.parent.parent / "analysis"

# TR search / detail URLs
_TR_SEARCH_URL = "https://ec.europa.eu/transparencyregister/public/search?lang=en&queryText={query}"
_TR_DETAIL_URL = "https://transparency-register.europa.eu/search-register-or-update/organisation-detail_en?id={tr_id}"


# ---------------------------------------------------------------------------
# Pass 1-3 helpers
# ---------------------------------------------------------------------------

def _clean_name(name: str) -> str:
    """Lowercase, strip legal suffixes and parenthetical acronyms."""
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", name).strip().lower()
    cleaned = _LEGAL_SUFFIXES.sub("", cleaned).strip()
    return cleaned


# Geographic / office suffixes commonly appended to org names in meeting data
_GEO_SUFFIXES = re.compile(
    r"\s+(?:"
    r"belgium|brussels|eu\s*office|europe|european\s*office|"
    r"france|germany|ireland|netherlands|italia|spain|"
    r"uk|united\s*kingdom|denmark|sweden|finland|austria|"
    r"portugal|greece|poland|czech\s*republic|hungary|romania|"
    r"croatia|slovakia|slovenia|bulgaria|cyprus|estonia|"
    r"latvia|lithuania|luxembourg|malta|"
    r"bureau\s*europ[ée]en|repr[ée]sentation|"
    r"eu\s*representation|eu\s*affairs|public\s*affairs"
    r")\s*$",
    re.IGNORECASE,
)


def _search_variants(name: str) -> list[str]:
    """Generate search query variants for a stub org name.

    Returns a list of up to 3 queries to try, in priority order:
    1. Original name
    2. Name with geographic/office suffixes stripped
    3. Name with parenthetical content removed (if different from #2)
    """
    variants: list[str] = [name]
    # Strip geographic suffixes
    stripped = _GEO_SUFFIXES.sub("", name).strip()
    if stripped and stripped.lower() != name.lower():
        variants.append(stripped)
    # Strip parenthetical content
    no_parens = re.sub(r"\s*\(.*?\)\s*", " ", name).strip()
    no_parens = _GEO_SUFFIXES.sub("", no_parens).strip()
    if no_parens and no_parens.lower() not in {v.lower() for v in variants}:
        variants.append(no_parens)
    return variants


# ---------------------------------------------------------------------------
# Pass 4 helpers
# ---------------------------------------------------------------------------

def _scrape_tr_search(query: str) -> list[dict]:
    """Fetch the TR public search page and return up to 5 result entries.

    Each entry is a dict with keys ``name`` and ``tr_id``.
    Returns an empty list on any error so callers can treat it as a safe
    no-op.
    """
    try:
        import requests  # optional dependency — checked at call time
    except ImportError:
        return []

    time.sleep(0.05)
    url = _TR_SEARCH_URL.format(query=requests.utils.quote(query))
    try:
        resp = requests.get(url, timeout=15, headers={"Accept-Language": "en"})
        resp.raise_for_status()
    except Exception:
        return []

    html = resp.text
    # Links look like:  href="search-details_en?id=3978240953-79"
    # The org name is in a span immediately inside (or near) that link.
    # We extract (id, name) pairs with a two-step regex approach.
    entries: list[dict] = []
    # Find all anchor tags that point to search-details_en
    for link_match in re.finditer(
        r'href=["\']search-details_en\?id=([0-9]+-[0-9]+)["\'][^>]*>(.*?)</a>',
        html,
        re.DOTALL | re.IGNORECASE,
    ):
        tr_id = link_match.group(1).strip()
        inner_html = link_match.group(2)
        # Strip any remaining HTML tags to get plain text name
        name = re.sub(r"<[^>]+>", "", inner_html).strip()
        # Collapse whitespace
        name = re.sub(r"\s+", " ", name).strip()
        if tr_id and name:
            entries.append({"name": name, "tr_id": tr_id})
        if len(entries) >= 5:
            break

    return entries


def _scrape_tr_detail(tr_id: str) -> dict | None:
    """Fetch the TR detail page for *tr_id* and extract metadata.

    Uses the inner EC endpoint which returns the actual org data HTML
    (the outer Drupal page at transparency-register.europa.eu is just a shell
    that loads content dynamically from this endpoint).

    Returns a dict with keys: ``name``, ``acronym``, ``interests_represented``,
    ``category``, ``country``, ``website``.
    Returns ``None`` on any failure.
    """
    try:
        import requests
    except ImportError:
        return None

    time.sleep(0.05)
    # The inner endpoint that serves the actual org data
    url = f"https://ec.europa.eu/transparencyregister/public/PUBLIC/ORGANISATION/{tr_id}?lang=en"
    try:
        resp = requests.get(url, timeout=15, headers={"Accept-Language": "en"})
        resp.raise_for_status()
    except Exception:
        return None

    html = resp.text

    def _cell_after_label(label: str) -> str:
        """Return the text of the <td> that follows a <td> containing *label*.

        The actual HTML wraps labels in <strong> tags:
          <td ...><strong>Label</strong>:</td>
          <td ...><strong>Value</strong></td>
        """
        pattern = re.compile(
            r'<td[^>]*>[^<]*<strong>\s*'
            + re.escape(label)
            + r'\s*</strong>[^<]*</td>\s*<td[^>]*>(.*?)</td>',
            re.DOTALL | re.IGNORECASE,
        )
        m = pattern.search(html)
        if not m:
            return ""
        raw = m.group(1)
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Decode HTML entities so label matching works (&#39; -> ')
    import html as html_mod
    html = html_mod.unescape(html)

    name = _cell_after_label("Organisation name")
    acronym = _cell_after_label("Acronym")
    website = _cell_after_label("Website")
    category = _cell_after_label("Category of registration")

    interests_represented = _cell_after_label("Applicant/registrant's representation")
    if not interests_represented:
        interests_represented = _cell_after_label("Interests represented")

    # Country: extract from the address block (all-caps country name)
    country = ""
    country_match = re.search(
        r'(?:GERMANY|BELGIUM|FRANCE|DENMARK|NETHERLANDS|ITALY|SPAIN|AUSTRIA|'
        r'SWEDEN|FINLAND|PORTUGAL|GREECE|POLAND|CZECH\s*REPUBLIC|HUNGARY|'
        r'ROMANIA|CROATIA|SLOVAKIA|SLOVENIA|BULGARIA|CYPRUS|ESTONIA|LATVIA|'
        r'LITHUANIA|LUXEMBOURG|MALTA|IRELAND|UNITED\s*KINGDOM|SWITZERLAND|'
        r'NORWAY|UNITED\s*STATES)',
        html,
    )
    if country_match:
        country = country_match.group(0).strip().title()

    if not name:
        return None

    # Treat "N/A" as empty string
    def _clean(v: str) -> str:
        return "" if v.strip().lower() in {"n/a", "n/a.", "-", "none"} else v.strip()

    return {
        "name": _clean(name),
        "acronym": _clean(acronym),
        "interests_represented": _clean(interests_represented),
        "category": _clean(category),
        "country": _clean(country),
        "website": _clean(website),
    }


def _ai_confirm_match(
    stub_name: str,
    tr_result: dict,
    meeting_context: str = "",
) -> dict:
    """Ask the Claude haiku model whether *stub_name* is the same org as *tr_result*.

    Returns a dict with keys ``match`` (one of ``"high"``, ``"medium"``,
    ``"low"``, ``"no_match"``) and ``reasoning`` (one sentence string).
    Returns ``{"match": "no_match", "reasoning": "parse_failed"}`` on any
    error so that callers can treat it as a safe no-match.
    """
    tr_name = tr_result.get("name", "")
    acronym = tr_result.get("acronym", "")
    country = tr_result.get("country", "")
    category = tr_result.get("category", "")
    interests = tr_result.get("interests_represented", "")

    context_line = f"Meeting context: {meeting_context}" if meeting_context else ""

    prompt = (
        f'Our database has an organization named "{stub_name}" that attended EU institutional meetings.\n'
        f"{context_line}\n\n"
        f"The EU Transparency Register search returned:\n"
        f'- Name: "{tr_name}"\n'
        f'- Acronym: "{acronym}"\n'
        f'- Country: "{country}"\n'
        f'- Category: "{category}"\n'
        f'- Interests: "{interests}"\n\n'
        "Are these the same organization? Consider name variants, acronyms, "
        "translations across EU languages, and abbreviations.\n\n"
        'Respond ONLY with JSON: {"match": "high"|"medium"|"low"|"no_match", "reasoning": "one sentence"}'
    )

    _FALLBACK = {"match": "no_match", "reasoning": "parse_failed"}

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30,
        )
        raw = result.stdout.strip()
        # Extract JSON even if the model wraps it in markdown fences
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return _FALLBACK
        parsed = json.loads(json_match.group(0))
        if parsed.get("match") not in {"high", "medium", "low", "no_match"}:
            return _FALLBACK
        return {
            "match": parsed["match"],
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception:
        return _FALLBACK


def run_tr_search_pass(
    client: Any,
    stubs_remaining: list[dict],
    logger: Any = None,
    dry_run: bool = True,
) -> dict:
    """Pass 4: look up remaining unmatched stubs on the EU Transparency Register.

    Parameters
    ----------
    client:
        Raw Supabase client.
    stubs_remaining:
        List of stub org dicts (each must have at least ``id`` and ``name``).
    logger:
        Optional logger with ``.info()`` / ``.warning()`` methods.
    dry_run:
        When ``True`` (default) no database writes are performed; results are
        written to ``analysis/org_dedup_report.csv`` for manual review.
        When ``False`` high-confidence matches are applied to the database.

    Returns
    -------
    Dict with counts: ``searched``, ``high``, ``medium``, ``skipped``,
    ``applied``, ``errors``.
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    stats: dict[str, int] = {
        "searched": 0,
        "high": 0,
        "medium": 0,
        "skipped": 0,
        "applied": 0,
        "errors": 0,
    }

    # Check requests is available before doing any work
    try:
        import requests as _req  # noqa: F401
    except ImportError:
        _err("Pass 4 skipped: 'requests' package is not installed.")
        return stats

    report_rows: list[dict] = []

    for stub in stubs_remaining:
        stub_id: str = stub["id"]
        stub_name: str = stub["name"].strip()

        # Step 1: search TR — try multiple query variants
        variants = _search_variants(stub_name)
        results: list[dict] = []
        for query in variants:
            results = _scrape_tr_search(query)
            if results:
                break
        stats["searched"] += 1

        if not results:
            stats["skipped"] += 1
            report_rows.append(
                {
                    "stub_id": stub_id,
                    "stub_name": stub_name,
                    "tr_id": "",
                    "tr_name": "",
                    "tr_acronym": "",
                    "confidence": "no_results",
                    "reasoning": f"No TR results for variants: {variants}",
                    "action": "skip",
                }
            )
            continue

        # Step 2: take top result and fetch detail page
        top = results[0]
        tr_id = top["tr_id"]
        detail = _scrape_tr_detail(tr_id)

        if detail is None:
            stats["skipped"] += 1
            report_rows.append(
                {
                    "stub_id": stub_id,
                    "stub_name": stub_name,
                    "tr_id": tr_id,
                    "tr_name": top["name"],
                    "tr_acronym": "",
                    "confidence": "detail_failed",
                    "reasoning": "Could not fetch TR detail page",
                    "action": "skip",
                }
            )
            continue

        # Step 3: AI confirmation
        ai = _ai_confirm_match(stub_name, detail)
        confidence = ai["match"]
        reasoning = ai["reasoning"]

        tr_name = detail.get("name", top["name"])
        tr_acronym = detail.get("acronym", "")

        if confidence == "high":
            stats["high"] += 1
            action = "apply" if not dry_run else "apply_dry"
        elif confidence == "medium":
            stats["medium"] += 1
            action = "review"
        else:
            stats["skipped"] += 1
            action = "skip"

        report_rows.append(
            {
                "stub_id": stub_id,
                "stub_name": stub_name,
                "tr_id": tr_id,
                "tr_name": tr_name,
                "tr_acronym": tr_acronym,
                "confidence": confidence,
                "reasoning": reasoning,
                "action": action,
            }
        )

        # Step 4: apply to database when not dry_run and high confidence
        if confidence == "high" and not dry_run:
            try:
                # Check if a canonical org with this TR ID already exists
                canonical_resp = (
                    client.table("organizations")
                    .select("id,name,eu_transparency_register_id,interests_represented")
                    .eq("eu_transparency_register_id", tr_id)
                    .execute()
                )
                canonical_orgs = canonical_resp.data or []

                if canonical_orgs:
                    # A canonical already exists — relink meetings from stub to canonical
                    canonical = canonical_orgs[0]
                    if canonical["id"] == stub_id:
                        # Stub IS the canonical, just enrich it
                        _apply_tr_enrichment(client, stub_id, tr_id, detail, _err)
                    else:
                        # Relink meetings then optionally delete stub
                        client.table("lobbying_meetings").update(
                            {"organization_id": canonical["id"]}
                        ).eq("organization_id", stub_id).execute()
                        stats["applied"] += 1
                        _log(
                            f"Pass 4 relinked '{stub_name}' -> canonical '{canonical['name']}' (TR {tr_id})"
                        )
                else:
                    # No canonical exists yet — enrich the stub org with TR data
                    _apply_tr_enrichment(client, stub_id, tr_id, detail, _err)
                    stats["applied"] += 1
                    _log(f"Pass 4 enriched stub '{stub_name}' with TR ID {tr_id}")

            except Exception as exc:
                _err(f"Pass 4 DB error for '{stub_name}': {exc}")
                stats["errors"] += 1

    # Write CSV report (always, even in non-dry-run, for auditability)
    if report_rows:
        _write_dedup_report(report_rows, _err)

    mode = "dry-run" if dry_run else "live"
    _log(
        f"Pass 4 ({mode}) complete: {stats['searched']} searched, "
        f"{stats['high']} high, {stats['medium']} medium, "
        f"{stats['skipped']} skipped, {stats['applied']} applied, "
        f"{stats['errors']} errors"
    )
    return stats


def _apply_tr_enrichment(
    client: Any,
    org_id: str,
    tr_id: str,
    detail: dict,
    err_fn: Any,
) -> None:
    """Update an org record with data from the TR detail page.

    Rules:
    - Always write ``eu_transparency_register_id`` if not already set.
    - Only write ``interests_represented`` when the current value is NULL or
      "Unknown".
    - Never overwrite ``website`` if one already exists.
    """
    try:
        current_resp = (
            client.table("organizations")
            .select("eu_transparency_register_id,interests_represented,website")
            .eq("id", org_id)
            .single()
            .execute()
        )
        current = current_resp.data or {}
    except Exception:
        current = {}

    updates: dict[str, str] = {}

    if not current.get("eu_transparency_register_id"):
        updates["eu_transparency_register_id"] = tr_id

    current_interests = (current.get("interests_represented") or "").strip()
    if not current_interests or current_interests.lower() == "unknown":
        new_interests = detail.get("interests_represented", "")
        if new_interests:
            updates["interests_represented"] = new_interests

    if not current.get("website") and detail.get("website"):
        updates["website"] = detail["website"]

    if detail.get("acronym") and not current.get("acronym"):
        updates["acronym"] = detail["acronym"]

    if detail.get("organization_type") and not current.get("organization_type"):
        updates["organization_type"] = detail.get("category", "")

    if detail.get("country") and not current.get("country"):
        updates["country"] = detail["country"]

    if updates:
        try:
            client.table("organizations").update(updates).eq("id", org_id).execute()
        except Exception as exc:
            err_fn(f"Failed to enrich org {org_id}: {exc}")


def _write_dedup_report(rows: list[dict], err_fn: Any) -> None:
    """Write *rows* to the analysis/org_dedup_report.csv file."""
    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _ANALYSIS_DIR / "org_dedup_report.csv"
    fieldnames = [
        "stub_id",
        "stub_name",
        "tr_id",
        "tr_name",
        "tr_acronym",
        "confidence",
        "reasoning",
        "action",
    ]
    try:
        with report_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        err_fn(f"Could not write dedup report to {report_path}: {exc}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_org_dedup(
    client: Any,
    logger: Any = None,
    dry_run: bool = True,
) -> dict[str, int]:
    """Run all four dedup passes against Supabase.

    Parameters
    ----------
    client:
        Raw Supabase client.
    logger:
        Optional logger with .info() / .warning() methods.
    dry_run:
        Passed through to pass 4.  When ``True`` (default) pass 4 writes a
        CSV report but makes no database changes.  Passes 1-3 are always
        applied regardless of this flag.

    Returns
    -------
    Dict with counts: tr_id_relinked, name_relinked, acronym_relinked,
    tr_search_high, tr_search_medium, tr_search_applied, total.
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    # Fetch canonical orgs (those with a TR ID)
    real_orgs: list[dict] = []
    offset = 0
    while True:
        resp = (
            client.table("organizations")
            .select("id,name,acronym,eu_transparency_register_id")
            .not_.is_("eu_transparency_register_id", "null")
            .range(offset, offset + 999)
            .execute()
        )
        real_orgs.extend(resp.data)
        if len(resp.data) < 1000:
            break
        offset += 1000
    _log(f"Canonical orgs (with TR ID): {len(real_orgs)}")

    # Build lookups
    by_tr_id: dict[str, dict] = {}
    by_lower: dict[str, dict] = {}
    by_cleaned: dict[str, dict] = {}
    by_acronym: dict[str, dict] = {}
    _acronym_seen: dict[str, int] = {}  # track ambiguous acronyms

    for o in real_orgs:
        tr_id = o.get("eu_transparency_register_id")
        if tr_id:
            by_tr_id[tr_id] = o
        name = o["name"].strip()
        by_lower[name.lower()] = o
        by_cleaned[_clean_name(name)] = o
        if o.get("acronym"):
            acr = o["acronym"].strip().lower()
            _acronym_seen[acr] = _acronym_seen.get(acr, 0) + 1
            by_acronym[acr] = o

    # Remove ambiguous acronyms (shared by multiple orgs)
    for acr, count in _acronym_seen.items():
        if count > 1:
            by_acronym.pop(acr, None)

    # Fetch stubs (no TR ID, no normalized_name)
    stubs: list[dict] = []
    offset = 0
    while True:
        resp = (
            client.table("organizations")
            .select("id,name")
            .is_("normalized_name", "null")
            .is_("eu_transparency_register_id", "null")
            .range(offset, offset + 999)
            .execute()
        )
        stubs.extend(resp.data)
        if len(resp.data) < 1000:
            break
        offset += 1000
    _log(f"Stub orgs to check: {len(stubs)}")

    stats: dict[str, int] = {
        "tr_id_relinked": 0,
        "name_relinked": 0,
        "acronym_relinked": 0,
    }

    unmatched_stubs: list[dict] = []

    for s in stubs:
        raw = s["name"].strip()
        match = None
        method = None

        # Pass 1: TR ID embedded in name
        tr_match = _TR_ID_RE.search(raw)
        if tr_match:
            candidate = by_tr_id.get(tr_match.group(1))
            if candidate and candidate["id"] != s["id"]:
                match = candidate
                method = "tr_id_relinked"

        # Pass 2: case-insensitive name / cleaned name
        if not match:
            candidate = by_lower.get(raw.lower()) or by_cleaned.get(_clean_name(raw))
            if candidate and candidate["id"] != s["id"]:
                match = candidate
                method = "name_relinked"

        # Pass 3: acronym (only for unambiguous acronyms with 5+ chars)
        if not match:
            acronym_lower = raw.strip().lower()
            if len(acronym_lower) >= 5 and acronym_lower in by_acronym:
                # Check it's unambiguous (only one real org has this acronym)
                candidate = by_acronym[acronym_lower]
                if candidate and candidate["id"] != s["id"]:
                    match = candidate
                    method = "acronym_relinked"

        if match:
            try:
                client.table("lobbying_meetings").update(
                    {"organization_id": match["id"]}
                ).eq("organization_id", s["id"]).execute()
                stats[method] += 1  # type: ignore[index]
            except Exception as exc:
                _err(f"Failed to relink {s['name']}: {exc}")
        else:
            unmatched_stubs.append(s)

    pass123_total = sum(stats.values())
    stats["total_pass123"] = pass123_total
    _log(
        f"Passes 1-3 complete: {stats['tr_id_relinked']} TR ID, "
        f"{stats['name_relinked']} name, {stats['acronym_relinked']} acronym "
        f"({pass123_total} total). {len(unmatched_stubs)} stubs remain for pass 4."
    )

    # Pass 4: EU Transparency Register web search
    p4_stats = run_tr_search_pass(client, unmatched_stubs, logger, dry_run=dry_run)
    stats["tr_search_high"] = p4_stats["high"]
    stats["tr_search_medium"] = p4_stats["medium"]
    stats["tr_search_applied"] = p4_stats["applied"]

    stats["total"] = pass123_total + p4_stats["applied"]
    _log(
        f"Org dedup complete: {stats['tr_id_relinked']} TR ID, "
        f"{stats['name_relinked']} name, {stats['acronym_relinked']} acronym, "
        f"{p4_stats['applied']} TR-search applied, {stats['total']} total relinks"
    )
    return stats
