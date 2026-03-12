"""Silver layer: Entity resolution for commission meetings.

Links organization names from scraped meetings to organizations in the database,
using transparency register IDs (from PDF) and fuzzy name matching.
"""

import re
from typing import Any, Optional


def normalize_org_name(name: str) -> str:
    """Normalize organization name for matching."""
    name = name.strip().lower()
    # Remove common legal suffixes
    name = re.sub(
        r"\b(aisbl|asbl|vzw|e\.v\.|gmbh|ltd|s\.a\.|inc|plc|"
        r"associação|association|a\.s\.b\.l\.)\b",
        "",
        name,
        flags=re.IGNORECASE,
    )
    # Remove punctuation and extra whitespace
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_organizations_from_raw(orgs_raw: str) -> list[str]:
    """Split raw organization text into individual org names.

    Handles multiple formats from the EC transparency initiative HTML:
    - Newline-separated orgs (clean format from separator="\n")
    - Abbreviations like (ABBREV) after org names
    - Concatenated names: (ABBREV)NextOrg from old get_text(strip=True)
    - Tab/whitespace mess from HTML
    """
    if not orgs_raw:
        return []

    # Step 1: Collapse tab sequences into newlines (HTML whitespace artifacts)
    text = re.sub(r"[\t ]{3,}", "\n", orgs_raw)

    # Step 2: Split (ABBREV) from the next org name when they're glued together
    # e.g., "(Bosch)Infineon Technologies AG" → "(Bosch)\nInfineon Technologies AG"
    # Pattern: closing paren followed immediately by an uppercase letter (no space)
    text = re.sub(r"\)([A-Z])", r")\n\1", text)

    # Step 3: Split on newlines and semicolons
    parts = re.split(r"\s*;\s*|\n", text)

    result = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 2:
            continue
        # Skip pure abbreviation entries like "(IIEA)", "(Bosch)", "(ST)"
        if re.match(r"^\([^)]+\)$", part):
            continue
        # Strip trailing abbreviation: "Org Name (ABBREV)" → "Org Name"
        clean = re.sub(r"\s*\([^)]{1,30}\)\s*$", "", part).strip()
        if clean and len(clean) > 1:
            result.append(clean)

    return result


def split_concatenated_names(
    names: list[str],
    known_names_lower: dict[str, str],
) -> list[str]:
    """Split concatenated org names using a dictionary of known names.

    Handles cases like 'BAE SystemsLeonardo S.p.A.GKN Aerospace' by finding
    known org names within the string via longest-match greedy search.

    Args:
        names: List of org name strings (some may be concatenated)
        known_names_lower: Dict mapping lowercase org name → original name

    Returns:
        List of individual org names (split where possible)
    """
    if not known_names_lower:
        return names

    # Sort known names by length descending for greedy longest-match
    sorted_known = sorted(known_names_lower.keys(), key=len, reverse=True)

    result = []
    for name in names:
        # Strip leading (ACRONYM) prefix before matching
        stripped = re.sub(r"^\([^)]+\)", "", name).strip() if name.startswith("(") else name
        name_lower = stripped.lower()

        # Quick check: if the name itself matches a known org, keep as-is
        if name_lower in known_names_lower:
            result.append(stripped)
            continue

        # Try to find known org names within the concatenated string
        found = []
        remaining = stripped
        remaining_lower = name_lower

        while remaining_lower:
            matched = False
            for known in sorted_known:
                if remaining_lower.startswith(known):
                    found.append(known_names_lower[known])
                    remaining = remaining[len(known):]
                    remaining_lower = remaining_lower[len(known):]
                    matched = True
                    break
            if not matched:
                # Skip one character and try again
                remaining = remaining[1:]
                remaining_lower = remaining_lower[1:]

        if len(found) > 1:
            result.extend(found)
        else:
            # Couldn't split — keep original
            result.append(name)

    return result


def match_by_tr_id(
    tr_ids: list[str], orgs_by_tr_id: dict[str, dict]
) -> list[dict]:
    """Match organizations by Transparency Register ID (highest confidence)."""
    matches = []
    for tr_id in tr_ids:
        if tr_id in orgs_by_tr_id:
            org = orgs_by_tr_id[tr_id]
            matches.append({
                "organization_id": org["id"],
                "organization_name": org.get("name", org.get("official_name", "")),
                "eu_transparency_register_id": tr_id,
                "match_method": "tr_id_exact",
            })
    return matches


def match_by_name(
    org_names: list[str],
    orgs_by_normalized_name: dict[str, dict],
    already_matched_ids: set[str],
) -> list[dict]:
    """Match organizations by normalized name (medium confidence)."""
    matches = []
    for name in org_names:
        normalized = normalize_org_name(name)
        if normalized in orgs_by_normalized_name:
            org = orgs_by_normalized_name[normalized]
            if org["id"] not in already_matched_ids:
                matches.append({
                    "organization_id": org["id"],
                    "organization_name": name,
                    "eu_transparency_register_id": org.get("eu_transparency_register_id"),
                    "match_method": "name_exact",
                })
                already_matched_ids.add(org["id"])
        else:
            # Unmatched — still record the org name for the junction table
            matches.append({
                "organization_id": None,
                "organization_name": name,
                "eu_transparency_register_id": None,
                "match_method": "unmatched",
            })
    return matches


def process_commission_meetings(
    bronze_data: list[dict],
    existing_orgs: list[dict],
    logger: Optional[Any] = None,
) -> dict[str, list[dict]]:
    """Process bronze meetings into silver format.

    Returns:
        {
            "meetings": list of meeting records ready for upload,
            "meeting_organizations": list of junction table records,
        }
    """
    # Build lookup indexes from existing organizations
    orgs_by_tr_id: dict[str, dict] = {}
    orgs_by_normalized_name: dict[str, dict] = {}
    # For dictionary-based splitting of concatenated names
    known_names_lower: dict[str, str] = {}

    for org in existing_orgs:
        tr_id = org.get("eu_transparency_register_id")
        if tr_id:
            orgs_by_tr_id[tr_id] = org
        name = org.get("normalized_name") or org.get("name", "")
        if name:
            orgs_by_normalized_name[normalize_org_name(name)] = org
            known_names_lower[name.lower()] = name

    if logger:
        logger.info(
            f"Org lookup: {len(orgs_by_tr_id)} by TR ID, "
            f"{len(orgs_by_normalized_name)} by name"
        )

    meetings = []
    meeting_orgs = []
    matched_by_tr = 0
    matched_by_name = 0
    unmatched_count = 0

    for raw in bronze_data:
        # Build the meeting record
        meeting = {
            "id": raw["id"],
            "actor_id": raw.get("actor_id"),
            "commissioner_name": raw["commissioner_name"],
            "commissioner_portfolio": raw.get("commissioner_portfolio"),
            "host_id": raw.get("host_id"),
            "meeting_type": raw.get("meeting_type", "commissioner"),
            "meeting_date": raw.get("meeting_date"),
            "location": raw.get("location"),
            "subject": raw.get("minutes_subject") or raw.get("subject"),
            "commission_representatives": raw.get("commission_representatives", []),
            "organizations_raw": raw.get("organizations_raw"),
            "transparency_register_ids": raw.get("transparency_register_ids", []),
            "points_raised": raw.get("points_raised"),
            "conclusions": raw.get("conclusions"),
            "ares_number": raw.get("ares_number"),
            "minutes_url": raw.get("minutes_url"),
            "source_url": raw.get("source_url"),
            "raw_data": raw,
        }
        meetings.append(meeting)

        # Resolve organizations
        already_matched = set()
        tr_ids = raw.get("transparency_register_ids", [])

        # First pass: match by TR ID (from PDF)
        tr_matches = match_by_tr_id(tr_ids, orgs_by_tr_id)
        for m in tr_matches:
            m["meeting_id"] = raw["id"]
            meeting_orgs.append(m)
            if m["organization_id"]:
                already_matched.add(m["organization_id"])
                matched_by_tr += 1

        # Second pass: match remaining org names
        # Prefer clean list from HTML parsing; fall back to raw text parsing
        org_names = raw.get("organizations", []) or parse_organizations_from_raw(raw.get("organizations_raw", ""))
        # Split any concatenated names using known org dictionary
        org_names = split_concatenated_names(org_names, known_names_lower)
        name_matches = match_by_name(org_names, orgs_by_normalized_name, already_matched)
        for m in name_matches:
            m["meeting_id"] = raw["id"]
            meeting_orgs.append(m)
            if m["match_method"] == "name_exact":
                matched_by_name += 1
            elif m["match_method"] == "unmatched":
                unmatched_count += 1

    if logger:
        logger.info(
            f"Entity resolution: {matched_by_tr} by TR ID, {matched_by_name} by name, "
            f"{unmatched_count} unmatched"
        )
        logger.info(f"Silver output: {len(meetings)} meetings, {len(meeting_orgs)} org links")

    return {
        "meetings": meetings,
        "meeting_organizations": meeting_orgs,
    }
