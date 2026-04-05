"""Step 6: Quantitative analysis — org influence, theme indicators, match density."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from ._helpers import _classify_by_regex, _meeting_text


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


def _compute_theme_lobbying_density(theme_indicators: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank themes by commission meeting count ('lobbying density')."""
    return sorted(
        [
            {
                "theme": t,
                "commission_meeting_count": ind["commission_meeting_count"],
                "total_meeting_count": ind["total_meeting_count"],
                "amendment_count": ind["amendment_count"],
                "active_org_count": ind["active_org_count"],
            }
            for t, ind in theme_indicators.items()
        ],
        key=lambda x: x["commission_meeting_count"],
        reverse=True,
    )


def _compute_amendment_lobbying_density(
    amendments: list[dict[str, Any]],
    positions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """For each amendment, count how many positions share at least one theme."""
    pos_by_theme: dict[str, list[str]] = defaultdict(list)
    for pos in positions:
        for t in pos.get("themes", []):
            pos_by_theme[t].append(pos.get("meeting_id", ""))

    results: list[dict[str, Any]] = []
    for am in amendments:
        am_themes = set(am.get("themes", []))
        matching_ids: set[str] = set()
        for t in am_themes:
            matching_ids.update(pos_by_theme.get(t, []))
        results.append(
            {
                "amendment_number": am.get("number"),
                "source": am.get("source"),
                "themes": sorted(am_themes),
                "matching_position_count": len(matching_ids),
                "authors": am.get("authors", []),
                "location": am.get("location", ""),
            }
        )
    results.sort(key=lambda x: x["matching_position_count"], reverse=True)
    return results


def step6_quantitative_analysis(
    data: dict[str, Any],
    amendments: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    compiled_patterns: dict[str, list[re.Pattern]],
    logger: Any = None,
) -> dict[str, Any]:
    """Compute org influence, theme indicators, MEP crossref, and match density."""
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

    theme_indicators = _build_theme_indicators(
        amendments, commission_meetings, lobbying_meetings, taxonomy, compiled_patterns
    )

    theme_lobbying_density = _compute_theme_lobbying_density(theme_indicators)
    amendment_lobbying_density = _compute_amendment_lobbying_density(amendments, positions)

    _log(f"MEPs analysed: {len(mep_crossref)}")
    _log(f"Organisations tracked: {len(org_influence)}")
    _log(f"Themes ranked by lobbying density: {len(theme_lobbying_density)}")

    return {
        "key_meps": key_meps,
        "mep_crossref": mep_crossref,
        "org_influence": org_influence,
        "theme_indicators": theme_indicators,
        "theme_lobbying_density": theme_lobbying_density,
        "amendment_lobbying_density": amendment_lobbying_density,
        "total_procedure_meetings": total_procedure_meetings,
    }
