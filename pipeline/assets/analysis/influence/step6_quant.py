"""Step 6: Quantitative analysis — LEI, ALAS, ICI, theme indicators, statistical tests."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ._helpers import _classify_by_regex, _meeting_text

# ---------------------------------------------------------------------------
# Optional scipy
# ---------------------------------------------------------------------------

try:
    from scipy import stats as scipy_stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Organisation interest weights
# ---------------------------------------------------------------------------

_ORG_INTEREST_WEIGHTS: dict[str, float] = {
    "Promotes their own interests or the collective interests of their members": 1.0,
    "Advances interests of their clients": 0.8,
    "Does not represent commercial interests": 0.15,
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
