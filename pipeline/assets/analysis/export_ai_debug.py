"""Export CSV files showing AI input→output for each step of the influence pipeline.

Usage:
    python -m pipeline.assets.analysis.export_ai_debug "2022/0272(COD)"

Produces CSVs in analysis/{procedure_id}/:
    step2_taxonomy.csv          — AI-generated theme taxonomy
    step4_classify.csv          — amendment diff → assigned themes
    step5_positions.csv         — commission meeting text → extracted position
    step7_alignment.csv         — (amendment, position) pair → toward/away/neutral + reasoning
    step9_proposal_alignment.csv— position vs proposal text → reflected/not_reflected
    step10_text_evolution.csv   — theme sections per stage → change direction
    step11_final_alignment.csv  — position vs final text → reflected/not_reflected
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

_cwd = os.getcwd()
if _cwd in sys.path:
    sys.path.remove(_cwd)
sys.path.insert(0, _cwd)

from dotenv import load_dotenv

load_dotenv(os.path.join(_cwd, ".env"))

from supabase import create_client


def _compute_readable_diff(orig: str, amend: str) -> str:
    """Produce a human-readable diff showing what actually changed."""
    import difflib

    orig_words = orig.split()
    amend_words = amend.split()

    sm = difflib.SequenceMatcher(None, orig_words, amend_words)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            old = " ".join(orig_words[i1:i2])
            new = " ".join(amend_words[j1:j2])
            parts.append(f"CHANGED: '{old[:150]}' -> '{new[:150]}'")
        elif tag == "insert":
            new = " ".join(amend_words[j1:j2])
            parts.append(f"ADDED: '{new[:150]}'")
        elif tag == "delete":
            old = " ".join(orig_words[i1:i2])
            parts.append(f"REMOVED: '{old[:150]}'")

    if not parts:
        return "(whitespace-only change)"
    return " | ".join(parts[:5])  # limit to 5 diff chunks


def main() -> None:
    procedure_id = sys.argv[1] if len(sys.argv) > 1 else "2022/0272(COD)"

    client = create_client(
        os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )

    pid_dir = procedure_id.replace("/", ":")
    out_dir = Path("analysis") / pid_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if a report already exists
    report_path = out_dir / "influence_report.json"
    if report_path.exists():
        print(f"Loading existing report from {report_path}")
        with open(report_path) as f:
            report = json.load(f)
        _export_from_report(report, out_dir, client, procedure_id)
    else:
        print(f"No existing report. Run the pipeline first:")
        print(f'  python -c "from pipeline.assets.analysis.influence import run_influence_pipeline; ..."')
        sys.exit(1)


def _export_from_report(
    report: dict, out_dir: Path, client, procedure_id: str
) -> None:
    """Extract CSVs from an existing report + Supabase data."""

    # --- Step 2: Taxonomy ---
    taxonomy = report.get("taxonomy", {})
    with open(out_dir / "step2_taxonomy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["theme_key", "description", "keywords", "salience"])
        for key, t in taxonomy.items():
            w.writerow([
                key,
                t.get("description", ""),
                "; ".join(t.get("keywords", [])),
                t.get("salience", ""),
            ])
    print(f"  step2_taxonomy.csv — {len(taxonomy)} themes")

    # --- Step 4: Amendment classification ---
    # Need to re-fetch amendments from Supabase to get the text
    from pipeline.assets.analysis.influence import fetch_all, compile_taxonomy_patterns

    db_amendments = fetch_all(
        client, "procedure_amendments", "*", {"procedure_id": procedure_id}
    )

    # We need the classified themes — re-classify or use theme_indicators to infer
    # Since we want the actual per-amendment classification, let's re-run step 4
    # But that costs AI calls. Instead, let's run it with regex-only for the CSV
    # and note that the actual pipeline uses AI.
    compiled = compile_taxonomy_patterns(taxonomy)

    from pipeline.assets.analysis.influence import _classify_by_regex

    with open(out_dir / "step4_classify.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "amendment_number", "document_id", "target_element", "authors",
            "diff_type", "diff_summary", "justification",
            "themes_assigned",
        ])
        for row in sorted(db_amendments, key=lambda x: x.get("amendment_number", 0)):
            orig = (row.get("original_text") or "").strip()
            amend = (row.get("amended_text") or "").strip()

            # Compute diff type and summary
            if not orig and not amend:
                diff_type = "empty"
                diff_summary = ""
            elif orig and amend and orig.strip() == amend.strip():
                diff_type = "no_change"
                diff_summary = "(identical text)"
            elif not orig:
                diff_type = "insertion"
                diff_summary = f"NEW: {amend[:400]}"
            elif not amend or amend.lower() == "deleted":
                diff_type = "deletion"
                diff_summary = f"DELETED: {orig[:400]}"
            else:
                diff_type = "modification"
                # Find the actual difference
                diff_summary = _compute_readable_diff(orig, amend)

            # Regex classification
            body = f"{orig} {amend}".strip()
            text = body + " " + (row.get("justification") or "") + " " + (row.get("target_element") or "")
            themes = _classify_by_regex(text, compiled)

            authors = row.get("submitted_by") or []
            w.writerow([
                row.get("amendment_number", ""),
                row.get("document_id", ""),
                row.get("target_element", ""),
                "; ".join(str(a) for a in authors[:5]),
                diff_type,
                diff_summary[:500],
                (row.get("justification") or "")[:200],
                "; ".join(themes),
            ])
    print(f"  step4_classify.csv — {len(db_amendments)} amendments (regex-classified; pipeline uses AI)")

    # --- Step 5: Position extraction ---
    positions = report.get("positions", [])
    with open(out_dir / "step5_positions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "meeting_id", "date", "commissioner", "organisations",
            "themes_extracted", "direction", "position_summary", "ai_enhanced",
        ])
        for pos in positions:
            # Handle orgs field (can be string repr of list or actual list)
            orgs_raw = pos.get("orgs") or pos.get("org_name") or ""
            if isinstance(orgs_raw, list):
                orgs = "; ".join(str(o) for o in orgs_raw)
            elif isinstance(orgs_raw, str) and orgs_raw.startswith("["):
                # String repr of list like "['OVH Groupe']"
                orgs = orgs_raw.strip("[]").replace("'", "").strip()
            else:
                orgs = str(orgs_raw)

            themes_raw = pos.get("themes") or []
            if isinstance(themes_raw, str) and themes_raw.startswith("["):
                themes = themes_raw.strip("[]").replace("'", "").strip()
            elif isinstance(themes_raw, list):
                themes = "; ".join(themes_raw)
            else:
                themes = str(themes_raw)

            w.writerow([
                pos.get("meeting_id") or pos.get("id", ""),
                pos.get("date", ""),
                pos.get("commissioner", ""),
                orgs,
                themes,
                pos.get("direction", ""),
                (pos.get("summary") or pos.get("position_summary") or "")[:400],
                pos.get("ai_enhanced", ""),
            ])
    print(f"  step5_positions.csv — {len(positions)} positions")

    # --- Step 7: Directional alignment ---
    alignment = report.get("directional_alignment", {})
    with open(out_dir / "step7_alignment.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "mep", "theme", "amendment_number", "position_org",
            "score", "reasoning",
        ])
        for mep, data in alignment.items():
            for theme, scores in data.get("theme_scores", {}).items():
                for pair in scores.get("pair_details", []):
                    score_label = {1: "toward", -1: "away", 0: "neutral"}.get(
                        pair.get("score"), "unknown"
                    )
                    w.writerow([
                        mep,
                        theme,
                        pair.get("amendment_number", ""),
                        pair.get("position_org", ""),
                        score_label,
                        (pair.get("reasoning") or "")[:300],
                    ])
    total_pairs = sum(
        len(scores.get("pair_details", []))
        for data in alignment.values()
        for scores in data.get("theme_scores", {}).values()
    )
    print(f"  step7_alignment.csv — {total_pairs} pairs")

    # --- Step 9: Proposal alignment ---
    prop_align = report.get("proposal_alignment", {})
    if prop_align and not prop_align.get("skipped"):
        with open(out_dir / "step9_proposal_alignment.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["theme", "org", "position_summary", "alignment", "evidence", "confidence"])
            for theme, tdata in prop_align.get("theme_results", {}).items():
                details = tdata if isinstance(tdata, list) else tdata.get("details", [])
                for detail in details:
                    w.writerow([
                        theme,
                        detail.get("org", ""),
                        (detail.get("summary") or detail.get("position_summary") or detail.get("position", ""))[:200],
                        detail.get("alignment") or detail.get("reflection_score", ""),
                        (detail.get("evidence") or detail.get("reasoning") or "")[:200],
                        detail.get("confidence", ""),
                    ])
        total_scored = sum(
            len(t) if isinstance(t, list) else len(t.get("details", []))
            for t in prop_align.get("theme_results", {}).values()
        )
        print(f"  step9_proposal_alignment.csv — {total_scored} position-proposal pairs")
    else:
        print("  step9_proposal_alignment.csv — skipped (no data)")

    # --- Step 10: Text evolution ---
    text_evo = report.get("text_evolution", {})
    if text_evo and not text_evo.get("skipped"):
        with open(out_dir / "step10_text_evolution.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "theme", "proposal_to_committee_direction", "proposal_to_committee_summary",
                "committee_to_adopted_direction", "committee_to_adopted_summary",
                "overall_trajectory",
            ])
            for theme, tdata in text_evo.get("theme_evolution", {}).items():
                p2c = tdata.get("proposal_to_committee", {})
                c2a = tdata.get("committee_to_adopted", {})
                w.writerow([
                    theme,
                    p2c.get("direction", p2c.get("change_direction", "")),
                    (p2c.get("summary", p2c.get("change_summary", "")))[:300],
                    c2a.get("direction", c2a.get("change_direction", "")),
                    (c2a.get("summary", c2a.get("change_summary", "")))[:300],
                    tdata.get("overall", tdata.get("overall_trajectory", "")),
                ])
        print(f"  step10_text_evolution.csv — {len(text_evo.get('theme_evolution', {}))} themes")
    else:
        print("  step10_text_evolution.csv — skipped (no data)")

    # --- Step 11: Lifecycle scores ---
    lifecycle = report.get("lifecycle_scores", {})
    if lifecycle and not lifecycle.get("skipped"):
        # Final alignment details
        final_align = lifecycle.get("final_text_alignment", {})
        if final_align and not final_align.get("skipped"):
            with open(out_dir / "step11_final_alignment.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["theme", "org", "position_summary", "alignment", "evidence", "confidence"])
                for theme, tdata in final_align.get("theme_results", {}).items():
                    details = tdata if isinstance(tdata, list) else tdata.get("details", [])
                    for detail in details:
                        w.writerow([
                            theme,
                            detail.get("org", ""),
                            (detail.get("summary") or detail.get("position_summary") or detail.get("position", ""))[:200],
                            detail.get("alignment") or detail.get("reflection_score", ""),
                            (detail.get("evidence") or detail.get("reasoning") or "")[:200],
                            detail.get("confidence", ""),
                        ])
            total_final = sum(
                len(t) if isinstance(t, list) else len(t.get("details", []))
                for t in final_align.get("theme_results", {}).values()
            )
            print(f"  step11_final_alignment.csv — {total_final} position-final pairs")

        # LII scores
        lii = lifecycle.get("lii_scores", {})
        if lii:
            with open(out_dir / "step11_lifecycle_scores.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "theme", "lii",
                    "commission_reflection_rate", "amendment_toward_rate",
                    "final_reflection_rate", "persistence_rate",
                ])
                for theme, data in sorted(lii.items(), key=lambda x: x[1].get("lii", 0), reverse=True):
                    comp = data.get("components", {})
                    w.writerow([
                        theme,
                        f"{data.get('lii', 0):.3f}",
                        f"{comp.get('commission_reflection_rate', 'N/A')}",
                        f"{comp.get('amendment_toward_rate', 'N/A')}",
                        f"{comp.get('final_reflection_rate', 'N/A')}",
                        f"{comp.get('persistence_rate', 'N/A')}",
                    ])
            print(f"  step11_lifecycle_scores.csv — {len(lii)} themes")
    else:
        print("  step11 — skipped (no data)")

    print(f"\nAll CSVs written to {out_dir}/")


if __name__ == "__main__":
    main()
