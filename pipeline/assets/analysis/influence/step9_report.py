"""Step 9: Report generation — assemble JSON report and generate evidence dossier one-pager."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import _config
from ._ai import ai_complete


def step9_generate_report(
    procedure_id: str,
    data: dict[str, Any],
    taxonomy: dict[str, Any],
    amendments: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    quant: dict[str, Any],
    commission_evidence: list[dict[str, Any]],
    amendment_evidence: list[dict[str, Any]],
    output_dir: Path | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Assemble the full structured report and write to disk."""
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
        "ai_provider": _config.AI_PROVIDER,
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
        "theme_lobbying_density": quant.get("theme_lobbying_density", []),
        "amendment_lobbying_density": quant.get("amendment_lobbying_density", []),
        "org_influence": quant.get("org_influence", {}),
        "mep_crossref": quant.get("mep_crossref", {}),
        "key_meps": quant.get("key_meps", {}),
        "positions": positions,
        "commission_evidence": commission_evidence,
        "amendment_evidence": amendment_evidence,
    }

    pid_dir_name = procedure_id.replace("/", ":")
    proc_dir = (output_dir or _config.ANALYSIS_OUTPUT_DIR) / pid_dir_name
    proc_dir.mkdir(parents=True, exist_ok=True)
    output_path = proc_dir / "influence_report.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    _log(f"JSON report written to: {output_path}")

    if _config.AI_PROVIDER is not None:
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
    """Generate an evidence-dossier one-pager from the report JSON."""
    _log = logger.info if logger else print

    prompt_path = Path(__file__).parent.parent / "one_pager_prompt.md"
    if not prompt_path.exists():
        _log("one_pager_prompt.md not found, skipping one-pager generation")
        return

    prompt_template = prompt_path.read_text(encoding="utf-8")

    # Trim report for prompt context window
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
    # Trim commission evidence: top 5 themes
    if "commission_evidence" in report_trimmed:
        report_trimmed["commission_evidence"] = report_trimmed["commission_evidence"][:5]
    # Trim amendment evidence: top 10 amendments
    if "amendment_evidence" in report_trimmed:
        report_trimmed["amendment_evidence"] = report_trimmed["amendment_evidence"][:10]
    # Trim amendment_lobbying_density to top 20
    if "amendment_lobbying_density" in report_trimmed:
        report_trimmed["amendment_lobbying_density"] = report_trimmed["amendment_lobbying_density"][:20]

    report_json = json.dumps(report_trimmed, indent=2, ensure_ascii=False, default=str)
    user_prompt = prompt_template.split("```json")[0] + "```json\n" + report_json + "\n```"

    system = (
        "You are a policy analyst at a European transparency think tank. "
        "You write concise, evidence-based briefings about lobbying activity on EU legislation. "
        "Your tone is factual and measured. You present data and evidence for the reader to "
        "evaluate — you never judge whether lobbying was successful or influential."
    )

    _log("Generating one-pager via AI ...")
    md = ai_complete(f"{system}\n\n{user_prompt}")

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
