"""Measure self-consistency (stability) of the production AI matcher.

Re-runs `pipeline.assets.procedures.matching.ai_classify_batch` on every
row of analysis/gold_procedure.csv N times (default 5) and reports how
often the model returns the same top-1 prediction across runs.

This is NOT an accuracy metric — it measures *precision* of the model
under stochasticity, not whether its answers are correct. It complements
gold_evaluate.py and answers the question: "if I rerun the matcher, do I
get the same answer?".

Stability runs do NOT require human-labelled `true_label`, so this can
ship before the gold set is fully reviewed.

Outputs:
    analysis/gold_stability_procedure.json
    analysis/gold_stability_procedure_report.md

Usage:
    .venv/bin/python scripts/gold_stability.py --n 5
    .venv/bin/python scripts/gold_stability.py --n 10 --top-k 3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

# supabase shadow-pkg guard: import before adding ROOT to sys.path.
from supabase import create_client

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import anthropic
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "matching", ROOT / "pipeline" / "assets" / "procedures" / "matching.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ai_classify_batch = _mod.ai_classify_batch
AIBatchError = _mod.AIBatchError
AIQuotaError = _mod.AIQuotaError


_HERE = Path(__file__).resolve().parent
PROCEDURE_PATH = _HERE / "gold_procedure.csv"
JSON_OUT       = _HERE / "gold_stability_procedure.json"
MD_OUT         = _HERE / "gold_stability_procedure_report.md"

# Production fuzzy step returns top-3 candidates to the AI (matching.py).
DEFAULT_TOP_K = 3
DEFAULT_BATCH = 50
DEFAULT_N = 5
DEFAULT_MODEL = "claude-sonnet-4-6"


def _load_gold(path: Path, log: logging.Logger) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} not found. Run scripts/gold_sample.py first.")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log.info(f"Loaded {len(rows)} gold rows from {path}")
    return rows


def _fetch_proc_details(client, procedure_ids: list[str], log: logging.Logger) -> dict[str, dict]:
    """Bulk-fetch proposal/decision dates + subjects for a set of procedure ids.

    Subjects make the AI prompt more informative and mirror production behaviour.
    Chunked to keep URL length under Supabase's cap.
    """
    out: dict[str, dict] = {}
    unique = list(dict.fromkeys(procedure_ids))
    for i in range(0, len(unique), 200):
        batch = unique[i : i + 200]
        resp = (
            client.table("procedures")
            .select("id,proposal_date,decision_date,subjects")
            .in_("id", batch)
            .execute()
        )
        for r in resp.data or []:
            out[r["id"]] = {
                "proposal_date": r.get("proposal_date") or "",
                "decision_date": r.get("decision_date") or "",
                "subjects": r.get("subjects") or [],
            }
    log.info(f"Fetched details for {len(out)}/{len(unique)} procedures")
    return out


def _build_batch_items(rows: list[dict], proc_details: dict[str, dict],
                       top_k: int) -> list[tuple[int, dict]]:
    """Build (gold_row_idx, batch_item) tuples ready for ai_classify_batch."""
    items: list[tuple[int, dict]] = []
    for idx, r in enumerate(rows):
        try:
            cands = json.loads(r.get("candidates_json") or "[]")
        except json.JSONDecodeError:
            cands = []
        cands = cands[:top_k]
        if not cands or not (r.get("meeting_text") or "").strip():
            # No candidates → AI step is skipped in production. Mark as a
            # forced "no_match" outcome with stability=1.0; this mirrors what
            # production would do.
            items.append((idx, None))  # type: ignore
            continue

        # Production passes candidates as: {procedure_id, title}
        ai_cands = [{"procedure_id": c["procedure_id"], "title": c["title"]} for c in cands]
        item = {
            "text": r.get("meeting_text") or "",
            "date": r.get("meeting_date") or "",
            "candidates": ai_cands,
            "_proc_details": {c["procedure_id"]: proc_details.get(c["procedure_id"], {})
                              for c in cands},
        }
        items.append((idx, item))
    return items


def _index_to_procedure_id(item: dict, chosen_index: int) -> str:
    if chosen_index < 0 or chosen_index >= len(item["candidates"]):
        return "no_match"
    return item["candidates"][chosen_index]["procedure_id"]


def _run_once(client_an, items_with_idx: list[tuple[int, dict]],
              batch_size: int, model: str, log: logging.Logger) -> list[str]:
    """Run the AI matcher once over all items. Returns predictions
    aligned to gold_row order: predictions[i] = procedure_id or 'no_match'."""
    n = len(items_with_idx)
    preds: list[str] = ["no_match"] * n
    actionable = [(i, item) for i, (_, item) in enumerate(items_with_idx) if item is not None]

    for start in range(0, len(actionable), batch_size):
        chunk = actionable[start : start + batch_size]
        batch = [c[1] for c in chunk]
        try:
            results = ai_classify_batch(batch, client_an, model=model)
        except AIQuotaError as e:
            log.error(f"Quota error — aborting: {e}")
            raise
        except AIBatchError as e:
            log.warning(f"Batch failed (skipping {len(batch)} items): {e}")
            continue

        for (out_idx, item), res in zip(chunk, results):
            if res.get("match") == "high":
                preds[out_idx] = _index_to_procedure_id(item, res.get("chosen_index", -1))
            else:
                preds[out_idx] = "no_match"

        log.info(f"  batch {start // batch_size + 1}/"
                 f"{(len(actionable) + batch_size - 1) // batch_size}: "
                 f"{len(batch)} items classified")
    return preds


def _aggregate(per_run: list[list[str]], rows: list[dict]) -> dict:
    """Compute per-row + aggregate stability stats.

    For each row, stability = (count of modal prediction) / N. Fully stable
    rows have stability == 1.0 (every run agreed). 'distinct' counts the
    number of unique predictions across runs.
    """
    n_runs = len(per_run)
    n_rows = len(per_run[0]) if per_run else 0

    per_row = []
    fully_stable = 0
    sum_stability = 0.0
    by_source: dict[str, list[float]] = {"lobbying": [], "commission": []}
    by_modal_class: dict[str, list[float]] = {"matched": [], "no_match": []}
    flips_examples: list[dict] = []

    for i in range(n_rows):
        run_preds = [per_run[r][i] for r in range(n_runs)]
        c = Counter(run_preds)
        modal_pred, modal_count = c.most_common(1)[0]
        stability = modal_count / n_runs
        distinct = len(c)
        sum_stability += stability
        if stability == 1.0:
            fully_stable += 1

        source = rows[i].get("source", "")
        if source in by_source:
            by_source[source].append(stability)

        modal_class = "no_match" if modal_pred == "no_match" else "matched"
        by_modal_class[modal_class].append(stability)

        per_row.append({
            "meeting_id": rows[i].get("meeting_id", ""),
            "source": source,
            "meeting_text_preview": (rows[i].get("meeting_text", "") or "")[:120],
            "runs": run_preds,
            "modal": modal_pred,
            "modal_count": modal_count,
            "distinct_predictions": distinct,
            "stability": stability,
        })

        if distinct > 1 and len(flips_examples) < 20:
            flips_examples.append(per_row[-1])

    def _summarise(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": 0.0, "fully_stable_rate": 0.0}
        return {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "fully_stable_rate": sum(1 for v in vals if v == 1.0) / len(vals),
        }

    return {
        "n_rows": n_rows,
        "n_runs": n_runs,
        "fully_stable_rate": fully_stable / n_rows if n_rows else 0.0,
        "mean_stability": sum_stability / n_rows if n_rows else 0.0,
        "by_source": {k: _summarise(v) for k, v in by_source.items()},
        "by_modal_class": {k: _summarise(v) for k, v in by_modal_class.items()},
        "flips_examples": flips_examples,
        "per_row": per_row,
    }


def _build_report(agg: dict, model: str, top_k: int) -> str:
    lines = ["# Self-consistency (stability) of the AI procedure matcher", ""]
    lines.append(
        f"Methodology: the production AI step (`ai_classify_batch`, "
        f"model `{model}`, top-{top_k} fuzzy candidates) was re-run "
        f"{agg['n_runs']} times over the {agg['n_rows']} gold-set rows. "
        "For each row, stability = (count of modal prediction across runs) / "
        f"{agg['n_runs']}. Reported numbers measure model self-consistency "
        "at the operating temperature, not predictive accuracy. Accuracy is "
        "evaluated separately in `gold_evaluate.py` once the gold set has "
        "been human-labelled."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append(f"- **Fully stable rows** (every run agreed): "
                 f"{agg['fully_stable_rate']:.1%}")
    lines.append(f"- **Mean per-row stability**: {agg['mean_stability']:.3f}")
    lines.append("")

    lines.append("## By source")
    lines.append("| source | n | mean stability | fully stable rate |")
    lines.append("|---|---|---|---|")
    for src, s in agg["by_source"].items():
        lines.append(f"| {src} | {s['n']} | {s['mean']:.3f} | {s['fully_stable_rate']:.1%} |")
    lines.append("")

    lines.append("## By modal-prediction class")
    lines.append("| modal class | n | mean stability | fully stable rate |")
    lines.append("|---|---|---|---|")
    for cls, s in agg["by_modal_class"].items():
        lines.append(f"| {cls} | {s['n']} | {s['mean']:.3f} | {s['fully_stable_rate']:.1%} |")
    lines.append("")

    if agg["flips_examples"]:
        lines.append("## Sample of unstable rows (first 20)")
        lines.append("| meeting_id | source | preview | distinct | runs |")
        lines.append("|---|---|---|---|---|")
        for ex in agg["flips_examples"]:
            preview = ex["meeting_text_preview"].replace("|", "\\|")
            runs = " / ".join(ex["runs"])
            lines.append(
                f"| `{ex['meeting_id'][:12]}…` | {ex['source']} | "
                f"{preview[:60]} | {ex['distinct_predictions']} | {runs} |"
            )
        lines.append("")

    lines.append("## Methodology paragraph (paste into thesis)")
    lines.append(
        f"To quantify the stochastic component of the AI matcher, the "
        f"production classification step (Claude {model}, default sampling "
        f"parameters) was applied to all {agg['n_rows']} gold-set inputs "
        f"{agg['n_runs']} independent times. For each input we recorded the "
        "modal prediction and computed stability as the fraction of runs "
        f"agreeing with the mode. The matcher returned the same answer "
        f"across all {agg['n_runs']} runs on "
        f"{agg['fully_stable_rate']:.1%} of inputs (mean per-input stability "
        f"{agg['mean_stability']:.3f}). This bounds the fraction of disagreement "
        "between matcher decisions and gold labels that can be attributed to "
        "model stochasticity rather than systematic error: any accuracy "
        "deficit smaller than (1 − mean stability) cannot be reliably "
        "distinguished from sampling noise."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of independent runs (default {DEFAULT_N}).")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Candidates per item passed to AI (default {DEFAULT_TOP_K}, "
                        "matching production).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                        help=f"Items per AI request (default {DEFAULT_BATCH}, "
                        "matching production).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Anthropic model id (default {DEFAULT_MODEL}).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("gold_stability")

    rows = _load_gold(PROCEDURE_PATH, log)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set.")
    client_an = anthropic.Anthropic(api_key=api_key)
    client_sb = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )

    # Collect every procedure_id that will appear in any candidate list, fetch
    # subjects once.
    all_pids: list[str] = []
    for r in rows:
        try:
            cands = json.loads(r.get("candidates_json") or "[]")[: args.top_k]
        except json.JSONDecodeError:
            cands = []
        all_pids.extend(c.get("procedure_id", "") for c in cands if c.get("procedure_id"))
    proc_details = _fetch_proc_details(client_sb, all_pids, log)

    items_with_idx = _build_batch_items(rows, proc_details, args.top_k)
    actionable = sum(1 for _, it in items_with_idx if it is not None)
    log.info(f"Actionable rows (passed to AI): {actionable}/{len(rows)}")

    per_run: list[list[str]] = []
    for run_i in range(args.n):
        log.info(f"=== Run {run_i + 1}/{args.n} ===")
        per_run.append(
            _run_once(client_an, items_with_idx, args.batch_size, args.model, log)
        )

    agg = _aggregate(per_run, rows)
    JSON_OUT.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    MD_OUT.write_text(_build_report(agg, args.model, args.top_k))

    log.info(f"Wrote {JSON_OUT}")
    log.info(f"Wrote {MD_OUT}")
    print()
    print(f"STABILITY: fully_stable={agg['fully_stable_rate']:.1%}, "
          f"mean={agg['mean_stability']:.3f}, "
          f"n_rows={agg['n_rows']}, n_runs={agg['n_runs']}")


if __name__ == "__main__":
    main()
