"""Evaluate the meeting → procedure matcher against the gold set.

Reads `gold_procedure.csv` (the hand-curated 200-row gold set), substitutes
the post-relabel `v2_true_label` on rows where the second annotation pass
disagreed with the first, joins per-row matcher predictions from
`gold_v2_enrichment.json`, and computes accuracy / precision / recall / F1
with Wilson 95% CIs (F1 with a percentile bootstrap CI) plus the confusion
matrix and a per-method precision breakdown.

The enrichment JSON is a frozen snapshot of what the production matcher
predicted for each gold row at evaluation time; it is what makes the cited
F1 numbers reproducible (Supabase has drifted since).

Outputs (written next to this script):
    gold_eval.json         — machine-readable metrics
    gold_eval_report.md    — thesis-paste-ready markdown

Usage:
    python scripts/gold_evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROCEDURE_PATH  = _HERE / "gold_procedure.csv"
ENRICHMENT_PATH = _HERE / "gold_v2_enrichment.json"
JSON_OUT        = _HERE / "gold_eval.json"
MD_OUT          = _HERE / "gold_eval_report.md"


# ---------------------------------------------------------------------------
# Wilson 95% CI (hand-rolled; same as compute_ai_reliability.py)
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Returns (point_estimate, lo, hi). All in [0,1]. n=0 → (0,0,0)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


# F1 is non-linear in the underlying counts, so Wilson does not apply.
# We bootstrap the test set: resample rows with replacement B times and
# recompute F1 on each resample, then take the 2.5/97.5 percentiles.
def bootstrap_f1_ci(items: list[dict], B: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Returns (point_estimate, lo, hi) for F1 over a list of pre-classified
    rows. Each `item` must have 'class' ∈ {TP, TN, FP, FN, FP_FN}."""
    if not items:
        return (0.0, 0.0, 0.0)

    def _f1(sample: list[dict]) -> float:
        tp = fp = fn = 0
        for r in sample:
            cls = r["class"]
            if cls == "TP":
                tp += 1
            elif cls == "FP":
                fp += 1
            elif cls == "FN":
                fn += 1
            elif cls == "FP_FN":
                fp += 1
                fn += 1
        if tp == 0:
            return 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        if (prec + rec) == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    rng = random.Random(seed)
    n = len(items)
    point = _f1(items)
    samples = []
    for _ in range(B):
        resample = [items[rng.randrange(n)] for _ in range(n)]
        samples.append(_f1(resample))
    samples.sort()
    lo = samples[int(0.025 * B)]
    hi = samples[int(0.975 * B)]
    return (point, lo, hi)


# ---------------------------------------------------------------------------
# Look up matcher's persisted prediction per input
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Confusion matrix arithmetic
# ---------------------------------------------------------------------------

def _is_match_label(label: str) -> bool:
    """A label is a 'positive' (=matched) if it's not in the special tokens."""
    if not label:
        return False
    return label.lower() not in {"no_match", "uncertain", "outside_candidates"}


def _classify(gold: str, predicted: str) -> str:
    """Return one of: TP, TN, FP, FN, FP_FN.

    FP_FN means matcher predicted X but gold says Y — both a false positive
    (matched something that wasn't gold) and a false negative (missed gold).
    We count them in both buckets in metrics.

    'uncertain' and 'outside_candidates' rows are excluded upstream.
    """
    gold_pos = _is_match_label(gold)
    pred_pos = _is_match_label(predicted)

    if not gold_pos and not pred_pos:
        return "TN"
    if gold_pos and not pred_pos:
        return "FN"
    if not gold_pos and pred_pos:
        return "FP"
    # both positive
    return "TP" if gold == predicted else "FP_FN"


def _metrics_by_method(by_class: list[dict]) -> dict:
    """Compute per-match-method precision breakdown.

    Groups rows by _predicted_method and counts TP vs FP/FP_FN for each
    method that produced a positive prediction. Also reports the no_match
    bucket (TN vs FN).
    """
    buckets: dict[str, dict] = {}
    for r in by_class:
        method = r.get("method") or ""
        pred_positive = r["predicted"].lower() != "no_match"
        if not pred_positive:
            method = "no_match"
        elif not method:
            method = "unknown"
        if method not in buckets:
            buckets[method] = {"n": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "fp_fn": 0}
        b = buckets[method]
        b["n"] += 1
        cls = r["class"]
        if cls == "TP":
            b["tp"] += 1
        elif cls == "TN":
            b["tn"] += 1
        elif cls == "FP":
            b["fp"] += 1
        elif cls == "FN":
            b["fn"] += 1
        elif cls == "FP_FN":
            b["fp_fn"] += 1

    out = {}
    for method, b in sorted(buckets.items()):
        if method == "no_match":
            correct = b["tn"]
            wrong = b["fn"]
        else:
            correct = b["tp"]
            wrong = b["fp"] + b["fp_fn"]
        total = correct + wrong
        p, lo, hi = wilson_ci(correct, total) if total else (0.0, 0.0, 0.0)
        out[method] = {
            "n": b["n"], "correct": correct, "wrong": wrong,
            "precision": p, "precision_ci": [lo, hi],
        }
    return out


def _metrics(rows: list[dict], log: logging.Logger) -> dict:
    """Compute confusion-matrix counts + precision/recall/F1/accuracy."""
    tp = tn = fp = fn = 0
    excluded = 0
    pred_no_match = 0
    pred_match = 0
    by_class: list[dict] = []

    for r in rows:
        gold = (r.get("true_label") or "").strip()
        pred = (r.get("_predicted_label") or "no_match").strip()
        method = (r.get("_predicted_method") or "").strip()
        if gold.lower() in ("uncertain", "outside_candidates", ""):
            excluded += 1
            continue
        cls = _classify(gold, pred)
        if cls == "TP":
            tp += 1
        elif cls == "TN":
            tn += 1
        elif cls == "FP":
            fp += 1
        elif cls == "FN":
            fn += 1
        elif cls == "FP_FN":
            fp += 1
            fn += 1
        if pred and pred.lower() == "no_match":
            pred_no_match += 1
        else:
            pred_match += 1
        by_class.append({"gold": gold, "predicted": pred, "class": cls, "method": method})

    # Precision = TP / (TP + FP); Recall = TP / (TP + FN)
    prec_p, prec_lo, prec_hi = wilson_ci(tp, tp + fp)
    rec_p, rec_lo, rec_hi = wilson_ci(tp, tp + fn)
    if (prec_p + rec_p) > 0:
        f1 = 2 * prec_p * rec_p / (prec_p + rec_p)
    else:
        f1 = 0.0

    acc_n = sum(1 for r in by_class if r["class"] in ("TP", "TN"))
    acc_d = len(by_class)
    acc_p, acc_lo, acc_hi = wilson_ci(acc_n, acc_d)

    f1_p, f1_lo, f1_hi = bootstrap_f1_ci(by_class)

    by_method = _metrics_by_method(by_class)

    return {
        "n_total_in_gold": len(rows),
        "n_excluded_uncertain_or_outside": excluded,
        "n_evaluated": acc_d,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "fp_fn_dual_count": sum(1 for r in by_class if r["class"] == "FP_FN"),
        "precision": prec_p, "precision_ci": [prec_lo, prec_hi],
        "recall": rec_p, "recall_ci": [rec_lo, rec_hi],
        "f1": f1_p, "f1_ci": [f1_lo, f1_hi],
        "accuracy": acc_p, "accuracy_ci": [acc_lo, acc_hi],
        "predicted_match": pred_match,
        "predicted_no_match": pred_no_match,
        "by_method": by_method,
        "_by_class": by_class,  # for downstream stratification
    }


# ---------------------------------------------------------------------------
# Per-component eval
# ---------------------------------------------------------------------------

def _eval_procedure(log: logging.Logger) -> dict:
    if not PROCEDURE_PATH.exists():
        sys.exit(f"{PROCEDURE_PATH} not found")
    if not ENRICHMENT_PATH.exists():
        sys.exit(f"{ENRICHMENT_PATH} not found — required for matcher predictions")

    with PROCEDURE_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Substitute v2_true_label on diff rows; non-diff rows keep their original
    # true_label (Opus v2 agreed with the v1 annotation there).
    n_subst = 0
    for r in rows:
        if r.get("v2_diff") == "1" and (r.get("v2_true_label") or "").strip():
            r["true_label"] = r["v2_true_label"]
            n_subst += 1
    log.info(f"substituted v2_true_label on {n_subst} diff rows")

    with ENRICHMENT_PATH.open() as f:
        enrich = json.load(f)
    preds_full: dict[tuple[str, str], dict] = {}
    for k, v in enrich["predictions"].items():
        src, mid = k.split(":", 1)
        preds_full[(src, mid)] = {
            "procedure_id": v.get("procedure_id") or "no_match",
            "match_method": v.get("match_method") or "",
        }
    log.info(f"loaded {len(preds_full)} matcher predictions from enrichment")

    for r in rows:
        key = (r.get("source"), r.get("meeting_id"))
        info = preds_full.get(key, {"procedure_id": "no_match", "match_method": ""})
        r["_predicted_label"] = info["procedure_id"]
        r["_predicted_method"] = info["match_method"]

    overall = _metrics(rows, log)
    by_source: dict[str, dict] = {}
    for src in ("lobbying", "commission"):
        sub = [r for r in rows if r.get("source") == src]
        if sub:
            by_source[src] = _metrics(sub, log)
    overall["by_source"] = by_source
    return overall


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt_pct_ci(p: float, ci: list[float]) -> str:
    return f"{p:.1%} (95% CI [{ci[0]:.1%}, {ci[1]:.1%}])"


def _build_report(proc: dict) -> str:
    lines = ["# Gold-standard evaluation of EU lobbying matcher", ""]
    lines.append(
        "Methodology: stratified random sample (50% matched / 50% no_match) "
        "with a two-pass annotation process. In the first pass, Claude Opus 4.7 "
        "proposed a label given the meeting text and the top-20 fuzzy "
        "candidates; the researcher accepted, corrected, or rejected each "
        "proposal. In the second pass, each row was enriched with the matcher's "
        "production signals (MEP-declared `related_procedure`, predicted "
        "procedure ID, match method, and matched alias) and Opus 4.7 "
        "re-evaluated the label with this additional context. The 38 rows where "
        "the second-pass proposal disagreed with the first-pass label were "
        "re-reviewed by the researcher, resulting in 35 corrected labels. Final "
        "labels from the second pass are used throughout."
    )
    lines.append("")
    lines.append(
        "Reported metrics use Wilson 95% confidence intervals for proportions "
        "(accuracy, precision, recall) and bootstrap (B=2000) percentile 95% "
        "confidence intervals for F1, which is non-linear in the underlying "
        "counts. Rows the researcher marked `uncertain` or `outside_candidates` "
        "are excluded from the denominator. A wrong-procedure-but-matched row "
        "counts as both a false positive (matched the wrong file) and a false "
        "negative (missed the correct one)."
    )
    lines.append("")

    for label, m in [("Meeting → Procedure", proc)]:
        if not m:
            lines.append(f"## {label}\n\n_Skipped — gold CSV not present._\n")
            continue
        lines.append(f"## {label}\n")
        lines.append(f"- Sample size: {m['n_total_in_gold']}")
        lines.append(f"- Excluded (uncertain / outside_candidates / unlabeled): {m['n_excluded_uncertain_or_outside']}")
        lines.append(f"- Evaluated: {m['n_evaluated']}")
        lines.append("")
        lines.append("**Confusion matrix:**")
        lines.append("")
        lines.append(f"|             | gold = match | gold = no_match |")
        lines.append(f"|-------------|--------------|-----------------|")
        lines.append(f"| pred match  | TP={m['tp']}            | FP={m['fp']}             |")
        lines.append(f"| pred no_m   | FN={m['fn']}            | TN={m['tn']}             |")
        lines.append("")
        if m["fp_fn_dual_count"]:
            lines.append(
                f"_Note: {m['fp_fn_dual_count']} rows had matcher predicting a "
                f"DIFFERENT match than gold. Each is counted as both FP and FN._\n"
            )
        lines.append("**Headline metrics:**")
        lines.append(f"- Accuracy: {_fmt_pct_ci(m['accuracy'], m['accuracy_ci'])}")
        lines.append(f"- Precision: {_fmt_pct_ci(m['precision'], m['precision_ci'])}")
        lines.append(f"- Recall: {_fmt_pct_ci(m['recall'], m['recall_ci'])}")
        lines.append(f"- F1: {m['f1']:.3f} (bootstrap 95% CI [{m['f1_ci'][0]:.3f}, {m['f1_ci'][1]:.3f}])")
        lines.append("")

        if m.get("by_source"):
            lines.append("**Per-source breakdown:**")
            lines.append("")
            lines.append("| source | n | accuracy | precision | recall | F1 (95% CI) |")
            lines.append("|---|---|---|---|---|---|")
            for src, sm in m["by_source"].items():
                lines.append(
                    f"| {src} | {sm['n_evaluated']} | "
                    f"{_fmt_pct_ci(sm['accuracy'], sm['accuracy_ci'])} | "
                    f"{_fmt_pct_ci(sm['precision'], sm['precision_ci'])} | "
                    f"{_fmt_pct_ci(sm['recall'], sm['recall_ci'])} | "
                    f"{sm['f1']:.3f} [{sm['f1_ci'][0]:.3f}, {sm['f1_ci'][1]:.3f}] |"
                )
            lines.append("")

        if m.get("by_method"):
            lines.append("**Per-match-method precision:**")
            lines.append("")
            lines.append("| match method | n | correct | wrong | precision (95% CI) |")
            lines.append("|---|---|---|---|---|")
            for method, mm in m["by_method"].items():
                label_col = "correct (TN)" if method == "no_match" else "correct (TP)"
                label_wrong = "wrong (FN)" if method == "no_match" else "wrong (FP)"
                lines.append(
                    f"| {method} | {mm['n']} | {mm['correct']} | "
                    f"{mm['wrong']} | {_fmt_pct_ci(mm['precision'], mm['precision_ci'])} |"
                )
            lines.append("")

    lines.append("## Methodology paragraph (paste into thesis)\n")
    lines.append(
        "To evaluate the accuracy of the procedure-matching pipeline, a "
        "gold-standard evaluation set was constructed through stratified random "
        "sampling of 200 meeting–procedure inputs, drawn equally from rows the "
        "matcher had classified as matched (n=100) and as no_match (n=100). "
        "Annotation followed a two-pass design to mitigate labelling bias. In "
        "the first pass, Claude Opus 4.7 (Anthropic, 2025) proposed a "
        "ground-truth label for each input given the meeting text and the top-20 "
        "fuzzy candidate procedures; the researcher then accepted, corrected, or "
        "rejected each proposal. In the second pass, each row was enriched with "
        "the matcher's production signals — the MEP-declared related procedure, "
        "the predicted procedure identifier, the match method, and the matched "
        "alias — after which Opus 4.7 re-evaluated its proposal with this "
        "additional context. The 38 rows where the second-pass proposal "
        "disagreed with the first-pass label were re-reviewed by the researcher, "
        "resulting in 35 corrected labels. Two inputs that the researcher could "
        "not resolve confidently were excluded, yielding 198 evaluated rows. "
        "The matcher's persisted production decision for each input was then "
        "compared to the final gold label, producing a confusion matrix from "
        "which precision, recall, and accuracy were computed with Wilson 95% "
        "confidence intervals; F1 was reported with a non-parametric bootstrap "
        "percentile interval (B=2,000 resamples)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    argparse.ArgumentParser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("gold_eval")

    proc = _eval_procedure(log)

    # Strip non-JSON-serialisable internal field before writing.
    def _strip(d: dict) -> dict:
        if not d:
            return d
        out = {k: v for k, v in d.items() if k != "_by_class"}
        if isinstance(out.get("by_source"), dict):
            out["by_source"] = {k: _strip(v) for k, v in out["by_source"].items()}
        return out

    JSON_OUT.write_text(json.dumps({"procedure": _strip(proc)}, indent=2))
    MD_OUT.write_text(_build_report(proc))

    log.info(f"Wrote {JSON_OUT}")
    log.info(f"Wrote {MD_OUT}")
    print()
    print(
        f"PROCEDURE: precision={proc['precision']:.1%}, recall={proc['recall']:.1%}, "
        f"F1={proc['f1']:.3f}, n={proc['n_evaluated']}"
    )


if __name__ == "__main__":
    main()
