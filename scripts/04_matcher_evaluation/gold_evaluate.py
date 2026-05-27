"""Evaluate the meeting → procedure matcher against the gold set.

Reads `gold_procedure.csv` (the hand-curated 200-row gold set plus two
columns — `predicted_procedure_id`, `predicted_match_method` — carrying the
matcher's frozen production decision per row), substitutes the post-relabel
`v2_true_label` on rows where the second annotation pass disagreed with the
first, and computes accuracy / precision / recall / F1 with Wilson 95% CIs
(F1 with a percentile bootstrap CI) plus the confusion matrix and a
per-method precision breakdown.

The frozen `predicted_*` columns are what make the cited F1 numbers
reproducible — Supabase has drifted since evaluation time, so live lookups
would return different values.

Output (written next to this script, overwritten on each run):
    gold_eval.json         — machine-readable metrics

Usage:
    python scripts/04_matcher_evaluation/gold_evaluate.py
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
PROCEDURE_PATH = _HERE / "gold_procedure.csv"
JSON_OUT       = _HERE / "gold_eval.json"


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

    # Frozen matcher predictions live in the CSV as columns:
    #   predicted_procedure_id, predicted_match_method
    for r in rows:
        r["_predicted_label"]  = (r.get("predicted_procedure_id") or "").strip() or "no_match"
        r["_predicted_method"] = (r.get("predicted_match_method") or "").strip()

    overall = _metrics(rows, log)
    by_source: dict[str, dict] = {}
    for src in ("lobbying", "commission"):
        sub = [r for r in rows if r.get("source") == src]
        if sub:
            by_source[src] = _metrics(sub, log)
    overall["by_source"] = by_source
    return overall


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

    log.info(f"Wrote {JSON_OUT}")
    print()
    print(
        f"PROCEDURE: precision={proc['precision']:.1%}, recall={proc['recall']:.1%}, "
        f"F1={proc['f1']:.3f}, n={proc['n_evaluated']}"
    )


if __name__ == "__main__":
    main()
