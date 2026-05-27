"""Ablation study: compare model × prompt configurations on the gold set.

Runs at temperature=0 for deterministic comparison:
  - 2 models   : claude-sonnet-4-6, claude-haiku-4-5-20251001
  - 2 prompts  : few-shot (production, 10 examples), zero-shot (rules only)
  = 4 configurations total

Each configuration is evaluated against the v2 gold labels using the same
confusion-matrix logic as gold_evaluate.py.

Outputs:
    analysis/gold_ablation.json
    analysis/gold_ablation_report.md

Usage:
    .venv/bin/python scripts/gold_ablation.py
    .venv/bin/python scripts/gold_ablation.py --configs sonnet-fewshot,haiku-zeroshot
    .venv/bin/python scripts/gold_ablation.py --batch-size 25
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import importlib.util

import anthropic

# Load matching.py directly to avoid the pipeline package __init__ chain
# (which triggers supabase import and hangs from iCloud-resident venvs).
_matching_spec = importlib.util.spec_from_file_location(
    "matching", ROOT / "pipeline" / "assets" / "procedures" / "matching.py"
)
_matching_mod = importlib.util.module_from_spec(_matching_spec)
_matching_spec.loader.exec_module(_matching_mod)

FEWSHOT_PROMPT = _matching_mod._PROMPT_STATIC_PREFIX
_build_match_prompt_dynamic = _matching_mod._build_match_prompt_dynamic
AIBatchError = _matching_mod.AIBatchError
AIQuotaError = _matching_mod.AIQuotaError

_HERE = Path(__file__).resolve().parent
GOLD_V2_PATH = _HERE / "gold_procedure.csv"
ENRICH_PATH  = _HERE / "gold_v2_enrichment.json"
JSON_OUT     = _HERE / "gold_ablation.json"
MD_OUT       = _HERE / "gold_ablation_report.md"

# ---------------------------------------------------------------------------
# Zero-shot prompt: task description + rules only, no worked examples
# ---------------------------------------------------------------------------

_RULES_END = "is no_match.\n\n"
_rules_idx = FEWSHOT_PROMPT.index(_RULES_END) + len(_RULES_END)
ZEROSHOT_PROMPT = (
    FEWSHOT_PROMPT[:_rules_idx]
    + "Now classify the meetings below.\n\n"
)

# ---------------------------------------------------------------------------
# Configuration matrix
# ---------------------------------------------------------------------------

CONFIGS = {
    "sonnet-fewshot": {
        "model": "claude-sonnet-4-6",
        "prompt": FEWSHOT_PROMPT,
        "label": "Sonnet + few-shot",
    },
    "sonnet-zeroshot": {
        "model": "claude-sonnet-4-6",
        "prompt": ZEROSHOT_PROMPT,
        "label": "Sonnet + zero-shot",
    },
    "haiku-fewshot": {
        "model": "claude-haiku-4-5-20251001",
        "prompt": FEWSHOT_PROMPT,
        "label": "Haiku + few-shot",
    },
    "haiku-zeroshot": {
        "model": "claude-haiku-4-5-20251001",
        "prompt": ZEROSHOT_PROMPT,
        "label": "Haiku + zero-shot",
    },
}


# ---------------------------------------------------------------------------
# Wilson CI (same as gold_evaluate.py)
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def bootstrap_f1_ci(items: list[dict], B: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    if not items:
        return (0.0, 0.0, 0.0)

    def _f1(sample):
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
                fp += 1; fn += 1
        if tp == 0:
            return 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    rng = random.Random(seed)
    n = len(items)
    point = _f1(items)
    samples = sorted(_f1([items[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return (point, samples[int(0.025 * B)], samples[int(0.975 * B)])


# ---------------------------------------------------------------------------
# Gold set loading
# ---------------------------------------------------------------------------

def _load_gold(log: logging.Logger) -> tuple[list[dict], dict[tuple[str, str], str]]:
    """Load v2 gold set and matcher predictions. Returns (rows, gold_labels)."""
    if not GOLD_V2_PATH.exists():
        sys.exit(f"{GOLD_V2_PATH} not found.")
    with GOLD_V2_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if r.get("v2_diff") == "1" and (r.get("v2_true_label") or "").strip():
            r["true_label"] = r["v2_true_label"]

    gold_labels = {}
    for r in rows:
        key = (r.get("source"), r.get("meeting_id"))
        gold_labels[key] = (r.get("true_label") or "").strip()

    log.info(f"Loaded {len(rows)} gold rows")
    return rows, gold_labels


def _fetch_proc_details(client_sb, procedure_ids: list[str], log: logging.Logger) -> dict[str, dict]:
    out: dict[str, dict] = {}
    unique = list(dict.fromkeys(procedure_ids))
    for i in range(0, len(unique), 200):
        batch = unique[i:i + 200]
        resp = (
            client_sb.table("procedures")
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
                       top_k: int = 3) -> list[tuple[int, dict | None]]:
    items = []
    for idx, r in enumerate(rows):
        try:
            cands = json.loads(r.get("candidates_json") or "[]")
        except json.JSONDecodeError:
            cands = []
        cands = cands[:top_k]
        if not cands or not (r.get("meeting_text") or "").strip():
            items.append((idx, None))
            continue
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


# ---------------------------------------------------------------------------
# AI classification with configurable prompt and temperature
# ---------------------------------------------------------------------------

_QUOTA_SIGNALS = (
    "credit balance", "credit_balance", "insufficient", "quota",
    "billing", "authentication_error", "invalid x-api-key",
    "invalid api key", "organization_disabled",
)


def _classify_batch(
    batch: list[dict],
    client: anthropic.Anthropic,
    model: str,
    prompt_prefix: str,
    temperature: float = 0.0,
) -> list[dict]:
    """Run AI classification on a batch with specified config."""
    if not batch:
        return []

    dynamic_prompt = _build_match_prompt_dynamic(batch)

    content_blocks = [
        {
            "type": "text",
            "text": prompt_prefix,
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": dynamic_prompt},
    ]

    for attempt in range(5):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=temperature,
                messages=[{"role": "user", "content": content_blocks}],
            )
            break
        except Exception as e:
            msg = str(e).lower()
            if any(sig in msg for sig in _QUOTA_SIGNALS):
                raise AIQuotaError(str(e)) from e
            if "429" in msg or "rate_limit" in msg:
                time.sleep(2 ** attempt * 3)
                continue
            raise AIBatchError(f"API error: {e}") from e
    else:
        raise AIBatchError("Rate limit retries exhausted")

    raw = response.content[0].text
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        raise AIBatchError("Response contained no JSON array")
    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        raise AIBatchError(f"JSON parse error: {e}") from e
    if not isinstance(parsed, list):
        raise AIBatchError(f"Expected list, got {type(parsed).__name__}")

    if len(parsed) < len(batch):
        parsed.extend(
            [{"match": "no_match", "chosen": "none", "reasoning": "model_undercounted"}]
            * (len(batch) - len(parsed))
        )
    elif len(parsed) > len(batch):
        parsed = parsed[:len(batch)]

    out = []
    for entry in parsed:
        match_val = (entry.get("match") or "no_match").lower().strip()
        chosen_raw = entry.get("chosen") or entry.get("chosen_index") or "none"
        if match_val == "high" and str(chosen_raw).strip().lower() != "none":
            chosen_letter = str(chosen_raw).strip().upper()
            idx = ord(chosen_letter[0]) - ord("A") if chosen_letter and chosen_letter[0].isalpha() else -1
            out.append({"match": "high", "chosen_index": idx, "reasoning": entry.get("reasoning", "")})
        else:
            out.append({"match": "no_match", "chosen_index": -1, "reasoning": entry.get("reasoning", "")})
    return out


def _run_config(
    config_name: str,
    config: dict,
    items_with_idx: list[tuple[int, dict | None]],
    rows: list[dict],
    client: anthropic.Anthropic,
    batch_size: int,
    log: logging.Logger,
) -> list[str]:
    """Run one configuration over all items. Returns predictions aligned to gold row order."""
    n = len(items_with_idx)
    preds = ["no_match"] * n
    actionable = [(i, item) for i, (_, item) in enumerate(items_with_idx) if item is not None]

    log.info(f"  Actionable rows: {len(actionable)}/{n}")

    for start in range(0, len(actionable), batch_size):
        chunk = actionable[start:start + batch_size]
        batch = [c[1] for c in chunk]
        try:
            results = _classify_batch(
                batch, client,
                model=config["model"],
                prompt_prefix=config["prompt"],
                temperature=0.0,
            )
        except AIQuotaError as e:
            log.error(f"Quota error — aborting config {config_name}: {e}")
            raise
        except AIBatchError as e:
            log.warning(f"Batch failed (skipping {len(batch)} items): {e}")
            continue

        for (out_idx, item), res in zip(chunk, results):
            if res.get("match") == "high":
                ci = res.get("chosen_index", -1)
                if 0 <= ci < len(item["candidates"]):
                    preds[out_idx] = item["candidates"][ci]["procedure_id"]
            # else stays "no_match"

        batch_num = start // batch_size + 1
        total_batches = (len(actionable) + batch_size - 1) // batch_size
        log.info(f"  batch {batch_num}/{total_batches}: {len(batch)} items")

    return preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _is_match(label: str) -> bool:
    if not label:
        return False
    return label.lower() not in {"no_match", "uncertain", "outside_candidates"}


def _evaluate(preds: list[str], rows: list[dict], gold_labels: dict) -> dict:
    """Compute metrics for one configuration's predictions."""
    tp = tn = fp = fn = 0
    excluded = 0
    by_class = []

    for i, r in enumerate(rows):
        key = (r.get("source"), r.get("meeting_id"))
        gold = gold_labels.get(key, "")
        if gold.lower() in ("uncertain", "outside_candidates", ""):
            excluded += 1
            continue

        pred = preds[i]
        gold_pos = _is_match(gold)
        pred_pos = _is_match(pred)

        if not gold_pos and not pred_pos:
            cls = "TN"; tn += 1
        elif gold_pos and not pred_pos:
            cls = "FN"; fn += 1
        elif not gold_pos and pred_pos:
            cls = "FP"; fp += 1
        elif gold == pred:
            cls = "TP"; tp += 1
        else:
            cls = "FP_FN"; fp += 1; fn += 1

        by_class.append({"class": cls})

    prec_p, prec_lo, prec_hi = wilson_ci(tp, tp + fp)
    rec_p, rec_lo, rec_hi = wilson_ci(tp, tp + fn)
    acc_n = sum(1 for r in by_class if r["class"] in ("TP", "TN"))
    acc_p, acc_lo, acc_hi = wilson_ci(acc_n, len(by_class))
    f1_p, f1_lo, f1_hi = bootstrap_f1_ci(by_class)

    return {
        "n_evaluated": len(by_class),
        "n_excluded": excluded,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "fp_fn": sum(1 for r in by_class if r["class"] == "FP_FN"),
        "precision": prec_p, "precision_ci": [prec_lo, prec_hi],
        "recall": rec_p, "recall_ci": [rec_lo, rec_hi],
        "f1": f1_p, "f1_ci": [f1_lo, f1_hi],
        "accuracy": acc_p, "accuracy_ci": [acc_lo, acc_hi],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(p: float, ci: list[float]) -> str:
    return f"{p:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]"


def _build_report(results: dict[str, dict]) -> str:
    lines = [
        "# Ablation study: model × prompt configurations",
        "",
        "All configurations run at temperature=0 on the v2 gold set (n=200, "
        "198 evaluable after excluding uncertain/outside_candidates rows). "
        "Few-shot = production prompt with 10 worked examples; zero-shot = "
        "task description and rules only, no examples.",
        "",
        "## Results",
        "",
        "| Configuration | Precision (95% CI) | Recall (95% CI) | F1 [95% CI] | Accuracy |",
        "|---|---|---|---|---|",
    ]
    for config_name, r in results.items():
        label = CONFIGS[config_name]["label"]
        lines.append(
            f"| {label} | {_fmt(r['precision'], r['precision_ci'])} | "
            f"{_fmt(r['recall'], r['recall_ci'])} | "
            f"{r['f1']:.3f} [{r['f1_ci'][0]:.3f}, {r['f1_ci'][1]:.3f}] | "
            f"{_fmt(r['accuracy'], r['accuracy_ci'])} |"
        )
    lines.append("")

    lines.append("## Confusion matrices")
    lines.append("")
    for config_name, r in results.items():
        label = CONFIGS[config_name]["label"]
        lines.append(f"**{label}:** TP={r['tp']}, FP={r['fp']}, FN={r['fn']}, TN={r['tn']}")
        if r["fp_fn"]:
            lines.append(f"  ({r['fp_fn']} rows counted as both FP and FN)")
    lines.append("")

    lines.append("## Methodology paragraph (paste into thesis)")
    lines.append("")
    config_names = list(results.keys())
    lines.append(
        "To assess the sensitivity of the AI classification step to model choice "
        "and prompt design, we ran an ablation study over four configurations on the "
        "same 198-row gold set at temperature zero for deterministic reproducibility. "
        "The configurations crossed two models (Claude Sonnet 4.6 and Claude Haiku 4.5) "
        "with two prompt variants: the production few-shot prompt containing ten worked "
        "examples alongside classification rules, and a zero-shot variant retaining only "
        "the task description and rules with no examples. All other parameters (candidate "
        "count, temporal filtering, batch structure) were held constant."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default=",".join(CONFIGS.keys()),
        help="Comma-separated config names to run (default: all four).",
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("gold_ablation")

    config_names = [c.strip() for c in args.configs.split(",")]
    for c in config_names:
        if c not in CONFIGS:
            sys.exit(f"Unknown config: {c}. Available: {', '.join(CONFIGS.keys())}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set.")
    client_an = anthropic.Anthropic(api_key=api_key)
    from supabase import create_client
    client_sb = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )

    rows, gold_labels = _load_gold(log)

    all_pids = []
    for r in rows:
        try:
            cands = json.loads(r.get("candidates_json") or "[]")[:args.top_k]
        except json.JSONDecodeError:
            cands = []
        all_pids.extend(c.get("procedure_id", "") for c in cands if c.get("procedure_id"))
    proc_details = _fetch_proc_details(client_sb, all_pids, log)

    items_with_idx = _build_batch_items(rows, proc_details, args.top_k)
    actionable = sum(1 for _, it in items_with_idx if it is not None)
    log.info(f"Actionable rows (passed to AI): {actionable}/{len(rows)}")

    results = {}
    for config_name in config_names:
        config = CONFIGS[config_name]
        log.info(f"=== {config['label']} ({config_name}) ===")
        preds = _run_config(config_name, config, items_with_idx, rows, client_an, args.batch_size, log)
        metrics = _evaluate(preds, rows, gold_labels)
        results[config_name] = metrics
        log.info(
            f"  => precision={metrics['precision']:.1%}, recall={metrics['recall']:.1%}, "
            f"F1={metrics['f1']:.3f}"
        )

    JSON_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    MD_OUT.write_text(_build_report(results))

    log.info(f"Wrote {JSON_OUT}")
    log.info(f"Wrote {MD_OUT}")

    print()
    print(f"{'Config':<25s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Acc':>6s}")
    print("-" * 55)
    for config_name, r in results.items():
        label = CONFIGS[config_name]["label"]
        print(
            f"{label:<25s} {r['precision']:>5.1%} {r['recall']:>5.1%} "
            f"{r['f1']:>5.3f} {r['accuracy']:>5.1%}"
        )


if __name__ == "__main__":
    main()
