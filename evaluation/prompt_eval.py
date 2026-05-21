#!/usr/bin/env python3
"""LLM classification evaluation for ALIGNED / OPPOSING / NOISE labels.

Tests the classifier against an annotated ground truth CSV across:
  - Prompt variant A: current prompt, includes organisation name
  - Prompt variant B: organisation name stripped
  - Prompt variant C: explicit instruction to default to NOISE when uncertain
  - Temperatures: 0.0, 0.5, 1.0

Also runs a stability test: 20 matches × 5 independent runs at temp=0.5
to measure label consistency.

Expected CSV columns (any extra columns are ignored):
  true_label         — one of ALIGNED, OPPOSING, NOISE
  source_text        — the org's meeting/feedback text
  amended_text       — the amendment text (amended_to)
  original_text      — original proposal text (can be blank for new insertions)
  justification      — amendment justification (optional)
  organisation       — org name (optional; used in variant A, ignored in B)
  source_type        — e.g. "commission_meeting" or "feedback" (optional)
  source_date        — ISO date string (optional)

Tip: export amend_matches_df from Amendment_influence_test.ipynb, add a
`true_label` column with your annotations, and pass it to this script.

Usage:
    python evaluation/prompt_eval.py --ground-truth path/to/annotated.csv

    # Run only specific variants / temperatures:
    python evaluation/prompt_eval.py --ground-truth gt.csv --variants A C --temperatures 0.0 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from itertools import product
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

load_dotenv()

LLM_MODEL  = "claude-sonnet-4-6"
MAX_TOKENS = 800
LABELS     = ["ALIGNED", "OPPOSING", "UNDETECTABLE", "NOISE"]

# ── Tool schema ────────────────────────────────────────────────────────────────

CLASSIFY_TOOL = {
    "name": "classify_match",
    "description": (
        "Classify whether a lobbying organisation's text aligns with, opposes, "
        "is undetectably related to, or is noise relative to a legislative text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "amendment_direction": {
                "type": "string",
                "description": (
                    "One sentence describing what the legislative provision does "
                    "(e.g. 'establishes mandatory safety stock requirements', "
                    "'creates strategic project status for critical medicines', "
                    "'requires procurement criteria beyond price')."
                ),
            },
            "label": {
                "type": "string",
                "enum": LABELS,
            },
            "reasoning": {
                "type": "string",
                "description": "One or two sentences explaining the classification.",
            },
        },
        "required": ["amendment_direction", "label", "reasoning"],
    },
}

# ── Prompt variants ────────────────────────────────────────────────────────────

_SHARED_INTRO = """\
You are given a match between a lobbying organisation's pre-proposal feedback and a
provision in a European legislative proposal (article or recital).

You must assess whether the organisation's expressed position aligns with, opposes,
or has no clear directional relationship to what the legislative provision says.
Base your classification solely on the text provided — do not infer positions or
connections beyond what is explicitly stated.

You are given:
  • AMENDED TO  — the legislative provision text (article or recital) as it
                  appears in the proposal
  • ORG POSITION — what the organisation expressed in their feedback

Classify via the tool:
  ALIGNED       — the org's position pushes in the same direction as the provision
  OPPOSING      — the org's position pushes in the opposite direction
  UNDETECTABLE  — there is topical overlap between the org's feedback and the
                  provision, but the org's position does not directionally match
                  what the provision does
  NOISE         — use this when the org text contains no substantive advocacy
                  position (e.g. org headers, background descriptions of existing
                  law, administrative text), OR when the subjects are completely
                  different\
"""

_NOISE_BIAS_ADDENDUM = """\

IMPORTANT: Only label ALIGNED or OPPOSING when the org's position clearly and
specifically points in the same or opposite direction as what the provision does.
When there is topical overlap but no clear directional match, use UNDETECTABLE.
When there is no substantive topical connection, use NOISE.\
"""


def _match_body(row: dict) -> str:
    parts = ["\n---"]

    amended = (row.get("amended_text") or "").strip()
    parts.append(f"\nAMENDED TO:\n{amended}")

    # Prefer chunk_text (the retrieved segment) over full source_text so the
    # prompt matches what the production pipeline actually classifies.
    context_before = (row.get("context_before") or "").strip()
    org_text       = (row.get("chunk_text") or row.get("source_text") or "").strip()
    context_after  = (row.get("context_after") or "").strip()

    if context_before:
        parts.append(f"\nORG POSITION — PRECEDING CONTEXT:\n{context_before}")
    parts.append(f"\nORG POSITION — MATCHED CHUNK:\n{org_text}")
    if context_after:
        parts.append(f"\nORG POSITION — FOLLOWING CONTEXT:\n{context_after}")

    return "\n".join(parts)


def build_prompt_A(row: dict) -> str:
    """Variant A — includes organisation name."""
    org = (row.get("organisation") or "Unknown Organisation").strip()
    return f"Organisation: {org}\n\n{_SHARED_INTRO}{_match_body(row)}"


def build_prompt_B(row: dict) -> str:
    """Variant B — organisation name stripped."""
    return f"{_SHARED_INTRO}{_match_body(row)}"


def build_prompt_C(row: dict) -> str:
    """Variant C — with org name, explicit default-to-NOISE instruction."""
    org = (row.get("organisation") or "Unknown Organisation").strip()
    return f"Organisation: {org}\n\n{_SHARED_INTRO}{_NOISE_BIAS_ADDENDUM}{_match_body(row)}"


def build_prompt_D(row: dict) -> str:
    """Variant D — chain of thought: org name + explicit step-by-step instruction."""
    org = (row.get("organisation") or "Unknown Organisation").strip()
    cot = (
        "\n\nBefore classifying, think step by step:\n"
        "  1. What does the legislative provision actually do or require?\n"
        "  2. What specific position does the org express — what do they want?\n"
        "  3. Do those two things point in the same direction, opposite directions, "
        "or is there no clear directional relationship?"
    )
    return f"Organisation: {org}\n\n{_SHARED_INTRO}{cot}{_match_body(row)}"


def _match_body_split_context(row: dict) -> str:
    """Match body that shows context_before/after as distinct labelled sections."""
    parts = ["\n---"]

    amended = (row.get("amended_text") or "").strip()
    parts.append(f"\nAMENDED TO:\n{amended}")

    context_before = (row.get("context_before") or "").strip()
    chunk_text     = (row.get("chunk_text") or row.get("source_text") or "").strip()
    context_after  = (row.get("context_after") or "").strip()

    if context_before:
        parts.append(f"\nORG POSITION — PRECEDING CONTEXT:\n{context_before}")
    parts.append(f"\nORG POSITION — MATCHED CHUNK:\n{chunk_text}")
    if context_after:
        parts.append(f"\nORG POSITION — FOLLOWING CONTEXT:\n{context_after}")

    return "\n".join(parts)


def build_prompt_E(row: dict) -> str:
    """Variant E — split context: preceding/matched/following shown as separate sections."""
    org = (row.get("organisation") or "Unknown Organisation").strip()
    return f"Organisation: {org}\n\n{_SHARED_INTRO}{_match_body_split_context(row)}"


def build_prompt_F(row: dict) -> str:
    """Variant F — includes org name and procedure ID for legislative context."""
    org       = (row.get("organisation") or "Unknown Organisation").strip()
    procedure = (row.get("procedure_id") or "").strip()
    proc_line = f"Procedure: {procedure}\n" if procedure else ""
    return f"Organisation: {org}\n{proc_line}\n{_SHARED_INTRO}{_match_body(row)}"


PROMPT_BUILDERS: dict[str, callable] = {
    "A": build_prompt_A,
    "B": build_prompt_B,
    "C": build_prompt_C,
    "D": build_prompt_D,
    "E": build_prompt_E,
    "F": build_prompt_F,
}

# ── API call ───────────────────────────────────────────────────────────────────

def classify(client: anthropic.Anthropic, prompt: str, temperature: float) -> dict | None:
    try:
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            tools=[CLASSIFY_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
            timeout=60.0,
        )
        tool_block = next((b for b in msg.content if b.type == "tool_use"), None)
        if tool_block is None:
            return None
        return tool_block.input
    except (anthropic.APIError, anthropic.APITimeoutError) as exc:
        print(f"    API error: {exc}")
        return None


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(true_labels: list[str], pred_labels: list[str]) -> dict:
    p, r, f, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=LABELS, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(p[i]),
            "recall":    float(r[i]),
            "f1":        float(f[i]),
            "support":   int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }
    cm = confusion_matrix(true_labels, pred_labels, labels=LABELS)
    return {"per_class": per_class, "confusion_matrix": cm.tolist()}


def print_metrics(condition: str, metrics: dict) -> None:
    print(f"\n{'─'*64}")
    print(f"  {condition}")
    print(f"{'─'*64}")
    print(f"  {'Label':<12}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'N':>5}")
    for label, v in metrics["per_class"].items():
        print(f"  {label:<12}  {v['precision']:>7.3f}  {v['recall']:>7.3f}  {v['f1']:>7.3f}  {v['support']:>5}")
    print()
    print("  Confusion matrix (rows=true, cols=pred):")
    header = f"  {'':12}  " + "  ".join(f"{l:>10}" for l in LABELS)
    print(header)
    for i, row_label in enumerate(LABELS):
        vals = metrics["confusion_matrix"][i]
        print(f"  {row_label:<12}  " + "  ".join(f"{v:>10}" for v in vals))


def print_summary(all_results: list[dict]) -> None:
    print(f"\n{'='*64}")
    print("  SUMMARY — F1 per condition and label class")
    print(f"{'='*64}")
    print(f"  {'Condition':<30}  " + "  ".join(f"{l:>10}" for l in LABELS))
    print(f"  {'─'*30}  " + "  ".join("─" * 10 for _ in LABELS))
    for r in all_results:
        cond = r["condition"]
        f1s  = [f"{r['metrics']['per_class'][l]['f1']:>10.3f}" for l in LABELS]
        print(f"  {cond:<30}  " + "  ".join(f1s))

    # Highlight the ALIGNED over-classification risk
    print()
    print("  ALIGNED recall (should be high) vs NOISE precision (should be high)")
    print(f"  {'Condition':<30}  {'ALIGNED recall':>16}  {'NOISE prec':>12}")
    print(f"  {'─'*30}  {'─'*16}  {'─'*12}")
    for r in all_results:
        cond       = r["condition"]
        al_rec     = r["metrics"]["per_class"]["ALIGNED"]["recall"]
        noise_prec = r["metrics"]["per_class"]["NOISE"]["precision"]
        print(f"  {cond:<30}  {al_rec:>16.3f}  {noise_prec:>12.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM ALIGNED/OPPOSING/NOISE classifier against annotated ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ground-truth", required=True, help="Annotated CSV file path")
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.0, 0.5, 1.0],
        help="Temperatures to sweep (default: 0.0 0.5 1.0)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=["A", "B", "C", "D", "E"],
        help="Prompt variants to run (default: A B C D E)",
    )
    parser.add_argument(
        "--stability-n", type=int, default=20,
        help="Number of matches for stability test (default: 20)",
    )
    parser.add_argument(
        "--stability-k", type=int, default=5,
        help="Runs per match in stability test (default: 5)",
    )
    parser.add_argument(
        "--stability-temp", type=float, default=0.5,
        help="Temperature for stability test (default: 0.5)",
    )
    parser.add_argument(
        "--stability-variant", default="A",
        help="Prompt variant to use for stability test (default: A)",
    )
    parser.add_argument(
        "--output", default="evaluation/results/prompt_eval.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--cache", default="evaluation/results/prompt_eval_cache.json",
        help="Cache file to avoid re-running identical prompts",
    )
    parser.add_argument(
        "--rate-limit-sleep", type=float, default=0.4,
        help="Seconds to sleep between API calls (default: 0.4)",
    )
    args = parser.parse_args()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Load ground truth
    gt = pd.read_csv(args.ground_truth)
    required = {"true_label", "source_text", "amended_text"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(
            f"Ground truth CSV is missing required columns: {missing}\n"
            "Required: true_label, source_text, amended_text\n"
            "Optional: original_text, justification, organisation, source_type, source_date"
        )
    gt = gt[gt["true_label"].notna() & gt["source_text"].notna() & gt["amended_text"].notna()].copy()
    gt = gt.fillna("")
    gt["true_label"] = gt["true_label"].str.upper().str.strip()
    invalid = set(gt["true_label"]) - set(LABELS)
    if invalid:
        raise ValueError(f"Unknown labels in ground truth: {invalid}. Valid: {LABELS}")

    print(f"Ground truth: {len(gt)} annotated matches")
    print(f"Label distribution: {gt['true_label'].value_counts().to_dict()}")
    print(f"Variants: {args.variants}  |  Temperatures: {args.temperatures}")

    # Cache
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            print(f"Loaded {len(cache)} cached responses from {cache_path}")
        except json.JSONDecodeError:
            print("Cache file corrupted — starting fresh.")

    def save_cache() -> None:
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

    def cached_classify(key: str, prompt: str, temperature: float) -> dict | None:
        if key in cache:
            return cache[key]
        result = classify(client, prompt, temperature)
        if result is not None:
            cache[key] = result
            save_cache()
        time.sleep(args.rate_limit_sleep)
        return result

    # ── Accuracy sweep ─────────────────────────────────────────────────────────
    all_results: list[dict] = []

    for variant, temperature in product(args.variants, args.temperatures):
        if variant not in PROMPT_BUILDERS:
            print(f"Unknown variant '{variant}' — skipping.")
            continue

        builder   = PROMPT_BUILDERS[variant]
        condition = f"variant={variant}  temp={temperature}"
        print(f"\nRunning {condition} ({len(gt)} calls) ...")

        true_labels: list[str] = []
        pred_labels: list[str] = []
        rows: list[dict] = []

        for idx, row in gt.iterrows():
            prompt    = builder(row.to_dict())
            cache_key = f"acc_{variant}_{temperature}_{idx}"
            result    = cached_classify(cache_key, prompt, temperature)

            pred = (result.get("label") or "").upper() if result else None
            if pred not in LABELS:
                pred = "NOISE"  # safe fallback

            true_labels.append(row["true_label"])
            pred_labels.append(pred)

            rows.append({
                "idx":               idx,
                "true_label":        row["true_label"],
                "pred_label":        pred,
                "amendment_direction": result.get("amendment_direction") if result else None,
                "reasoning":         result.get("reasoning") if result else None,
            })

        metrics = compute_metrics(true_labels, pred_labels)
        print_metrics(condition, metrics)

        all_results.append({
            "condition":   condition,
            "variant":     variant,
            "temperature": temperature,
            "metrics":     metrics,
            "rows":        rows,
        })

    print_summary(all_results)

    # ── Stability test ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(
        f"  STABILITY TEST  "
        f"(variant={args.stability_variant}, n={args.stability_n}, "
        f"k={args.stability_k} runs each, temp={args.stability_temp})"
    )
    print(f"{'='*64}")

    sample_size   = min(args.stability_n, len(gt))
    stab_sample   = gt.sample(n=sample_size, random_state=42)
    stability_rows: list[dict] = []

    stab_builder = PROMPT_BUILDERS.get(args.stability_variant, build_prompt_A)
    for idx, row in stab_sample.iterrows():
        prompt     = stab_builder(row.to_dict())
        run_labels = []

        for run_i in range(args.stability_k):
            # Stability runs are intentionally NOT cached — they must be independent draws.
            result = classify(client, prompt, args.stability_temp)
            label  = (result.get("label") or "NOISE").upper() if result else "NOISE"
            if label not in LABELS:
                label = "NOISE"
            run_labels.append(label)
            time.sleep(args.rate_limit_sleep)

        stable  = len(set(run_labels)) == 1
        majority = max(set(run_labels), key=run_labels.count)
        stability_rows.append({
            "idx":        idx,
            "true_label": row["true_label"],
            "run_labels": run_labels,
            "stable":     stable,
            "majority":   majority,
        })
        tag = "stable  " if stable else "UNSTABLE"
        print(f"  Row {idx:>4}  [{tag}]  true={row['true_label']:<10}  runs={run_labels}")

    n_stable   = sum(r["stable"] for r in stability_rows)
    stable_pct = n_stable / len(stability_rows) if stability_rows else 1.0
    print(f"\n  Stable: {n_stable} / {len(stability_rows)}  ({stable_pct:.0%})")

    if stable_pct < 0.80:
        print(
            "  WARNING: < 80% stable.\n"
            "  Recommendations: reduce temperature, or use majority vote across k=3 runs.\n"
            "  Majority vote distribution on stable subset:\n"
            + "\n".join(
                f"    {l}: {sum(1 for r in stability_rows if r['stable'] and r['majority'] == l)}"
                for l in LABELS
            )
        )
    else:
        print("  OK: >= 80% stable at this temperature.")

    # Majority-vote accuracy on the stability sample (using the k runs as the prediction)
    stab_true  = [r["true_label"] for r in stability_rows]
    stab_pred  = [r["majority"]   for r in stability_rows]
    stab_metrics = compute_metrics(stab_true, stab_pred)
    print("\n  Majority-vote accuracy on stability sample:")
    print_metrics(f"stability majority-vote  temp={args.stability_temp}", stab_metrics)

    # ── Save output ────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "ground_truth_n":      len(gt),
        "label_distribution":  gt["true_label"].value_counts().to_dict(),
        "accuracy_evaluation": all_results,
        "stability": {
            "n_matches":     sample_size,
            "k_runs":        args.stability_k,
            "temperature":   args.stability_temp,
            "stable_pct":    stable_pct,
            "stable_n":      n_stable,
            "metrics":       stab_metrics,
            "rows":          stability_rows,
        },
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    main()
