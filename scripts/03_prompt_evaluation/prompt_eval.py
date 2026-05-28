#!/usr/bin/env python3
"""LLM classification evaluation for ALIGNED / OPPOSING / UNDETECTABLE / NOISE labels.

Tests the classifier against an annotated ground truth CSV across:
  - Prompt variant A: baseline, includes organisation name
  - Prompt variant B: organisation name stripped
  - Prompt variant D: chain-of-thought step-by-step (production pre-proposal prompt)
  - Prompt variant H: two-path ALIGNED + UNDETECTABLE as fallback
  - Prompt variant I: amendment pipeline — direction-first CoT (production amendment prompt)

Expected CSV columns (any extra columns are ignored):
  true_label         — one of ALIGNED, OPPOSING, UNDETECTABLE, NOISE
  source_text        — the org's meeting/feedback text
  amended_text       — the amendment text (amended_to)
  original_text      — original proposal text (can be blank for new insertions)
  justification      — amendment justification (optional)
  organisation       — org name (optional; used in variant A, ignored in B)
  source_type        — e.g. "commission_meeting" or "feedback" (optional)
  source_date        — ISO date string (optional)

Usage:
    python evaluation/prompt_eval.py --ground-truth path/to/annotated.csv

    # Run only specific variants / temperatures:
    python evaluation/prompt_eval.py --ground-truth gt.csv --variants A B --temperatures 0.0
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
            "provision_effect": {
                "type": "string",
                "description": (
                    "One sentence describing what the legislative provision establishes or requires "
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
        "required": ["provision_effect", "label", "reasoning"],
    },
}

# ── Prompt variants ────────────────────────────────────────────────────────────

_SHARED_INTRO = """\
You are given a match between a lobbying organisation's pre-proposal feedback and a
provision in a European legislative proposal (article or recital).

You must assess whether the organisation's expressed position supports, opposes,
or has no clear relationship to what the legislative provision establishes.
Base your classification solely on the text provided — do not infer positions or
connections beyond what is explicitly stated.

You are given:
  • AMENDED TO  — the legislative provision text (article or recital) as it
                  appears in the proposal
  • ORG POSITION — what the organisation expressed in their feedback

Classify via the tool:
  ALIGNED       — the org's position supports or is consistent with what the provision establishes
  OPPOSING      — the org's position contradicts or pushes back against what the provision establishes
  UNDETECTABLE  — there is topical overlap between the org's feedback and the
                  provision, but no clear positional relationship can be established
  NOISE         — use this when the org text contains no substantive advocacy
                  position (e.g. org headers, background descriptions of existing
                  law, administrative text), OR when the subjects are completely
                  different\
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


def build_prompt_D(row: dict) -> str:
    """Variant D — chain of thought: org name + explicit step-by-step instruction."""
    org = (row.get("organisation") or "Unknown Organisation").strip()
    cot = (
        "\n\nBefore classifying, think step by step:\n"
        "  1. What does the legislative provision actually do or require?\n"
        "  2. What specific position does the org express — what do they want?\n"
        "  3. Does the org's position support, contradict, or have no clear relationship "
        "to what the provision establishes?"
    )
    return f"Organisation: {org}\n\n{_SHARED_INTRO}{cot}{_match_body(row)}"



_H_INTRO = """\
You are given a match between a lobbying organisation's pre-proposal text and a
provision in a European Commission legislative proposal (article or recital).

Your task is to assess whether the provision responds to what the organisation
advocated for — not merely whether both discuss the same topic.
Base your classification solely on the text provided. Do not infer positions or
connections beyond what is explicitly stated.

You are given:
  • LEGISLATIVE PROVISION — the article or recital as it appears in the proposal
  • ORG POSITION          — what the organisation expressed before the proposal
                            was tabled (feedback submission or commissioner meeting)

Classify via the tool:
  ALIGNED       — there is a clear tie between what the org advocates for and what the
                  provision does. Two paths to ALIGNED:
                  (a) The org makes a specific ask and the provision delivers it (or more
                      of it in the same direction).
                  (b) The org advocates for action/initiative in a specific direction, and
                      the provision takes concrete action in that direction — not merely
                      defines or neutrally describes the area. If you can argue a clear
                      connection between the org's position and the provision's action,
                      classify ALIGNED.
  OPPOSING      — the org explicitly argues against what the provision establishes,
                  or the provision does the opposite of what the org asked for.
  UNDETECTABLE  — the org has a specific advocacy position related to the provision's
                  subject, but it is genuinely unclear whether the provision satisfies,
                  contradicts, or ignores it. Use this when you can see a real connection
                  but cannot confidently determine direction. If uncertain → fall back here.
  NOISE         — the org text contains no substantive advocacy position on this topic:
                  boilerplate, background descriptions of existing law, administrative text,
                  general endorsements without a specific stance
                  (e.g. "we support EU action on medicines", "we welcome the initiative"),
                  OR the subjects are unrelated.

Critical distinctions:
  • An org that argues a mechanism should NOT EXIST is OPPOSING the provision that
    creates it, even if the provision also includes safeguards the org likes.
  • A provision that merely defines a term, lists background context, or describes
    the current state of affairs is NOT taking action — path (b) for ALIGNED does not
    apply to purely definitional or recital provisions that take no concrete stance.
  • General values ("we support supply chain resilience") with no specific direction
    → NOISE. A position needs to be specific enough that you can evaluate whether the
    provision responds to it.
  • UNDETECTABLE is the fallback when a genuine connection exists but direction is
    ambiguous — not when the org text is simply vague or general (that is NOISE).\
"""

_H_COT = (
    "\n\nBefore classifying, think step by step:\n"
    "  1. What does the provision specifically establish, require, or do? "
    "State the concrete action — or note if it only defines/describes.\n"
    "  2. Does the org text express a specific advocacy position on this subject "
    "(not just a general value or endorsement)?\n"
    "     If only general support/concern with no directional stance → NOISE (stop here).\n"
    "  3. Counterfactual test: could an org with the OPPOSITE general stance on this topic "
    "produce a text that would also match this provision in the same direction?\n"
    "     If yes — the org's position is too broad to constitute specific alignment → NOISE.\n"
    "     Example: 'we support supply chain resilience' would match any supply chain provision "
    "regardless of the org's actual position, so it fails this test.\n"
    "  4. Is there a clear and specific tie between the org's position and the provision's action?\n"
    "     • Clear specific tie, provision acts in the direction the org advocated → ALIGNED\n"
    "     • The org opposes this mechanism or asks for the opposite → OPPOSING\n"
    "     • Real connection exists but direction is genuinely ambiguous → UNDETECTABLE"
)


def _match_body_H(row: dict) -> str:
    """Match body using LEGISLATIVE PROVISION label to match production pipeline wording."""
    parts = ["\n---"]
    amended = (row.get("amended_text") or "").strip()
    parts.append(f"\nLEGISLATIVE PROVISION:\n{amended}")

    context_before = (row.get("context_before") or "").strip()
    org_text       = (row.get("chunk_text") or row.get("source_text") or "").strip()
    context_after  = (row.get("context_after") or "").strip()

    if context_before:
        parts.append(f"\nORG POSITION — PRECEDING CONTEXT:\n{context_before}")
    parts.append(f"\nORG POSITION — MATCHED CHUNK:\n{org_text}")
    if context_after:
        parts.append(f"\nORG POSITION — FOLLOWING CONTEXT:\n{context_after}")

    return "\n".join(parts)


def build_prompt_H(row: dict) -> str:
    """Variant H — two-path ALIGNED (specific ask OR clear directional tie), UNDETECTABLE as fallback."""
    org = (row.get("organisation") or "Unknown Organisation").strip()
    return f"Organisation: {org}\n\n{_H_INTRO}{_H_COT}{_match_body_H(row)}"



_AMD_INTRO = """\
You are given a match between a lobbying organisation's pre-proposal text and a
parliamentary amendment to a European Commission legislative proposal.

Your task is to assess whether the organisation's expressed position advocates
for the change the amendment makes — not merely whether they discuss the same topic.

You are given:
  • ORIGINAL TEXT  — the provision as it appeared in the Commission proposal
                     (omitted if the amendment is a new insertion)
  • AMENDED TO     — the text as revised by the amendment
  • JUSTIFICATION  — the amendment's stated rationale (may be absent)
  • ORG POSITION   — what the organisation expressed before the proposal was tabled

The organisation does not need to mention the specific article. What matters is
whether their expressed position advocates for the direction of change the amendment
makes to the original text.

Classify via the tool:
  ALIGNED       — the amendment moves the original text in the direction the org
                  advocated for. The org's position must be specific enough that
                  you can say the amendment responds to it.
  OPPOSING      — the amendment moves the original text in the opposite direction
                  from what the org advocated for, or the org explicitly opposes
                  what the amendment introduces.
  UNDETECTABLE  — the org has a specific position related to this area, but it is
                  genuinely unclear whether the amendment's direction satisfies,
                  contradicts, or ignores it. Use when a real connection exists
                  but direction is ambiguous. Fall back here when uncertain.
  NOISE         — the org text contains no substantive advocacy position on this
                  topic: boilerplate, background descriptions, general endorsements
                  without a specific directional stance
                  (e.g. "we support this initiative", "we welcome EU action"),
                  OR the subjects are unrelated.

Critical distinctions:
  • The direction of the amendment is what matters — not whether both texts discuss
    the same subject. Determine first what the amendment ADDS, REMOVES, STRENGTHENS,
    or WEAKENS relative to the original, then evaluate the org's position against that.
  • General support for the policy area → NOISE, not UNDETECTABLE.
  • Counterfactual test: could an org with the OPPOSITE stance produce text that
    also matches this amendment in the same direction? If yes, the org's position
    is too broad → NOISE.
  • UNDETECTABLE is the fallback when a genuine directional connection exists but
    the relationship is ambiguous. It is NOT the default when the org text is vague.\
"""

_AMD_COT = (
    "\n\nBefore classifying, think step by step:\n"
    "  1. What does the ORIGINAL TEXT establish? If no original, note this is a new insertion.\n"
    "  2. What does the AMENDMENT change it to? What specifically does it add, remove, "
    "strengthen, weaken, or clarify?\n"
    "  3. State the DIRECTION of change in one sentence "
    "(e.g. 'the amendment makes X mandatory where it was optional', "
    "'the amendment removes the obligation to Y').\n"
    "  4. Does the org text express a specific advocacy position on this subject — "
    "not just general support or concern?\n"
    "     If only general endorsement with no directional stance → NOISE (stop here).\n"
    "  5. Counterfactual: could an org with the opposite position also match this amendment "
    "in the same direction? If yes → NOISE.\n"
    "  6. Does the org's position advocate for pushing the original text in this direction?\n"
    "     • Yes, the amendment goes where the org wanted → ALIGNED\n"
    "     • The org wanted the opposite direction → OPPOSING\n"
    "     • Real connection but genuinely unclear whether this direction satisfies the org → UNDETECTABLE"
)


def _match_body_amendment(row: dict) -> str:
    parts = ["\n---"]
    original = (row.get("original_text") or "").strip()
    amended  = (row.get("amended_text")  or "").strip()
    justif   = (row.get("justification") or "").strip()

    if original:
        parts.append(f"\nORIGINAL TEXT:\n{original}")
    else:
        parts.append("\nORIGINAL TEXT:\n(new insertion — no original text)")
    parts.append(f"\nAMENDED TO:\n{amended}")
    if justif:
        parts.append(f"\nJUSTIFICATION:\n{justif}")

    context_before = (row.get("context_before") or "").strip()
    chunk_text     = (row.get("chunk_text") or row.get("source_text") or "").strip()
    context_after  = (row.get("context_after") or "").strip()

    if context_before:
        parts.append(f"\nORG POSITION — PRECEDING CONTEXT:\n{context_before}")
    parts.append(f"\nORG POSITION — MATCHED CHUNK:\n{chunk_text}")
    if context_after:
        parts.append(f"\nORG POSITION — FOLLOWING CONTEXT:\n{context_after}")

    return "\n".join(parts)


def build_prompt_I(row: dict) -> str:
    """Variant I — amendment pipeline: direction-first CoT + noise guards."""
    org = (row.get("organisation") or "Unknown Organisation").strip()
    return f"Organisation: {org}\n\n{_AMD_INTRO}{_AMD_COT}{_match_body_amendment(row)}"


PROMPT_BUILDERS: dict[str, callable] = {
    "A": build_prompt_A,
    "B": build_prompt_B,
    "D": build_prompt_D,
    "H": build_prompt_H,
    "I": build_prompt_I,
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
        "--variants", nargs="+", default=["A", "B", "D", "H"],
        help="Prompt variants to run (default: A B D H)",
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
        "--no-stability", action="store_true",
        help="Skip the stability test entirely",
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
                "idx":           idx,
                "true_label":    row["true_label"],
                "pred_label":    pred,
                "provision_effect": result.get("provision_effect") if result else None,
                "reasoning":     result.get("reasoning") if result else None,
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
    if args.no_stability:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "ground_truth_n":      len(gt),
            "label_distribution":  gt["true_label"].value_counts().to_dict(),
            "accuracy_evaluation": all_results,
        }
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\nFull results saved → {out_path}")
        return
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
    if stab_true and stab_pred:
        stab_metrics = compute_metrics(stab_true, stab_pred)
        print("\n  Majority-vote accuracy on stability sample:")
        print_metrics(f"stability majority-vote  temp={args.stability_temp}", stab_metrics)
    else:
        stab_metrics = {}
        print("\n  Stability sample empty — skipped.")

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
