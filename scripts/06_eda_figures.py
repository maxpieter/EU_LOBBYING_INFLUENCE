"""Cross-procedural EDA figures and Gini table — produces:

  * eda_distributions.pdf          (Fig. §4.4.1 — meeting-count histograms per stage)
  * eda_alignment_distribution.pdf (Fig. §4.4.1 — aligned-provision histograms per stage)
  * gini_coefficients.csv          (Table tab:gini-coefficients, §4.4.1)

Pools across the nine validation procedures using the dyad CSVs that
live next to this script's analysis/<proc>/ folders.

Usage:
    python scripts/eda_figures.py
    python scripts/eda_figures.py --out-dir "path/to/THESIS/images"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANA  = ROOT / "analysis"

PROCEDURES = {
    "AI Act":        "2021:0106COD",
    "DSA":           "2020:0361COD",
    "DMA":           "2020:0374COD",
    "Digital Euro":  "2023:0212COD",
    "ELV":           "2023:0284COD",
    "CMA":           "2025:0102COD",
    "PPWR":          "2022:0396COD",
    "EMFA":          "2022:0277COD",
    "CSDDD":         "2022:0051COD",
}


def _latest(p: Path, prefix: str) -> Path | None:
    cands = sorted(p.glob(f"{prefix}_*_dyads.csv"))
    return cands[-1] if cands else None


def _load(p: Path, stage: str, proc_label: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "preproposal_meetings_total" in df.columns:
        df = df.rename(columns={
            "preproposal_meetings_total":      "meetings_total",
            "preproposal_meetings_commission": "meetings_commission",
            "preproposal_meetings_ep":         "meetings_ep",
        })
    if "article_number" in df.columns:
        df["provision_id"] = df["article_number"].astype(str)
    elif "amendment_number" in df.columns:
        df["provision_id"] = df["amendment_number"].astype(str)
    df["stage"] = stage
    df["procedure"] = proc_label
    if "source_type" in df.columns:
        df = df[df["source_type"].isna() | (df["source_type"] == "hys_feedback")]
    return df


def load_all_dyads() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, slug in PROCEDURES.items():
        base = ANA / slug
        for stage in ("preproposal", "amendment"):
            p = _latest(base, stage)
            if p is not None:
                frames.append(_load(p, stage, label))
    return pd.concat(frames, ignore_index=True, sort=False)


def org_level(dy: pd.DataFrame) -> pd.DataFrame:
    return (
        dy.groupby(["organisation", "procedure", "stage"])
          .agg(
              meetings_total = ("meetings_total", "first"),
              count_aligned  = ("label", lambda x: (x == "ALIGNED").sum()),
              n_provisions   = ("provision_id", "nunique"),
          )
          .reset_index()
          .assign(meetings_total=lambda d: d["meetings_total"].fillna(0))
    )


def gini(values: np.ndarray) -> float:
    """Standard Gini coefficient (0 = perfect equality, 1 = max inequality).
    Returns 0.0 on empty / all-zero input."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0 or v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def plot_eda_distributions(orgs: pd.DataFrame, out: Path) -> None:
    """Per-organisation meeting-count distribution by stage.

    Unit of analysis is a unique organisation (summed across procedures within
    each stage). Filters to organisations with at least one disclosed meeting.
    Linear y-axis. Pre-proposal blue, amendment orange."""
    per_org = (orgs.groupby(["organisation", "stage"])["meetings_total"]
                 .sum().reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
    for ax, stage, color, title in [
        (axes[0], "preproposal", "#5b9bd5", "(a) Pre-proposal stage"),
        (axes[1], "amendment",   "#ed7d31", "(b) Amendment stage"),
    ]:
        m = per_org.loc[(per_org["stage"] == stage) & (per_org["meetings_total"] > 0),
                        "meetings_total"]
        if m.empty:
            ax.set_title(title); continue
        bins = np.arange(0, int(m.max()) + 3) - 0.5
        ax.hist(m, bins=bins, color=color, edgecolor="white", linewidth=0.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Disclosed meetings per organisation")
        ax.set_ylabel("Number of organisations")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "eda_distributions.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_alignment_distribution(orgs: pd.DataFrame, out: Path) -> None:
    """Aligned-dyad count distribution per organisation × procedure, by stage.

    Linear y-axis. Bars start at 1. Median dashed line + legend per panel."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
    for ax, stage, color, title in [
        (axes[0], "preproposal", "#ed7d31", "(a) Pre-proposal stage"),
        (axes[1], "amendment",   "#9b59b6", "(b) Amendment stage"),
    ]:
        n = orgs.loc[(orgs["stage"] == stage) & (orgs["count_aligned"] > 0), "count_aligned"]
        if n.empty:
            ax.set_title(title); continue
        bins = np.arange(1, int(n.max()) + 2) - 0.5
        ax.hist(n, bins=bins, color=color, edgecolor="white", linewidth=0.4)
        med = int(n.median())
        ax.axvline(med, color="black", lw=1.0, ls="--", label=f"median = {med}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("ALIGNED dyads per (organisation × procedure)")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "eda_alignment_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def gini_table(orgs: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for proc in PROCEDURES:
        for stage in ("preproposal", "amendment"):
            sub = orgs[(orgs["procedure"] == proc) & (orgs["stage"] == stage)]
            rows.append({
                "procedure":      proc,
                "stage":          stage,
                "n_orgs":         len(sub),
                "gini_meetings":  round(gini(sub["meetings_total"].values), 3),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "gini_coefficients.csv", index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",
                        default=str(Path(__file__).resolve().parent / "images"),
                        help="Where to write the PDFs (default: scripts/images)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dy = load_all_dyads()
    print(f"loaded {len(dy):,} dyads across {dy['procedure'].nunique()} procedures")
    orgs = org_level(dy)
    print(f"org × procedure × stage rows: {len(orgs):,}")

    plot_eda_distributions(orgs, out)
    plot_alignment_distribution(orgs, out)
    g = gini_table(orgs, out)
    print("\nGini coefficients (disclosed meetings):")
    print(g.pivot(index="procedure", columns="stage", values="gini_meetings"))
    print(f"\nwrote → {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
