"""RQ2 pooled analysis across all procedures.

Tests whether textual similarity (operationalised as alignment rate) between
lobbying positions and legislative outputs is predicted by the intensity of
disclosed access to decision-makers, and whether this effect is stronger at
the pre-proposal stage than the amendment stage.

Output: console report + analysis/rq2_pooled.json

Run: .venv/bin/python3.14 scripts/rq2_pooled_analysis.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 150)
pd.set_option("display.float_format", "{:.4f}".format)

# ── Procedure registry ────────────────────────────────────────────────────────
PROCEDURES = {
    "AI Act":        "analysis/2021:0106COD",
    "DSA":           "analysis/2020:0361COD",
    "DMA":           "analysis/2020:0374COD",
    "Digital Euro":  "analysis/2023:0212COD",
    "ELV":           "analysis/2023:0284COD",
    "CMA":           "analysis/2025:0102COD",
    "PPWR":          "analysis/2022:0396COD",
    "EMFA":          "analysis/2022:0277COD",
    "CSDDD":         "analysis/2022:0051COD",
}

OUT = Path("analysis/rq2_pooled.json")


# ── Loading + harmonisation ───────────────────────────────────────────────────
def _latest(base: Path, stage: str) -> Path | None:
    files = sorted(base.glob(f"{stage}_*_dyads.csv"), reverse=True)
    return files[0] if files else None


_RENAME = {
    "meetings_total": "total_meetings",
    "meetings_commission": "commission_meetings",
    "meetings_ep": "ep_meetings",
    "preproposal_meetings_total": "total_meetings",
    "preproposal_meetings_commission": "commission_meetings",
    "preproposal_meetings_ep": "ep_meetings",
}


def load(proc_name: str, base: str) -> pd.DataFrame:
    """Load both stages for one procedure into a single tidy frame."""
    p = Path(base)
    frames = []
    for stage in ("preproposal", "amendment"):
        f = _latest(p, stage)
        if f is None:
            continue
        df = pd.read_csv(f)
        df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns})
        df["stage"] = stage
        df["procedure"] = proc_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_all() -> pd.DataFrame:
    frames = [load(name, base) for name, base in PROCEDURES.items()]
    pooled = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    # Backstop: missing meeting cols → 0
    for c in ("total_meetings", "commission_meetings", "ep_meetings"):
        if c not in pooled.columns:
            pooled[c] = 0
        pooled[c] = pooled[c].fillna(0).astype(int)
    return pooled


# ── Helpers ───────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n{'═' * 90}")
    print(f"  {title}")
    print(f"{'═' * 90}\n")


def subsection(title):
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")


def descriptives(pooled: pd.DataFrame) -> pd.DataFrame:
    """Per (procedure, stage) summary stats."""
    rows = []
    for (proc, stage), g in pooled.groupby(["procedure", "stage"]):
        sub = g[g["label"].isin(["ALIGNED", "OPPOSING"])]
        org_meetings = g.groupby("organisation")["total_meetings"].first()
        rows.append({
            "procedure": proc,
            "stage": stage,
            "n_dyads": len(g),
            "n_orgs": g["organisation"].nunique(),
            "n_substantive": len(sub),
            "alignment_rate": (sub["label"] == "ALIGNED").mean() if len(sub) > 0 else np.nan,
            "noise_pct": (g["label"] == "NOISE").mean(),
            "mean_similarity": g["similarity_score"].mean(),
            "mean_meetings": org_meetings.mean(),
            "median_meetings": org_meetings.median(),
            "max_meetings": org_meetings.max(),
            "pct_zero_meetings": (org_meetings == 0).mean(),
        })
    return pd.DataFrame(rows).sort_values(["procedure", "stage"])


def _logit_with_cluster_se(df: pd.DataFrame, formula: str, cluster_col: str = "organisation"):
    """Logistic regression with cluster-robust SEs (cluster on organisation).

    Drops rows with NA in any modelled column.
    """
    needed_cols = set([cluster_col, "aligned"])
    for term in formula.replace("aligned ~", "").replace("+", " ").replace("*", " ").split():
        needed_cols.add(term.strip())
    work = df.dropna(subset=[c for c in needed_cols if c in df.columns]).copy()
    if len(work) < 30:
        return None, None
    try:
        model = smf.logit(formula=formula, data=work).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": work[cluster_col]},
        )
        return model, work
    except Exception as e:
        print(f"    LOGIT FAILED ({formula}): {e}")
        return None, None


def _print_logit(model, label: str):
    if model is None:
        print(f"  [{label}] model failed to fit")
        return
    print(f"\n  [{label}] n={int(model.nobs)}  pseudo-R²={model.prsquared:.3f}  llf={model.llf:.1f}")
    print("  " + "─" * 78)
    params, se, pvals = model.params, model.bse, model.pvalues
    or_ = np.exp(params)
    ci = model.conf_int().apply(np.exp)
    for name in params.index:
        sig = ("***" if pvals[name] < 0.001 else "**" if pvals[name] < 0.01
               else "*" if pvals[name] < 0.05 else "")
        print(f"    {name:<40s}  β={params[name]:>7.3f}  SE={se[name]:.3f}  "
              f"OR={or_[name]:>6.3f} [{ci.loc[name,0]:.3f},{ci.loc[name,1]:.3f}]  "
              f"p={pvals[name]:.4f} {sig}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pooled = load_all()
    pooled["aligned"] = (pooled["label"] == "ALIGNED").astype(int)
    pooled["substantive"] = pooled["label"].isin(["ALIGNED", "OPPOSING"])
    pooled["log_meetings"] = np.log1p(pooled["total_meetings"])
    pooled["any_meetings"] = (pooled["total_meetings"] > 0).astype(int)
    pooled["is_preproposal"] = (pooled["stage"] == "preproposal").astype(int)

    results = {"meta": {"n_dyads": len(pooled), "n_procedures": pooled["procedure"].nunique()}}

    # ════════════════════════════════════════════════════════════════════════════
    section("STEP 0 — DESCRIPTIVES BASELINE")
    desc = descriptives(pooled)
    print(desc.to_string(index=False))
    results["descriptives"] = desc.to_dict(orient="records")

    # ════════════════════════════════════════════════════════════════════════════
    section("STEP 1 — POOLED LABEL DISTRIBUTION & ALIGNMENT")

    subsection("Pooled label distribution by stage")
    by_stage = (
        pooled.groupby("stage")["label"]
        .value_counts(normalize=True).unstack().fillna(0) * 100
    )
    print(by_stage.round(1).to_string())

    subsection("Alignment rate (ALIGNED / substantive) by procedure × stage")
    align_tbl = (
        pooled[pooled["substantive"]]
        .groupby(["procedure", "stage"])["aligned"]
        .agg(["sum", "size", "mean"])
        .rename(columns={"sum": "aligned", "size": "n_substantive", "mean": "rate"})
    )
    print(align_tbl.round(3).to_string())
    results["alignment_table"] = align_tbl.round(4).reset_index().to_dict(orient="records")

    # Cross-stage paired test per procedure (Fisher's exact on contingency)
    subsection("Within-procedure: alignment shift preproposal → amendment")
    shift_rows = []
    for proc in pooled["procedure"].unique():
        sub = pooled[(pooled["procedure"] == proc) & (pooled["substantive"])]
        if sub["stage"].nunique() < 2:
            continue
        ct = pd.crosstab(sub["stage"], sub["label"])
        if ct.shape != (2, 2):
            continue
        odds, p = stats.fisher_exact(ct)
        pp_rate = sub[sub["stage"] == "preproposal"]["aligned"].mean()
        am_rate = sub[sub["stage"] == "amendment"]["aligned"].mean()
        shift_rows.append({
            "procedure": proc,
            "preprop_rate": round(pp_rate, 3),
            "amend_rate": round(am_rate, 3),
            "delta": round(am_rate - pp_rate, 3),
            "fisher_OR": round(odds, 3),
            "p": round(p, 4),
        })
    shift = pd.DataFrame(shift_rows).sort_values("delta")
    print(shift.to_string(index=False))
    results["preprop_amend_shift"] = shift.to_dict(orient="records")

    # ════════════════════════════════════════════════════════════════════════════
    section("STEP 2 — DYAD-LEVEL LOGISTIC REGRESSION (meetings → alignment)")
    print("  Subset:  ALIGNED ∪ OPPOSING dyads")
    print("  Outcome: aligned (1) vs opposing (0)")
    print("  Cluster: organisation (cluster-robust SEs)")
    print("  Procedure FE: included via dummy variables\n")

    sub = pooled[pooled["substantive"]].copy()

    m_main, _ = _logit_with_cluster_se(
        sub, "aligned ~ log_meetings + C(procedure)"
    )
    _print_logit(m_main, "Main effect: log_meetings (procedure FE)")
    results["model_main"] = (
        {k: float(v) for k, v in m_main.params.items()}
        if m_main is not None else None
    )

    m_stage, _ = _logit_with_cluster_se(
        sub, "aligned ~ log_meetings + C(stage) + C(procedure)"
    )
    _print_logit(m_stage, "Add stage FE")
    results["model_with_stage"] = (
        {k: float(v) for k, v in m_stage.params.items()}
        if m_stage is not None else None
    )

    # ════════════════════════════════════════════════════════════════════════════
    section("STEP 3 — STAGE × MEETINGS INTERACTION (direct RQ2 test)")
    print("  H_RQ2: meetings-effect on alignment is STRONGER at preproposal stage")
    print("  Coefficient on (log_meetings × is_preproposal): positive → support H_RQ2\n")

    m_int, _ = _logit_with_cluster_se(
        sub, "aligned ~ log_meetings * is_preproposal + C(procedure)"
    )
    _print_logit(m_int, "Interaction: log_meetings × is_preproposal")
    results["model_interaction"] = (
        {k: float(v) for k, v in m_int.params.items()}
        if m_int is not None else None
    )

    # Marginal-by-stage: meetings effect within each stage separately
    subsection("Stratified: meetings effect within each stage")
    stratified = {}
    for stg in ("preproposal", "amendment"):
        s = sub[sub["stage"] == stg]
        m, _ = _logit_with_cluster_se(s, "aligned ~ log_meetings + C(procedure)")
        _print_logit(m, f"{stg.upper()} only")
        stratified[stg] = (
            {k: float(v) for k, v in m.params.items()} if m is not None else None
        )
    results["model_stratified"] = stratified

    # ════════════════════════════════════════════════════════════════════════════
    section("STEP 4 — SENSITIVITY: drop low-N procedure × stage cells")
    LOW_N_CUTOFF = 80
    cells = (
        sub.groupby(["procedure", "stage"]).size().reset_index(name="n")
    )
    too_small = cells[cells["n"] < LOW_N_CUTOFF]
    print(f"  Cells dropped (<{LOW_N_CUTOFF} substantive dyads):")
    print(too_small.to_string(index=False) if len(too_small) > 0 else "  (none)")
    drop_keys = set(zip(too_small["procedure"], too_small["stage"]))
    sub_robust = sub[~sub.apply(lambda r: (r["procedure"], r["stage"]) in drop_keys, axis=1)]
    print(f"  Robust sample: {len(sub_robust)} substantive dyads (was {len(sub)})\n")

    m_robust, _ = _logit_with_cluster_se(
        sub_robust, "aligned ~ log_meetings * is_preproposal + C(procedure)"
    )
    _print_logit(m_robust, "Interaction model (robust subsample)")
    results["model_robust"] = (
        {k: float(v) for k, v in m_robust.params.items()}
        if m_robust is not None else None
    )

    # ════════════════════════════════════════════════════════════════════════════
    section("STEP 5 — SIMPSON'S CHECK: org-level vs dyad-level effect")
    print("  If dyad- and org-level analyses disagree in direction → aggregation matters\n")

    # Org-level aggregation: per (org, procedure, stage)
    org_lvl = (
        sub.groupby(["procedure", "stage", "organisation"])
        .agg(
            n_dyads=("aligned", "size"),
            n_aligned=("aligned", "sum"),
            meetings=("total_meetings", "first"),
        )
        .reset_index()
    )
    org_lvl["alignment_rate"] = org_lvl["n_aligned"] / org_lvl["n_dyads"]
    org_lvl["log_meetings"] = np.log1p(org_lvl["meetings"])

    subsection("Spearman correlation (org-level): meetings × alignment_rate")
    for stg in ("preproposal", "amendment", "ALL"):
        ss = org_lvl if stg == "ALL" else org_lvl[org_lvl["stage"] == stg]
        if len(ss) < 5:
            continue
        rho, p = stats.spearmanr(ss["meetings"], ss["alignment_rate"])
        print(f"  {stg:<13s} ρ = {rho:+.3f}  p = {p:.4f}  n_orgs = {len(ss)}")

    subsection("Org-level OLS: alignment_rate ~ log_meetings + procedure FE + stage")
    org_lvl["is_preproposal"] = (org_lvl["stage"] == "preproposal").astype(int)
    try:
        m_orglvl = smf.ols(
            "alignment_rate ~ log_meetings * is_preproposal + C(procedure)",
            data=org_lvl,
        ).fit(cov_type="cluster", cov_kwds={"groups": org_lvl["organisation"]})
        print(m_orglvl.summary().tables[1])
        results["model_org_level"] = {
            k: float(v) for k, v in m_orglvl.params.items()
        }
    except Exception as e:
        print(f"  org-level OLS failed: {e}")
        results["model_org_level"] = None

    # ════════════════════════════════════════════════════════════════════════════
    section("SUMMARY — what does the evidence say?")
    if m_int is not None:
        beta_meet = m_int.params.get("log_meetings", np.nan)
        beta_int = m_int.params.get("log_meetings:is_preproposal", np.nan)
        p_int = m_int.pvalues.get("log_meetings:is_preproposal", np.nan)
        print(f"  Pooled interaction model:")
        print(f"    β(log_meetings)            = {beta_meet:+.3f}  (amendment baseline effect)")
        print(f"    β(log_meetings × preprop)  = {beta_int:+.3f}  p = {p_int:.4f}")
        if not np.isnan(beta_int):
            stronger = "STRONGER" if beta_int > 0 else "WEAKER"
            sig = "significantly" if p_int < 0.05 else "not significantly"
            print(f"    → meeting-alignment effect is {sig} {stronger} at preproposal stage")

    OUT.parent.mkdir(exist_ok=True, parents=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUT}")


if __name__ == "__main__":
    main()
