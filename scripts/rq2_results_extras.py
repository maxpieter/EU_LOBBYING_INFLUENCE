"""Follow-up to rq2_pooled_analysis.py:
1. Predicted probabilities of alignment at meetings = {0, 1, 5, 10, 20}, by stage
2. Publication-quality coefficient plot of the interaction model

Outputs:
  analysis/rq2_predicted_probs.json
  analysis/rq2_predicted_probs.csv
  analysis/rq2_coefficient_plot.png
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

_RENAME = {
    "meetings_total": "total_meetings",
    "meetings_commission": "commission_meetings",
    "meetings_ep": "ep_meetings",
    "preproposal_meetings_total": "total_meetings",
    "preproposal_meetings_commission": "commission_meetings",
    "preproposal_meetings_ep": "ep_meetings",
}


def _latest(base: Path, stage: str) -> Path | None:
    files = sorted(base.glob(f"{stage}_*_dyads.csv"), reverse=True)
    return files[0] if files else None


def load_all() -> pd.DataFrame:
    frames = []
    for name, base in PROCEDURES.items():
        p = Path(base)
        for stage in ("preproposal", "amendment"):
            f = _latest(p, stage)
            if f is None:
                continue
            df = pd.read_csv(f)
            df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns})
            df["stage"] = stage
            df["procedure"] = name
            frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    for c in ("total_meetings", "commission_meetings", "ep_meetings"):
        if c not in pooled.columns:
            pooled[c] = 0
        pooled[c] = pooled[c].fillna(0).astype(int)
    return pooled


def fit_interaction(pooled: pd.DataFrame):
    sub = pooled[pooled["label"].isin(["ALIGNED", "OPPOSING"])].copy()
    sub["aligned"] = (sub["label"] == "ALIGNED").astype(int)
    sub["log_meetings"] = np.log1p(sub["total_meetings"])
    sub["is_preproposal"] = (sub["stage"] == "preproposal").astype(int)
    m = smf.logit(
        "aligned ~ log_meetings * is_preproposal + C(procedure)",
        data=sub,
    ).fit(disp=False, cov_type="cluster", cov_kwds={"groups": sub["organisation"]})
    return m, sub


def predicted_probs(model, sub: pd.DataFrame) -> pd.DataFrame:
    """Predicted P(ALIGNED) at meetings = {0,1,5,10,20} for each stage.

    Averages over procedure FE by computing predictions at each procedure and
    then taking the unweighted mean across procedures (representative procedure).
    """
    meeting_levels = [0, 1, 5, 10, 20]
    procedures = sub["procedure"].unique()
    rows = []
    for stage_label, is_pp in [("preproposal", 1), ("amendment", 0)]:
        for m_count in meeting_levels:
            preds = []
            for proc in procedures:
                grid = pd.DataFrame({
                    "log_meetings": [np.log1p(m_count)],
                    "is_preproposal": [is_pp],
                    "procedure": [proc],
                })
                p = model.predict(grid).iloc[0]
                preds.append(p)
            mean_p = float(np.mean(preds))
            # 95% CI via delta method on the linear predictor
            # Use mean linear predictor and SE across procedures
            xlevels = []
            for proc in procedures:
                # Build the design row manually (matches model.model.exog_names)
                exog_names = model.model.exog_names
                row = np.zeros(len(exog_names))
                for i, n in enumerate(exog_names):
                    if n == "Intercept":
                        row[i] = 1
                    elif n == "log_meetings":
                        row[i] = np.log1p(m_count)
                    elif n == "is_preproposal":
                        row[i] = is_pp
                    elif n == "log_meetings:is_preproposal":
                        row[i] = np.log1p(m_count) * is_pp
                    elif n == f"C(procedure)[T.{proc}]":
                        row[i] = 1
                xlevels.append(row)
            X = np.array(xlevels)
            lin = X @ model.params.values
            se = np.sqrt(np.diag(X @ model.cov_params().values @ X.T))
            lo = 1 / (1 + np.exp(-(lin - 1.96 * se)))
            hi = 1 / (1 + np.exp(-(lin + 1.96 * se)))
            rows.append({
                "stage": stage_label,
                "meetings": m_count,
                "p_aligned": round(mean_p, 4),
                "p_aligned_lo95": round(float(np.mean(lo)), 4),
                "p_aligned_hi95": round(float(np.mean(hi)), 4),
            })
    return pd.DataFrame(rows)


def coefficient_plot(model, out_path: Path):
    """Forest plot of model coefficients with 95% CIs (odds-ratio scale)."""
    params = model.params
    ci = model.conf_int()
    pvals = model.pvalues
    # Exclude intercept and procedure FEs from the focal plot — show meeting + stage effects
    focal = [
        ("log_meetings", "Meetings (log) — amendment baseline"),
        ("is_preproposal", "Preproposal stage (vs amendment)"),
        ("log_meetings:is_preproposal", "Meetings × Preproposal (interaction)"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    y = list(range(len(focal)))[::-1]
    for i, (key, label) in enumerate(focal):
        b = params[key]
        lo, hi = ci.loc[key]
        or_ = np.exp(b)
        or_lo, or_hi = np.exp(lo), np.exp(hi)
        p = pvals[key]
        color = "#1f77b4" if p < 0.05 else "#888"
        ax.errorbar(
            or_, y[i],
            xerr=[[or_ - or_lo], [or_hi - or_]],
            fmt="o", color=color, capsize=4, lw=1.5, markersize=8,
        )
        sig = " *" if p < 0.05 else ""
        ax.text(or_hi + 0.05, y[i], f"OR={or_:.2f}{sig}", va="center", fontsize=9)

    ax.axvline(1, ls="--", color="black", lw=0.8, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([lbl for _, lbl in focal], fontsize=10)
    ax.set_xlabel("Odds ratio (95% CI)")
    ax.set_xscale("log")
    ax.set_title("RQ2: predictors of provision-level alignment (n=3,250 substantive dyads)",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved coefficient plot → {out_path}")


def main():
    pooled = load_all()
    model, sub = fit_interaction(pooled)
    print(f"  Fitted on {int(model.nobs)} substantive dyads")
    print(f"  Pseudo R² = {model.prsquared:.3f}\n")

    pred = predicted_probs(model, sub)
    print("Predicted P(ALIGNED) by stage × meetings:")
    print(pred.to_string(index=False))

    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)
    pred.to_csv(out_dir / "rq2_predicted_probs.csv", index=False)
    with open(out_dir / "rq2_predicted_probs.json", "w") as f:
        json.dump(pred.to_dict(orient="records"), f, indent=2)

    coefficient_plot(model, out_dir / "rq2_coefficient_plot.png")


if __name__ == "__main__":
    main()
