"""RQ2 robustness and extensions.

Six follow-up analyses appended to rq2_pooled_analysis.py:

  1. Lobbyist-type stratification (industry / NGO / trade-assoc / academic / other)
  2. Commission-side vs EP-side meeting decomposition
  3. Continuous similarity_score as outcome (LLM-classification-independent test)
  4. Organisation fixed effects (within-org test of access hypothesis)
  5. Access concentration (Gini per procedure × stage; high vs low concentration interaction)
  6. Procedure-salience interaction (HYS volume + amendment volume as salience proxies)

Outputs to analysis/rq2_robustness.json + console report.
"""

import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from dotenv import load_dotenv
from scipy import stats
from supabase import create_client

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 150)
pd.set_option("display.float_format", "{:.4f}".format)

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
PROC_IDS = {
    "AI Act":       "2021/0106(COD)",
    "DSA":          "2020/0361(COD)",
    "DMA":          "2020/0374(COD)",
    "Digital Euro": "2023/0212(COD)",
    "ELV":          "2023/0284(COD)",
    "CMA":          "2025/0102(COD)",
    "PPWR":         "2022/0396(COD)",
    "EMFA":         "2022/0277(COD)",
    "CSDDD":        "2022/0051(COD)",
}

_RENAME = {
    "meetings_total": "total_meetings",
    "meetings_commission": "commission_meetings",
    "meetings_ep": "ep_meetings",
    "preproposal_meetings_total": "total_meetings",
    "preproposal_meetings_commission": "commission_meetings",
    "preproposal_meetings_ep": "ep_meetings",
}


# ── Data loading ──────────────────────────────────────────────────────────────
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


def fetch_org_metadata(client, names: list[str]) -> pd.DataFrame:
    """Pull organization_type and industry_sector from Supabase for given org names."""
    rows = []
    for i in range(0, len(names), 100):
        batch = list(names[i:i + 100])
        try:
            r = (
                client.table("organizations")
                .select("name, organization_type, industry_sector")
                .in_("name", batch)
                .execute()
            )
            rows.extend(r.data or [])
        except Exception:
            continue
    df = pd.DataFrame(rows).drop_duplicates(subset=["name"], keep="first")
    return df


# ── Modelling helpers ─────────────────────────────────────────────────────────
def fit_logit_cluster(df, formula, cluster_col="organisation"):
    df = df.dropna(subset=["aligned"]).copy()
    try:
        return smf.logit(formula, data=df).fit(
            disp=False, cov_type="cluster",
            cov_kwds={"groups": df[cluster_col]},
            maxiter=200,
        )
    except Exception as e:
        print(f"    LOGIT FAILED ({formula}): {e}")
        return None


def fit_ols_cluster(df, formula, cluster_col="organisation"):
    df = df.dropna(subset=[formula.split("~")[0].strip()]).copy()
    try:
        return smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[cluster_col]},
        )
    except Exception as e:
        print(f"    OLS FAILED ({formula}): {e}")
        return None


def coef_dict(model):
    if model is None:
        return None
    out = {}
    for k in model.params.index:
        out[k] = {
            "beta": float(model.params[k]),
            "se": float(model.bse[k]),
            "p": float(model.pvalues[k]),
        }
    return out


def print_focal(model, focal_terms, label):
    print(f"\n  [{label}] n = {int(model.nobs) if model else 0}")
    if model is None:
        return
    print(f"  pseudo-R² / R² = {getattr(model, 'prsquared', getattr(model, 'rsquared', 0)):.3f}")
    print("  " + "─" * 78)
    for term in focal_terms:
        if term not in model.params.index:
            continue
        b = model.params[term]
        se = model.bse[term]
        p = model.pvalues[term]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        or_ = np.exp(b) if hasattr(model, "prsquared") else None
        or_str = f"  OR={or_:.3f}" if or_ is not None else ""
        print(f"    {term:<42s} β={b:+.3f}  SE={se:.3f}{or_str}  p={p:.4f} {sig}")


def section(title):
    print(f"\n{'═' * 90}")
    print(f"  {title}")
    print(f"{'═' * 90}")


def subsection(title):
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")


def gini(values):
    arr = np.array(sorted(v for v in values if not np.isnan(v) and v >= 0))
    n = len(arr)
    if n < 2 or arr.sum() == 0:
        return 0.0
    cumvals = np.cumsum(arr)
    return float((2 * np.sum((np.arange(1, n + 1)) * arr) - (n + 1) * cumvals[-1]) / (n * cumvals[-1]))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    pooled = load_all()
    pooled["aligned"] = (pooled["label"] == "ALIGNED").astype(int)
    pooled["substantive"] = pooled["label"].isin(["ALIGNED", "OPPOSING"])
    pooled["log_meetings"] = np.log1p(pooled["total_meetings"])
    pooled["log_comm_meetings"] = np.log1p(pooled["commission_meetings"])
    pooled["log_ep_meetings"] = np.log1p(pooled["ep_meetings"])
    pooled["is_preproposal"] = (pooled["stage"] == "preproposal").astype(int)
    sub = pooled[pooled["substantive"]].copy()

    results = {"meta": {"n_dyads": len(pooled), "n_substantive": len(sub)}}

    # ════════════════════════════════════════════════════════════════════════════
    section("1 — LOBBYIST-TYPE STRATIFICATION")
    print("  Pulling organization_type from Supabase…")
    org_meta = fetch_org_metadata(client, sub["organisation"].unique().tolist())
    if org_meta.empty:
        print("  No organisation metadata fetched — skipping.")
        results["lobbyist_type"] = None
    else:
        # Simplify org types into a few buckets
        def bucket(t):
            if not isinstance(t, str):
                return "Unknown"
            t = t.lower()
            if "trade" in t or "association" in t or "federation" in t or "chamber" in t:
                return "Trade/Association"
            if "ngo" in t or "non-governmental" in t or "civil society" in t:
                return "NGO"
            if "company" in t or "in-house" in t or "consultancy" in t or "law" in t:
                return "Industry"
            if "academic" in t or "research" in t or "think tank" in t:
                return "Academic/Research"
            if "public" in t or "authority" in t:
                return "Public"
            return "Other"

        org_meta["bucket"] = org_meta["organization_type"].apply(bucket)
        merged = sub.merge(org_meta[["name", "bucket"]],
                           left_on="organisation", right_on="name", how="left")
        merged["bucket"] = merged["bucket"].fillna("Unknown")

        subsection("Org-type distribution")
        print(merged["bucket"].value_counts().to_string())

        subsection("Alignment rate by org-type × stage")
        tbl = (
            merged.groupby(["bucket", "stage"])["aligned"]
            .agg(["mean", "size"])
            .rename(columns={"mean": "alignment_rate", "size": "n"})
        )
        print(tbl.round(3).to_string())
        results["alignment_by_type"] = tbl.round(4).reset_index().to_dict(orient="records")

        subsection("Interaction model with org-type × meetings (industry as ref)")
        # Re-fit interaction, adding type buckets and their interaction with meetings
        focus_types = ["Industry", "Trade/Association", "NGO"]
        merged["bucket_f"] = merged["bucket"].where(merged["bucket"].isin(focus_types), "Other")
        m_type = fit_logit_cluster(
            merged,
            "aligned ~ log_meetings * is_preproposal + log_meetings * C(bucket_f) + C(procedure)",
        )
        print_focal(
            m_type,
            ["log_meetings", "is_preproposal", "log_meetings:is_preproposal",
             "C(bucket_f)[T.NGO]", "C(bucket_f)[T.Trade/Association]",
             "log_meetings:C(bucket_f)[T.NGO]", "log_meetings:C(bucket_f)[T.Trade/Association]"],
            "Type × Meetings interaction",
        )
        results["model_type_interaction"] = coef_dict(m_type)

    # ════════════════════════════════════════════════════════════════════════════
    section("2 — COMMISSION vs EP MEETING DECOMPOSITION")
    print("  Replacing total meetings with separate Commission/EP terms")
    print("  Hypothesis: EP meetings drive amendment-stage alignment,")
    print("              Commission meetings drive pre-proposal-stage alignment\n")

    m_decomp = fit_logit_cluster(
        sub,
        "aligned ~ log_comm_meetings * is_preproposal + log_ep_meetings * is_preproposal + C(procedure)",
    )
    print_focal(
        m_decomp,
        ["log_comm_meetings", "log_ep_meetings", "is_preproposal",
         "log_comm_meetings:is_preproposal", "log_ep_meetings:is_preproposal"],
        "Commission + EP meetings × Stage",
    )
    results["model_meeting_decomp"] = coef_dict(m_decomp)

    subsection("Stratified by stage (Comm + EP)")
    decomp_strat = {}
    for stg in ("preproposal", "amendment"):
        s = sub[sub["stage"] == stg]
        m = fit_logit_cluster(s, "aligned ~ log_comm_meetings + log_ep_meetings + C(procedure)")
        print_focal(m, ["log_comm_meetings", "log_ep_meetings"], f"{stg.upper()} only")
        decomp_strat[stg] = coef_dict(m)
    results["model_decomp_stratified"] = decomp_strat

    # ════════════════════════════════════════════════════════════════════════════
    section("3 — CONTINUOUS SIMILARITY SCORE AS OUTCOME")
    print("  Tests whether the meeting × stage finding is independent of LLM labelling")
    print("  Outcome = cosine similarity (raw retrieval signal)\n")

    m_sim_full = fit_ols_cluster(
        pooled, "similarity_score ~ log_meetings * is_preproposal + C(procedure)"
    )
    print_focal(
        m_sim_full,
        ["log_meetings", "is_preproposal", "log_meetings:is_preproposal"],
        "Similarity ~ meetings × stage (all dyads)",
    )
    results["model_similarity_all"] = coef_dict(m_sim_full)

    m_sim_sub = fit_ols_cluster(
        sub, "similarity_score ~ log_meetings * is_preproposal + C(procedure)"
    )
    print_focal(
        m_sim_sub,
        ["log_meetings", "is_preproposal", "log_meetings:is_preproposal"],
        "Similarity ~ meetings × stage (substantive subset only)",
    )
    results["model_similarity_substantive"] = coef_dict(m_sim_sub)

    # ════════════════════════════════════════════════════════════════════════════
    section("4 — ORGANISATION FIXED EFFECTS (within-org test)")
    print("  Adds organisation FE — does a given organisation gain more alignment")
    print("  when it has more meetings (within-org effect, controlling for baseline)?\n")

    # Keep only orgs with variation in meetings across stages/procedures
    sub_var = sub.copy()
    org_counts = sub_var.groupby("organisation").size()
    big_orgs = org_counts[org_counts >= 10].index
    sub_var = sub_var[sub_var["organisation"].isin(big_orgs)]
    print(f"  Restricted to orgs with ≥10 substantive dyads: {sub_var['organisation'].nunique()} orgs, {len(sub_var)} dyads\n")

    # Use linear probability model with org FE (logit with org FE is unstable for many orgs)
    m_orgfe = fit_ols_cluster(
        sub_var,
        "aligned ~ log_meetings * is_preproposal + C(procedure) + C(organisation)",
    )
    if m_orgfe is not None:
        focal = ["log_meetings", "is_preproposal", "log_meetings:is_preproposal"]
        print_focal(m_orgfe, focal, "Within-org LPM (org FE)")
        results["model_org_fe"] = {k: coef_dict(m_orgfe).get(k) for k in focal}
    else:
        results["model_org_fe"] = None

    # ════════════════════════════════════════════════════════════════════════════
    section("5 — ACCESS CONCENTRATION (GINI)")
    print("  Gini of meetings per organisation, by procedure × stage")
    print("  Higher Gini → more concentrated access\n")

    gini_rows = []
    for (proc, stage), g in pooled.groupby(["procedure", "stage"]):
        org_meet = g.groupby("organisation")["total_meetings"].first()
        gini_rows.append({
            "procedure": proc,
            "stage": stage,
            "n_orgs": int(g["organisation"].nunique()),
            "gini": round(gini(org_meet), 3),
            "share_top1": round((org_meet.nlargest(1).sum() / max(org_meet.sum(), 1)), 3),
            "share_top5": round((org_meet.nlargest(5).sum() / max(org_meet.sum(), 1)), 3),
        })
    gini_df = pd.DataFrame(gini_rows).sort_values(["stage", "gini"])
    print(gini_df.to_string(index=False))
    results["access_concentration"] = gini_df.to_dict(orient="records")

    # Does the meeting effect differ in high-concentration procedures?
    subsection("Interaction with procedure-level concentration")
    gini_map = {(r["procedure"], r["stage"]): r["gini"] for r in gini_rows}
    sub["proc_gini"] = [gini_map.get((p, s), np.nan) for p, s in zip(sub["procedure"], sub["stage"])]
    # Median split
    median_gini = sub["proc_gini"].median()
    sub["high_concentration"] = (sub["proc_gini"] > median_gini).astype(int)
    m_conc = fit_logit_cluster(
        sub,
        "aligned ~ log_meetings * high_concentration + is_preproposal + C(procedure)",
    )
    print_focal(
        m_conc,
        ["log_meetings", "high_concentration", "log_meetings:high_concentration", "is_preproposal"],
        "Meetings × concentration",
    )
    results["model_concentration"] = coef_dict(m_conc)

    # ════════════════════════════════════════════════════════════════════════════
    section("6 — PROCEDURE SALIENCE INTERACTION")
    print("  Salience proxies per procedure: total HYS chunks, total amendments")
    print("  Hypothesis: meeting effect stronger in high-salience procedures\n")

    # Pull salience from the data we already have
    salience = {}
    for proc_name, proc_id in PROC_IDS.items():
        try:
            n_chunks = (
                client.table("hys_feedback_chunks")
                .select("feedback_id", count="exact")
                .eq("procedure_id", proc_id)
                .limit(1)
                .execute()
                .count
            )
            n_amends = (
                client.table("procedure_amendments")
                .select("id", count="exact")
                .eq("procedure_id", proc_id)
                .limit(1)
                .execute()
                .count
            )
        except Exception:
            n_chunks, n_amends = 0, 0
        salience[proc_name] = {"chunks": n_chunks, "amends": n_amends}

    sal_df = pd.DataFrame([
        {"procedure": k, "n_chunks": v["chunks"], "n_amends": v["amends"]}
        for k, v in salience.items()
    ])
    sal_df["log_salience"] = np.log1p(sal_df["n_chunks"] + sal_df["n_amends"])
    print(sal_df.to_string(index=False))
    print(f"\n  Median log salience: {sal_df['log_salience'].median():.2f}")
    results["procedure_salience"] = sal_df.to_dict(orient="records")

    sal_map = {r["procedure"]: r["log_salience"] for r in sal_df.to_dict(orient="records")}
    sub["log_salience"] = sub["procedure"].map(sal_map)

    m_sal = fit_logit_cluster(
        sub,
        "aligned ~ log_meetings * log_salience + is_preproposal + C(procedure)",
    )
    print_focal(
        m_sal,
        ["log_meetings", "log_salience", "log_meetings:log_salience", "is_preproposal"],
        "Meetings × salience",
    )
    results["model_salience"] = coef_dict(m_sal)

    # ════════════════════════════════════════════════════════════════════════════
    section("SAVE")
    out_path = Path("analysis/rq2_robustness.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results → {out_path}")


if __name__ == "__main__":
    main()
