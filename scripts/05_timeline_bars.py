"""DSA validation-case timeline figures — produces:

  * timeline_bar_dsa.pdf         (Fig. §4.3 — monthly meetings, stacked by institutional source)
  * timeline_bar_aligned_dsa.pdf (Fig. §4.3 — same data, stacked by org's dominant HYS alignment)

Pulls DSA-linked meetings from Supabase (Commission + lobbying) and joins the
dominant ALIGNED/OPPOSING/UNDETECTABLE/NOISE label that each attending
organisation receives across its DSA dyads. The dominant label is the
plurality label across an org's preproposal + amendment dyads.

Five vertical dashed lines mark the procedure milestones.

Usage:
    python scripts/timeline_bars.py
    python scripts/timeline_bars.py --out-dir "path/to/THESIS/images"
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parent.parent
ANA  = ROOT / "analysis_results"

DSA_PROCEDURE_ID = "2020/0361(COD)"
DSA_FOLDER       = "2020:0361COD"

# Five DSA legislative milestones (Brussels time, ISO date) — colour per line
DSA_MILESTONES = [
    ("Commission proposal", date(2020, 12, 15), "#666666"),
    ("Draft report",        date(2021,  5, 21), "#888888"),
    ("Committee report",    date(2021, 12, 14), "#7c3aed"),
    ("EP plenary vote",     date(2022,  1, 20), "#ed7d31"),
    ("Final adoption",      date(2022,  7,  5), "#c0504d"),
]


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------

def _client():
    load_dotenv(ROOT / ".env")
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])


def _paginate(query_fn, page_size: int = 1000) -> list[dict]:
    out: list[dict] = []
    offset = 0
    while True:
        resp = query_fn(offset, page_size).execute()
        rows = resp.data or []
        if not rows:
            break
        out.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return out


def fetch_amendments_tabled_date(client) -> Optional[date]:
    """Fetch the amendments_tabled_date for the DSA procedure from Supabase."""
    resp = (client.table("procedures")
            .select("amendments_tabled_date")
            .eq("id", DSA_PROCEDURE_ID)
            .limit(1)
            .execute())
    rows = resp.data or []
    if rows and rows[0].get("amendments_tabled_date"):
        return date.fromisoformat(rows[0]["amendments_tabled_date"])
    return None


def fetch_dsa_meetings(client) -> pd.DataFrame:
    """One row per (meeting × organisation) for DSA-linked meetings.

    Columns: meeting_date, source ('lobbying' | 'commission'), organisation
    """
    # ---- Lobbying side: one org per meeting (mep_id ↔ organization_id)
    lob = _paginate(lambda off, lim: (
        client.table("meeting_procedure_links")
        .select("lobbying_meeting_id, lobbying_meetings(meeting_date, organization_id, organizations(name))")
        .eq("procedure_id", DSA_PROCEDURE_ID)
        .eq("is_primary", True)
        .not_.is_("lobbying_meeting_id", "null")
        .range(off, off + lim - 1)
    ))
    lob_rows = []
    for r in lob:
        m = r.get("lobbying_meetings") or {}
        if not m.get("meeting_date"):
            continue
        org = (m.get("organizations") or {}).get("name") or ""
        lob_rows.append({"meeting_date": m["meeting_date"], "source": "lobbying", "organisation": org})

    # ---- Commission side: many orgs per meeting (junction table)
    com = _paginate(lambda off, lim: (
        client.table("meeting_procedure_links")
        .select("commission_meeting_id, commission_meetings(meeting_date)")
        .eq("procedure_id", DSA_PROCEDURE_ID)
        .eq("is_primary", True)
        .not_.is_("commission_meeting_id", "null")
        .range(off, off + lim - 1)
    ))
    com_ids = [r["commission_meeting_id"] for r in com if r.get("commission_meeting_id")]
    com_dates = {
        r["commission_meeting_id"]: (r.get("commission_meetings") or {}).get("meeting_date")
        for r in com
    }
    # one row per (meeting × org) by joining commission_meeting_organizations
    com_rows = []
    for i in range(0, len(com_ids), 200):
        chunk = com_ids[i : i + 200]
        resp = (client.table("commission_meeting_organizations")
                .select("meeting_id, organization_name")
                .in_("meeting_id", chunk)
                .execute())
        for r in (resp.data or []):
            d = com_dates.get(r["meeting_id"])
            if d:
                com_rows.append({"meeting_date": d,
                                 "source": "commission",
                                 "organisation": (r.get("organization_name") or "").strip()})

    df = pd.DataFrame(lob_rows + com_rows)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"], errors="coerce")
    df = df.dropna(subset=["meeting_date"])
    return df


# ---------------------------------------------------------------------------
# Dominant alignment label per org (from DSA dyads)
# ---------------------------------------------------------------------------

def _load_dsa_dyads() -> pd.DataFrame:
    folder = ANA / DSA_FOLDER
    frames = []
    for prefix in ("preproposal", "amendment"):
        cands = sorted(folder.glob(f"{prefix}_*_dyads.csv"))
        if cands:
            frames.append(pd.read_csv(cands[-1]))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def dominant_labels() -> dict[str, str]:
    """Plurality label per organisation across all DSA dyads."""
    df = _load_dsa_dyads()
    if df.empty:
        return {}
    out: dict[str, str] = {}
    for org, sub in df.groupby("organisation"):
        counts = Counter(sub["label"].dropna().str.upper())
        if counts:
            out[org.strip()] = counts.most_common(1)[0][0]
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

SOURCE_COLOURS = {"commission": "#1f77b4", "lobbying": "#ff7f0e"}
ALIGN_COLOURS  = {"ALIGNED": "#2ca02c", "OTHER": "#cccccc"}


def _monthly_pivot(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["meeting_date"].dt.to_period("M").dt.to_timestamp()
    return (
        df.groupby(["month", category_col]).size()
          .unstack(fill_value=0)
          .sort_index()
    )


def _milestone_lines(ax) -> None:
    ymax = ax.get_ylim()[1]
    for label, d, colour in DSA_MILESTONES:
        ax.axvline(pd.Timestamp(d), color=colour, lw=1.0, ls="--", alpha=0.85)
        ax.text(pd.Timestamp(d) + pd.Timedelta(days=4), ymax * 0.97, label,
                rotation=90, va="top", ha="left", fontsize=8, color=colour)


def _amendment_marker(ax, amendments_date: Optional[date]) -> None:
    if amendments_date is None:
        return
    ts = pd.Timestamp(amendments_date)
    ymax = ax.get_ylim()[1]
    colour = "#d62728"
    ax.axvline(ts, color=colour, lw=0.7, ls=":", alpha=0.7)
    ax.text(ts + pd.Timedelta(days=4), ymax * 0.97, "Amendments tabled",
            rotation=90, va="top", ha="left", fontsize=8, color=colour)


def _style_axes(ax, ymax_pad: float = 1.05) -> None:
    """Common cosmetic: monthly ticks every 3 months, ISO Y-m format, y-grid."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("Meetings linked to procedure")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    cur = ax.get_ylim()[1]
    ax.set_ylim(0, cur * ymax_pad)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_timeline_source(df: pd.DataFrame, out: Path,
                         amendments_date: Optional[date] = None) -> None:
    pv = _monthly_pivot(df, "source")
    for col in ("commission", "lobbying"):
        if col not in pv.columns:
            pv[col] = 0
    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    ax.bar(pv.index, pv["commission"], width=24,
           color=SOURCE_COLOURS["commission"], label="Commissioner meetings", edgecolor="none")
    ax.bar(pv.index, pv["lobbying"], width=24, bottom=pv["commission"],
           color=SOURCE_COLOURS["lobbying"], label="MEP meetings", edgecolor="none")
    ax.set_title("DSA: Lobbying meetings over the legislative timeline", fontsize=10)
    _style_axes(ax)
    ax.legend(frameon=True, loc="upper left", fontsize=9)
    _milestone_lines(ax)
    _amendment_marker(ax, amendments_date)
    fig.tight_layout()
    fig.savefig(out / "timeline_bar_dsa.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_timeline_aligned(df: pd.DataFrame, dom: dict[str, str], out: Path,
                          amendments_date: Optional[date] = None) -> None:
    df = df.copy()
    df["align"] = df["organisation"].str.strip().map(dom)
    df["bucket"] = np.where(df["align"] == "ALIGNED", "ALIGNED", "OTHER")
    pv = _monthly_pivot(df, "bucket")
    for col in ("ALIGNED", "OTHER"):
        if col not in pv.columns:
            pv[col] = 0
    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    ax.bar(pv.index, pv["ALIGNED"], width=24,
           color=ALIGN_COLOURS["ALIGNED"], label="Aligned", edgecolor="none")
    ax.bar(pv.index, pv["OTHER"], width=24, bottom=pv["ALIGNED"],
           color=ALIGN_COLOURS["OTHER"], label="Other / no data", edgecolor="none")
    ax.set_title("DSA: Lobbying meetings by alignment profile", fontsize=10)
    _style_axes(ax)
    ax.legend(frameon=True, loc="upper left", fontsize=9)
    _milestone_lines(ax)
    _amendment_marker(ax, amendments_date)
    fig.tight_layout()
    fig.savefig(out / "timeline_bar_aligned_dsa.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",
                        default=str(Path(__file__).resolve().parent / "images"),
                        help="Where to write the PDFs (default: scripts/images)")
    parser.add_argument("--from-cutoff", default="2019-09-01",
                        help="Drop meetings before this date (default 2019-09-01)")
    parser.add_argument("--to-cutoff", default="2022-12-31",
                        help="Drop meetings after this date (default 2022-12-31)")
    args = parser.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    client = _client()
    df = fetch_dsa_meetings(client)
    lo, hi = pd.Timestamp(args.from_cutoff), pd.Timestamp(args.to_cutoff)
    df = df[(df["meeting_date"] >= lo) & (df["meeting_date"] <= hi)]
    print(f"fetched {len(df):,} (meeting × org) rows for DSA between {lo.date()} and {hi.date()}")
    print(df["source"].value_counts().to_string())

    amend_date = fetch_amendments_tabled_date(client)
    print(f"amendments tabled date: {amend_date}")

    dom = dominant_labels()
    print(f"dominant labels for {len(dom):,} DSA orgs")

    plot_timeline_source(df, out, amendments_date=amend_date)
    plot_timeline_aligned(df, dom, out, amendments_date=amend_date)
    print(f"wrote → {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
