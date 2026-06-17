"""Graphs from the contributor-day analysis.

Produces the two headline graphs from the brief plus supporting views:
  1. How often do people do >Nh of work in a day? (count + % of contributor-days
     over the threshold, by quarter)
  2. Engineering hours per contributor-day over time, by quarter
  3. Distribution of hours per contributor-day (with the threshold marked)
  4. Total estimated engineering hours by quarter

PNGs are written to data/<repo>/plots/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as cfgmod


def _load(cfg: cfgmod.Config, repo_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dd = cfg.data_dir(repo_name)
    cd = pd.read_csv(dd / "contributor_days.csv")
    prs = pd.read_csv(dd / "pr_estimates.csv")
    return prs, cd


def _qorder(df: pd.DataFrame, col: str = "quarter") -> list[str]:
    return sorted(df[col].dropna().unique().tolist())


def make_plots(cfg: cfgmod.Config, repo_name: str) -> Path:
    prs, cd = _load(cfg, repo_name)
    threshold = float(cfg.analyze.get("hours_per_day_threshold", 24))
    out = cfg.data_dir(repo_name) / "plots"
    out.mkdir(exist_ok=True)
    quarters = _qorder(cd)

    # 1. >Nh/day frequency by quarter -----------------------------------------
    g = cd.groupby("quarter")
    over = g.apply(lambda x: (x["hours"] > threshold).sum()).reindex(quarters).fillna(0)
    frac = g.apply(lambda x: 100 * (x["hours"] > threshold).mean()).reindex(quarters).fillna(0)
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(quarters, over.values, color="#c44", alpha=0.75, label=f">{threshold:g}h-days (count)")
    ax1.set_ylabel(f"# contributor-days > {threshold:g}h", color="#c44")
    ax1.set_xlabel("quarter")
    ax2 = ax1.twinx()
    ax2.plot(quarters, frac.values, "o-", color="#225", label="% of contributor-days")
    ax2.set_ylabel("% of contributor-days", color="#225")
    plt.title(f"How often do contributors log >{threshold:g}h of estimated work in a day? — {repo_name}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "over_threshold_by_quarter.png", dpi=130)
    plt.close(fig)

    # 2. Mean hours per contributor-day by quarter -----------------------------
    means = g["hours"].mean().reindex(quarters)
    medians = g["hours"].median().reindex(quarters)
    counts = g["hours"].count().reindex(quarters)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(quarters, means.values, "o-", label="mean", color="#225")
    ax.plot(quarters, medians.values, "s--", label="median", color="#693")
    ax.axhline(8, color="grey", ls=":", lw=1, label="8h reference")
    for x, c in zip(quarters, counts.values):
        ax.annotate(f"n={int(c)}", (x, means[x]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="grey")
    ax.set_ylabel("estimated engineering hours / contributor-day")
    ax.set_xlabel("quarter")
    ax.legend()
    plt.title(f"Engineering hours per contributor-day over time — {repo_name}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "hours_per_contributor_day_by_quarter.png", dpi=130)
    plt.close(fig)

    # 3. Distribution of hours per contributor-day -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    clipped = cd["hours"].clip(upper=48)
    ax.hist(clipped, bins=48, color="#479", alpha=0.85)
    ax.axvline(threshold, color="#c44", ls="--", label=f"{threshold:g}h/day")
    ax.set_xlabel("estimated engineering hours in a (contributor, day) — clipped at 48")
    ax.set_ylabel("# contributor-days")
    ax.legend()
    plt.title(f"Distribution of estimated hours per contributor-day — {repo_name}")
    fig.tight_layout()
    fig.savefig(out / "hours_per_contributor_day_hist.png", dpi=130)
    plt.close(fig)

    # 4. Total estimated hours by quarter --------------------------------------
    if "merged_at" in prs.columns:
        p = prs.dropna(subset=["hours_ensemble", "merged_at"]).copy()
        p["q"] = pd.to_datetime(p["merged_at"], errors="coerce", utc=True)
        p = p.dropna(subset=["q"])
        p["quarter"] = p["q"].dt.year.astype(str) + "Q" + ((p["q"].dt.month - 1) // 3 + 1).astype(str)
        tot = p.groupby("quarter")["hours_ensemble"].sum().reindex(_qorder(p))
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(tot.index, tot.values, color="#693", alpha=0.8)
        ax.set_ylabel("total estimated engineering hours")
        ax.set_xlabel("quarter (by merge date)")
        plt.title(f"Total estimated unassisted-human engineering hours by quarter — {repo_name}")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out / "total_hours_by_quarter.png", dpi=130)
        plt.close(fig)

    print(f"[plots] wrote graphs -> {out}")
    return out
