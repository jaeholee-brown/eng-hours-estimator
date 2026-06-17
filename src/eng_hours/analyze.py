"""Ensemble the per-model estimates and aggregate into contributor-day metrics.

Definitions (per the project brief):
  * Ensemble hours for a PR = median (configurable) of the available model estimates.
  * A *contributor-day* (c, d) is a distinct (author, calendar-day) pair among the
    PR's commits — its branch commits (from GitHub) AND the squash/merge commit.
  * A PR's hours are split EVENLY across the distinct contributor-days its commits
    touch. So 60 h over two distinct days (one author) = 30 h on each day; 60 h
    across two authors on one day each = 30 h each.
  * Each (c, d)'s total = the sum of per-PR allocations landing on that (c, d).

Calendar days are taken in UTC for consistency (GitHub's GraphQL `authoredDate`
is UTC; local offsets aren't available there). Bot authors are flagged and
optionally excluded.

Outputs (CSV, under data/<repo>/):
  * pr_estimates.csv      — one row per PR (ensemble + per-model hours, diffstat,
                            #contributor-days, per-(c,d) allocation)
  * contributor_days.csv  — one row per (contributor, day): summed hours, #PRs,
                            month, quarter
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import config as cfgmod
from .extract import load_prs
from .judge import load_estimates


def _to_utc_date(ts: str | None) -> str | None:
    """ISO8601 (with Z or offset) -> 'YYYY-MM-DD' in UTC."""
    if not ts:
        return None
    s = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.fromisoformat(s[:19])
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).date().isoformat()


def _ensemble(values: list[float], how: str) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if how == "mean":
        return statistics.fmean(vals)
    if how == "trimmed_mean" and len(vals) >= 3:
        s = sorted(vals)
        return statistics.fmean(s[1:-1])
    return statistics.median(vals)


def _bot_matcher(patterns: list[str]):
    regexes = [re.compile(p, re.IGNORECASE) for p in patterns]

    def is_bot(*fields: str | None) -> bool:
        for fld in fields:
            if fld and any(rx.search(fld) for rx in regexes):
                return True
        return False

    return is_bot


def _contributor_days_for_pr(pr: dict, is_bot, exclude_bots: bool) -> list[tuple[str, str]]:
    """Distinct (contributor_email, utc_day) pairs touched by the PR's commits."""
    pairs: set[tuple[str, str]] = set()

    def add(email: str | None, name: str | None, login: str | None, ts: str | None):
        if not email or not ts:
            return
        if exclude_bots and is_bot(email, name, login):
            return
        day = _to_utc_date(ts)
        if day:
            pairs.add((email.lower(), day))

    # Branch commits (the real work, spread across days)
    for c in pr.get("branch_commits") or []:
        add(c.get("email"), c.get("name"), c.get("login"), c.get("authored_date"))

    # The squash/merge commit itself
    add(pr.get("author_email"), pr.get("author_name"),
        pr.get("pr_author_login"), pr.get("author_date"))

    return sorted(pairs)


def analyze(cfg: cfgmod.Config, repo_name: str) -> dict[str, pd.DataFrame]:
    prs = load_prs(cfg, repo_name)
    estimates = load_estimates(cfg, repo_name)

    ac = cfg.analyze
    how = ac.get("ensemble", "median")
    exclude_bots = bool(ac.get("exclude_bots", True))
    is_bot = _bot_matcher(ac.get("bot_email_patterns", []))
    model_keys = [m.key for m in cfg.models]

    # (number, model_key) -> hours (last write wins)
    est: dict[tuple[int, str], float | None] = {}
    for r in estimates:
        est[(r["number"], r["model_key"])] = r.get("hours")

    pr_rows: list[dict[str, Any]] = []
    # (email, day) -> accumulators
    cd_hours: dict[tuple[str, str], float] = defaultdict(float)
    cd_prs: dict[tuple[str, str], set[int]] = defaultdict(set)

    for pr in prs:
        num = pr["number"]
        per_model = {k: est.get((num, k)) for k in model_keys}
        hours = _ensemble(list(per_model.values()), how)
        pairs = _contributor_days_for_pr(pr, is_bot, exclude_bots)

        row: dict[str, Any] = {
            "number": num,
            "title": pr.get("title"),
            "merged_at": pr.get("merged_at") or pr.get("commit_date"),
            "author_email": (pr.get("author_email") or "").lower(),
            "files_changed": pr.get("files_changed"),
            "additions": pr.get("additions"),
            "deletions": pr.get("deletions"),
            "hours_ensemble": hours,
            "n_models": sum(1 for v in per_model.values() if v is not None),
            "n_contributor_days": len(pairs),
        }
        for k in model_keys:
            row[f"hours_{k}"] = per_model[k]
        per_pair = (hours / len(pairs)) if (hours is not None and pairs) else None
        row["hours_per_contributor_day"] = per_pair
        pr_rows.append(row)

        if per_pair is not None:
            for pair in pairs:
                cd_hours[pair] += per_pair
                cd_prs[pair].add(num)

    pr_df = pd.DataFrame(pr_rows)

    cd_rows = []
    for (email, day), hrs in cd_hours.items():
        d = datetime.fromisoformat(day)
        cd_rows.append(
            {
                "contributor": email,
                "day": day,
                "hours": hrs,
                "n_prs": len(cd_prs[(email, day)]),
                "month": f"{d.year}-{d.month:02d}",
                "quarter": f"{d.year}Q{(d.month - 1) // 3 + 1}",
                "year": d.year,
            }
        )
    cd_df = pd.DataFrame(cd_rows).sort_values(["day", "contributor"]).reset_index(drop=True)

    # Persist
    dd = cfg.data_dir(repo_name)
    pr_df.to_csv(dd / "pr_estimates.csv", index=False)
    cd_df.to_csv(dd / "contributor_days.csv", index=False)
    print(f"[analyze] wrote {dd/'pr_estimates.csv'} ({len(pr_df)} PRs)")
    print(f"[analyze] wrote {dd/'contributor_days.csv'} ({len(cd_df)} contributor-days)")

    _print_summary(pr_df, cd_df, float(ac.get("hours_per_day_threshold", 24)))
    return {"prs": pr_df, "contributor_days": cd_df}


def _print_summary(pr_df: pd.DataFrame, cd_df: pd.DataFrame, threshold: float) -> None:
    scored = pr_df[pr_df["hours_ensemble"].notna()]
    print("\n──────── summary ────────")
    print(f"PRs with an ensemble estimate : {len(scored)} / {len(pr_df)}")
    if len(scored):
        print(f"Total estimated eng-hours     : {scored['hours_ensemble'].sum():,.0f}")
        print(f"Median PR hours               : {scored['hours_ensemble'].median():.2f}")
    if len(cd_df):
        over = cd_df[cd_df["hours"] > threshold]
        print(f"Contributor-days              : {len(cd_df):,}")
        print(f"  > {threshold:g}h/day               : {len(over):,} "
              f"({100*len(over)/len(cd_df):.1f}%)")
        print(f"Mean hours / contributor-day  : {cd_df['hours'].mean():.2f}")
        print("\nMean hours / contributor-day by quarter:")
        byq = cd_df.groupby("quarter")["hours"].agg(["mean", "count"])
        for q, r in byq.iterrows():
            print(f"  {q}: {r['mean']:.2f}  (n={int(r['count'])})")
    print("─────────────────────────\n")
