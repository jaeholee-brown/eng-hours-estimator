"""Tiered, properly-measured contributor-day analysis.

Corrections over the first-pass analyze.py:
  (a) Contributor-days are measured from BRANCH COMMITS ONLY (the days work was
      actually authored), not branch+squash — the squash/merge commit's date is a
      process artifact (when it landed), not necessarily a day worked. We also
      emit the branch+merge variant for comparison.
  (b) Contributors are keyed by CANONICAL GitHub login (merging a person's
      multiple emails), via the prior identity mapping, not by raw email.
  (c) Tiers use the real rosters: everybody / openai-ish (322) / collaborator-41,
      from ~/Downloads/org_public_repo_loc_analysis.

Outputs CSVs + plots under data/codex/tiers/.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/Users/jaeholee0404/eng-hours-estimator")
LOC = Path("/Users/jaeholee0404/Downloads/org_public_repo_loc_analysis")
DATA = ROOT / "data" / "codex"
OUT = DATA / "tiers"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["haiku-4.5", "gpt-5.4-mini", "gemini-3-flash"]
THRESH = 24.0
TIERS = ["everybody", "openai-ish", "collab41"]


# ── rosters / identity ───────────────────────────────────────────────────────
def load_identity():
    collab = {l.strip().lower() for l in
              open(LOC / "anthropic_method_active_contributor_graphs/collaborator41_authors.txt")
              if l.strip()}
    login_flags: dict[str, tuple[bool, bool]] = {}
    email_login: dict[str, str] = {}
    imap = LOC / "contributor_day_graphs/flagship_openai_filtered/codex_contributor_day_identity_mapping.csv"
    for r in csv.DictReader(open(imap)):
        lg = (r.get("mapped_login") or "").lower()
        if lg:
            login_flags[lg] = (r.get("is_collab41") == "True", r.get("is_openaiish") == "True")
        m = re.search(r"<([^>]+)>", r.get("author_id", "") or "")
        if m:
            email_login[m.group(1).lower()] = lg
    return collab, login_flags, email_login


COLLAB, LOGIN_FLAGS, EMAIL_LOGIN = load_identity()


def classify(login, email):
    """Return (canonical_id, is_collab41, is_openaiish)."""
    lg = (login or "").lower()
    e = (email or "").lower()
    canon = lg or EMAIL_LOGIN.get(e) or e
    f_canon = LOGIN_FLAGS.get(canon, (False, False))
    f_lg = LOGIN_FLAGS.get(lg, (False, False))
    c41 = (lg in COLLAB) or (EMAIL_LOGIN.get(e) in COLLAB) or f_canon[0] or f_lg[0]
    oi = f_canon[1] or f_lg[1] or (EMAIL_LOGIN.get(e) in COLLAB) or e.endswith("@openai.com") or c41
    return canon, c41, oi


def utc_day(ts):
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        d = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc).date().isoformat()
    except ValueError:
        return None


def quarter(day):
    d = datetime.fromisoformat(day)
    return f"{d.year}Q{(d.month - 1)//3 + 1}"


def month(day):
    return day[:7]


# ── data ─────────────────────────────────────────────────────────────────────
def load_prs():
    seen, out = set(), []
    for line in open(DATA / "prs.jsonl"):
        r = json.loads(line)
        if r["number"] in seen:
            continue
        seen.add(r["number"]); out.append(r)
    return out


def load_hours():
    est = {}
    for line in open(DATA / "estimates.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("hours") is not None and r["model_key"] in MODELS:
            est[(r["number"], r["model_key"])] = r["hours"]
    hours = {}
    for line in open(DATA / "prs.jsonl"):
        n = json.loads(line)["number"]
        vals = [est[(n, m)] for m in MODELS if (n, m) in est]
        if vals:
            hours[n] = statistics.median(vals)
    return hours


def contributor_days(pr, include_merge):
    """Distinct (canonical_id, utc_day, c41, oi) for a PR."""
    rows = {}
    for c in pr.get("branch_commits") or []:
        d = utc_day(c.get("authored_date"))
        if not d:
            continue
        cid, c41, oi = classify(c.get("login"), c.get("email"))
        rows[(cid, d)] = (c41, oi)
    if include_merge:
        d = utc_day(pr.get("author_date"))
        if d:
            cid, c41, oi = classify(pr.get("pr_author_login"), pr.get("author_email"))
            rows[(cid, d)] = (c41, oi)
    return [(cid, d, f[0], f[1]) for (cid, d), f in rows.items()]


def build(prs, hours, include_merge):
    """Aggregate hours per (contributor, day) under the chosen allocation."""
    cd = defaultdict(lambda: {"hours": 0.0, "c41": False, "oi": False})
    for pr in prs:
        h = hours.get(pr["number"])
        if h is None:
            continue
        pairs = contributor_days(pr, include_merge)
        # exclude bot-looking ids
        pairs = [p for p in pairs if not re.search(r"bot|dependabot|github-actions", p[0], re.I)]
        if not pairs:
            continue
        per = h / len(pairs)
        for cid, d, c41, oi in pairs:
            k = (cid, d)
            cd[k]["hours"] += per
            cd[k]["c41"] |= c41
            cd[k]["oi"] |= oi
    rows = []
    for (cid, d), v in cd.items():
        rows.append({"contributor": cid, "day": d, "hours": v["hours"],
                     "quarter": quarter(d), "month": month(d),
                     "collab41": v["c41"], "openai-ish": v["oi"]})
    return rows


def tier_filter(rows, tier):
    if tier == "everybody":
        return rows
    if tier == "openai-ish":
        return [r for r in rows if r["openai-ish"]]
    if tier == "collab41":
        return [r for r in rows if r["collab41"]]
    return rows


def quarters_of(rows):
    return sorted({r["quarter"] for r in rows})


# ── plots ────────────────────────────────────────────────────────────────────
def plot_hours_per_cd(rows_by_tier, qs, fname, title):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for tier, color in zip(TIERS, ["#225", "#693", "#c44"]):
        rows = rows_by_tier[tier]
        means = []
        for q in qs:
            hv = [r["hours"] for r in rows if r["quarter"] == q]
            means.append(statistics.fmean(hv) if hv else float("nan"))
        ax.plot(qs, means, "o-", label=tier, color=color)
    ax.axhline(8, color="grey", ls=":", lw=1)
    ax.set_ylabel("mean estimated eng-hours / contributor-day")
    ax.set_xlabel("quarter")
    ax.legend(title="tier")
    ax.set_title(title)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130); plt.close(fig)


def plot_over_threshold(rows_by_tier, qs, fname, title):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for tier, color in zip(TIERS, ["#225", "#693", "#c44"]):
        rows = rows_by_tier[tier]
        frac = []
        for q in qs:
            hv = [r["hours"] for r in rows if r["quarter"] == q]
            frac.append(100 * sum(1 for x in hv if x > THRESH) / len(hv) if hv else float("nan"))
        ax.plot(qs, frac, "o-", label=tier, color=color)
    ax.set_ylabel(f"% of contributor-days > {THRESH:g}h")
    ax.set_xlabel("quarter")
    ax.legend(title="tier")
    ax.set_title(title)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130); plt.close(fig)


def plot_throughput(rows_by_tier, qs, fname, title):
    """Mean estimated eng-hours per active contributor per month, by quarter."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for tier, color in zip(TIERS, ["#225", "#693", "#c44"]):
        rows = rows_by_tier[tier]
        permonth = defaultdict(lambda: defaultdict(float))  # month -> contributor -> hours
        for r in rows:
            permonth[r["month"]][r["contributor"]] += r["hours"]
        q_vals = defaultdict(list)
        for mo, contribs in permonth.items():
            q = quarter(mo + "-01")
            q_vals[q].append(statistics.fmean(contribs.values()))
        means = [statistics.fmean(q_vals[q]) if q_vals.get(q) else float("nan") for q in qs]
        ax.plot(qs, means, "o-", label=tier, color=color)
    ax.set_ylabel("mean eng-hours per active contributor-month")
    ax.set_xlabel("quarter")
    ax.legend(title="tier")
    ax.set_title(title)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130); plt.close(fig)


def summary_table(rows_by_tier, qs, label):
    print(f"\n=== mean hours/contributor-day by quarter — {label} ===")
    print(f"{'quarter':<9}" + "".join(f"{t:>14}" for t in TIERS))
    for q in qs:
        cells = []
        for t in TIERS:
            hv = [r["hours"] for r in rows_by_tier[t] if r["quarter"] == q]
            cells.append(f"{statistics.fmean(hv):.2f} (n={len(hv)})" if hv else "—")
        print(f"{q:<9}" + "".join(f"{c:>14}" for c in cells))


def main():
    prs = load_prs()
    hours = load_hours()
    print(f"PRs with ensemble hours: {len(hours)}/{len(prs)}")

    for include_merge, tag, label in [(False, "branch_only", "BRANCH COMMITS ONLY (proper)"),
                                       (True, "branch_merge", "branch+merge (previous)")]:
        rows = build(prs, hours, include_merge)
        qs = quarters_of(rows)
        by_tier = {t: tier_filter(rows, t) for t in TIERS}
        # persist
        with open(OUT / f"contributor_days_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["contributor", "day", "hours", "quarter", "month", "collab41", "openai-ish"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        # tier sizes
        sizes = {t: len({r["contributor"] for r in by_tier[t]}) for t in TIERS}
        cds = {t: len(by_tier[t]) for t in TIERS}
        print(f"\n[{tag}] distinct contributors: " + ", ".join(f"{t}={sizes[t]}" for t in TIERS))
        print(f"[{tag}] contributor-days: " + ", ".join(f"{t}={cds[t]}" for t in TIERS))
        summary_table(by_tier, qs, label)
        plot_hours_per_cd(by_tier, qs, f"hours_per_contributor_day__{tag}.png",
                          f"Eng-hours per contributor-day by tier — {label}")
        plot_over_threshold(by_tier, qs, f"over_24h__{tag}.png",
                            f">{THRESH:g}h-day frequency by tier — {label}")
        plot_throughput(by_tier, qs, f"throughput_per_contributor_month__{tag}.png",
                        f"Eng-hours per active contributor-month by tier — {label}")
    print(f"\nwrote plots + CSVs -> {OUT}")


if __name__ == "__main__":
    main()
