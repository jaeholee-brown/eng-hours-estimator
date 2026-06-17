"""Generate the report figures into report/ (tracked, shareable)."""
from __future__ import annotations
import re, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tier_analysis as ta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ta.ROOT / "report"
OUT.mkdir(exist_ok=True)
COLORS = {"everybody": "#2b2d6b", "openai-ish": "#5a8a2a", "collab41": "#c0392b"}


def cd_table():
    prs, hours = ta.load_prs(), ta.load_hours()
    cd = defaultdict(lambda: {"h": 0.0, "l": 0.0, "c41": False, "oi": False, "q": None})
    for pr in prs:
        h = hours.get(pr["number"])
        if h is None:
            continue
        L = (pr.get("additions") or 0) + (pr.get("deletions") or 0)
        pairs = [p for p in ta.contributor_days(pr, include_merge=False)
                 if not re.search(r"bot|dependabot|github-actions", p[0], re.I)]
        if not pairs:
            continue
        ph, pl = h / len(pairs), L / len(pairs)
        for cid, d, c41, oi in pairs:
            e = cd[(cid, d)]
            e["h"] += ph; e["l"] += pl; e["c41"] |= c41; e["oi"] |= oi; e["q"] = ta.quarter(d)
    return list(cd.values())


ROWS = cd_table()
QS = sorted({r["q"] for r in ROWS})
def tier(t): return ROWS if t == "everybody" else [r for r in ROWS if (r["oi"] if t == "openai-ish" else r["c41"])]


def lines_fig(yfn, ylabel, title, fname, ref=None):
    fig, ax = plt.subplots(figsize=(11, 5.3))
    for t in ta.TIERS:
        ys = [yfn([r for r in tier(t) if r["q"] == q]) for q in QS]
        ax.plot(QS, ys, "o-", label=t, color=COLORS[t], lw=2)
    if ref:
        ax.axhline(ref, color="grey", ls=":", lw=1)
    ax.set_ylabel(ylabel); ax.set_xlabel("quarter"); ax.legend(title="tier")
    ax.set_title(title); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / fname, dpi=135); plt.close(fig)


# 1. hours per contributor-day
lines_fig(lambda x: statistics.fmean([r["h"] for r in x]) if x else float("nan"),
          "mean est. eng-hours / contributor-day",
          "Estimated engineering-hours per contributor-day, by group (codex)",
          "fig1_hours_per_contributor_day.png", ref=8)
# 2. >24h frequency
lines_fig(lambda x: 100*sum(1 for r in x if r["h"] > 24)/len(x) if x else float("nan"),
          "% of contributor-days > 24h",
          "Frequency of >24h estimated-work days, by group (codex)",
          "fig2_over_24h.png")
# 3. hours per 1,000 lines
def hpk(x):
    Lk = sum(r["l"] for r in x)/1000
    return sum(r["h"] for r in x)/Lk if Lk > 0 else float("nan")
lines_fig(hpk, "est. eng-hours per 1,000 changed lines",
          "Estimated effort per 1,000 lines changed, by group (codex)",
          "fig3_hours_per_kloc.png")

# 4. composition: share of active contributor-days
fig, ax = plt.subplots(figsize=(11, 5.3))
for t in ["openai-ish", "collab41"]:
    share = [100*len([r for r in tier(t) if r["q"] == q])/len([r for r in ROWS if r["q"] == q]) for q in QS]
    ax.plot(QS, share, "o-", label=t, color=COLORS[t], lw=2)
ax.set_ylabel("% of active contributor-days"); ax.set_xlabel("quarter")
ax.set_ylim(0, 105); ax.legend(title="group")
ax.set_title("Composition of active contributor-days over time (codex)")
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(OUT / "fig4_composition.png", dpi=135); plt.close(fig)

print("wrote figures ->", OUT)
for q in QS:
    a = [r for r in ROWS if r["q"] == q]
    print(f"  {q}: cd={len(a)}  oai_share={100*len([r for r in a if r['oi']])/len(a):.0f}%")
