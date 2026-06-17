# eng-hours-estimator

Estimate the **human-equivalent engineering hours** behind every merged PR in a
repository, using an **LLM-as-a-judge ensemble**, then aggregate those hours into
**contributor-day** metrics.

The judge answers one counterfactual question per PR, adapted from
[Amy Deng / METR's "time without AI" judge prompt](https://metr.org/notes/2026-02-17-exploratory-transcript-analysis-for-estimating-time-savings-from-coding-agents/#appendix-a-llm-judge-prompts):

> How long would **one unassisted human software engineer, working alone, with no
> AI tools of any kind**, take to produce the net change that this PR merged?

It is built to run over **any repo** — Codex is just the first subfolder under
`repos/`. Drop another checkout in `repos/` and add it to `config.yaml`.

---

## Why this exists

The usual "AI uplift" proxies (LoC per contributor, LoC per contributor-day) didn't
show the expected monotonic trend. This tool tries a different signal: instead of
counting lines, it asks an ensemble of models to **estimate the unassisted-human
effort** each merged PR represents, then looks at how that effort distributes per
contributor per day — e.g. *"how often does a contributor rack up >24h of
estimated work in a single day?"* (a proxy for AI leverage) and *"engineering
hours per contributor-day, by quarter"*.

---

## How it works

```
 repos/<repo>  ──extract──▶  data/<repo>/prs.jsonl
   (git history)               (title, body, net diff, branch commits)
                                       │
                                    judge (async ensemble: Haiku 4.5 + gpt-5.4-mini + Gemini 3 Flash)
                                       │
                                       ▼
                              data/<repo>/estimates.jsonl   (hours per PR per model)
                                       │
                                    analyze (ensemble + contributor-day allocation)
                                       │
                          ┌────────────┴───────────┐
                          ▼                         ▼
              data/<repo>/pr_estimates.csv   data/<repo>/contributor_days.csv
                                       │
                                     plot
                                       ▼
                              data/<repo>/plots/*.png
```

**Data sources.** The repo here uses squash-merge (one commit per PR, no merge
commits), so local git gives us each PR's *net diff* for free. The original
**branch commits** (authors + authoring dates) — needed so a PR's hours can be
spread across the real days/people who worked on it — are fetched from the **GitHub
GraphQL API**. Diffs never hit the API.

**The contributor-day model.** A *contributor-day* `(c, d)` is a distinct
(author, calendar-day in UTC) pair among a PR's commits (its branch commits **and**
the squash/merge commit). A PR's estimated hours are split **evenly** across the
distinct contributor-days its commits touch. So 60 h spread over two days (one
author) → 30 h/day; 60 h across two authors on one day each → 30 h each. Each
`(c, d)`'s total is the sum of allocations from every PR landing on it.

**The ensemble.** Three models judge every PR independently. No reasoning/extended
thinking is used; each model *talks out loud* briefly, then ends with
`FINAL_ESTIMATE_HOURS: <number>`, which is parsed robustly. The per-PR estimate is
the **median** of the available model answers (configurable).

---

## Setup

```bash
cd eng-hours-estimator
python -m venv .venv && source .venv/bin/activate
pip install -e .                      # or: pip install -r requirements.txt
cp .env.example .env                  # then fill in your API keys

# point the tool at a repo (symlink an existing clone — no extra disk)
ln -s /path/to/codex repos/codex
```

`.env` holds `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, and an
optional `GITHUB_TOKEN` (falls back to your `gh auth` token). It is gitignored.

---

## Usage

```bash
# run everything below with the venv active; either form works:
eng-hours <cmd> ...
PYTHONPATH=src python -m eng_hours <cmd> ...

eng-hours list                      # show configured repos & models
eng-hours extract codex             # build data/codex/prs.jsonl (git + GitHub)
eng-hours cost codex                # precise $ estimate for a full run
eng-hours test codex --limit 3      # tiny live smoke test (a few PRs, all models)
eng-hours judge codex               # run the full ensemble (resumable; prints live spend)
eng-hours spend codex               # actual $ spent so far, split per model (from token usage)
eng-hours analyze codex             # ensemble + contributor-day CSVs + summary
eng-hours plot codex                # render graphs to data/codex/plots/
eng-hours all codex                 # extract -> judge -> analyze -> plot
```

`judge` is **resumable** — it skips `(PR, model)` pairs already in
`estimates.jsonl`, so you can stop and restart freely.

### Adding another repo

```bash
ln -s /path/to/other-repo repos/other
```
```yaml
# config.yaml
repos:
  - name: codex
    path: repos/codex
    github: openai/codex
  - name: other
    path: repos/other
    github: someorg/other-repo
```
Then `eng-hours all other`.

---

## Configuration (`config.yaml`)

Everything is configurable: which repos/models, pricing (for the cost estimate),
the diff cap, GitHub concurrency, the ensembling method, the `>Nh/day` threshold,
and whether to exclude bot authors. See the comments in `config.yaml`.

### GitHub rate limits (the "optimize for speed" part)

Diffs come from local git, so the API is used only for commit metadata. The fetch
is built to stay fast and within limits:

- **GraphQL, batched** — one query fetches `graphql_batch_size` PRs at once (each
  aliased), each with up to `commits_per_pr` commits. ~7.5k PRs → a few hundred
  queries instead of tens of thousands of REST calls.
- **Bounded concurrency** (`max_concurrency`) keeps us under the secondary limits
  (≤100 concurrent, ≲900 points/min) while saturating throughput.
- **Header-aware backoff** — every response's `rateLimit { remaining resetAt }` is
  read; we pause until reset near the floor, and honor `Retry-After` / back off on
  403/429.

> Note: a teammate's ("Kwa's") GitHub rate-limit notes were referenced in the
> brief but weren't recoverable from the local Codex session logs (the keyword
> hits were all unrelated). The strategy above is the standard best-practice set;
> swap in any specific numbers from that note via `config.yaml → github`.

---

## Outputs

| File | What |
|---|---|
| `data/<repo>/prs.jsonl` | one record per merged PR (title, body, net diff, branch commits, diffstat) |
| `data/<repo>/estimates.jsonl` | one record per (PR, model): hours, parse method, token usage, raw tail |
| `data/<repo>/pr_estimates.csv` | per PR: ensemble + per-model hours, diffstat, #contributor-days |
| `data/<repo>/contributor_days.csv` | per (contributor, day): summed hours, #PRs, month, quarter |
| `data/<repo>/plots/*.png` | the graphs |

All of `data/` is gitignored.

---

## Notes & caveats

- The estimate is a **counterfactual** ("no AI, one engineer, alone"), not actual
  wall-clock time. It's a relative signal, best read as trends, not absolutes.
- Calendar days are UTC (GitHub's GraphQL `authoredDate` is UTC; local offsets
  aren't exposed there). Configurable assumptions live in `config.yaml`.
- Model pricing in `config.yaml` is current as of mid-2026; update if rates move —
  the `cost` command reads it directly.
