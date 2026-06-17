"""Extract per-merged-PR records from a repo's local git history + GitHub.

For each merged PR (a squash/merge commit on the default branch whose subject ends
in `(#1234)`), we assemble:

  * title, body, author, dates, and net diffstat — from local git
  * the net diff (the merged change) — from local git, capped
  * the PR's branch commits (authors + authoring dates) — from the GitHub GraphQL
    API, so the contributor-day spread reflects the real work, not just the
    squash commit

Output: data/<repo>/prs.jsonl, one JSON object per PR.
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import config as cfgmod
from .github_api import GitHubGraphQL, normalize_pr_node, resolve_token

_US = "\x1f"  # unit separator (between fields)
_RS = "\x1e"  # record separator (between commits / numstat blocks)


def _git(repo: Path, args: list[str]) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr[:300]}")
    return out.stdout


def parse_git_log(repo: Path, pr_regex: str) -> list[dict[str, Any]]:
    """One pass for metadata, one for numstat. Returns squash-commit records."""
    pr_re = re.compile(pr_regex)

    # Pass 1: metadata, NUL-separated commits, US-separated fields.
    fmt = _US.join(["%H", "%ae", "%an", "%aI", "%cI", "%s", "%b"])
    raw = _git(repo, ["log", "--no-merges", "-z", f"--format={fmt}"])
    records: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for chunk in raw.split("\x00"):
        if not chunk.strip():
            continue
        fields = chunk.split(_US)
        if len(fields) < 7:
            continue
        sha, ae, an, aI, cI, subject, body = (
            fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], _US.join(fields[6:])
        )
        m = pr_re.search(subject)
        if not m:
            continue
        number = int(m.group(1))
        rec = {
            "number": number,
            "sha": sha,
            "title": subject,
            "body": body.strip(),
            "author_email": ae,
            "author_name": an,
            "author_date": aI,
            "commit_date": cI,
            "additions": 0,
            "deletions": 0,
            "files_changed": 0,
        }
        records[sha] = rec
        order.append(sha)

    # Pass 2: numstat. Each block: "<RS><sha>\n\n<add>\t<del>\t<path>...".
    raw2 = _git(repo, ["log", "--no-merges", "--numstat", f"--format={_RS}%H"])
    for block in raw2.split(_RS):
        block = block.strip("\n")
        if not block:
            continue
        lines = block.split("\n")
        sha = lines[0].strip()
        rec = records.get(sha)
        if not rec:
            continue
        adds = dels = files = 0
        for ln in lines[1:]:
            if not ln.strip():
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            a, d = parts[0], parts[1]
            files += 1
            if a.isdigit():
                adds += int(a)
            if d.isdigit():
                dels += int(d)
        rec["additions"] = adds
        rec["deletions"] = dels
        rec["files_changed"] = files

    return [records[s] for s in order]


def fetch_diff(repo: Path, sha: str, cap: int) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "show", sha, "--format=", "--no-color", "-p"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    diff = out.stdout
    return diff[:cap] if len(diff) > cap else diff


def _attach_diffs(repo: Path, recs: list[dict], cap: int, workers: int = 16) -> None:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        diffs = list(ex.map(lambda r: fetch_diff(repo, r["sha"], cap), recs))
    for r, d in zip(recs, diffs):
        r["diff"] = d


async def _attach_branch_commits(cfg: cfgmod.Config, github: str, recs: list[dict]) -> None:
    owner, repo = github.split("/", 1)
    gh = GitHubGraphQL(
        token=resolve_token(cfgmod.env("GITHUB_TOKEN")),
        batch_size=int(cfg.github.get("graphql_batch_size", 25)),
        max_concurrency=int(cfg.github.get("max_concurrency", 6)),
        commits_per_pr=int(cfg.github.get("commits_per_pr", 100)),
        min_remaining_points=int(cfg.github.get("min_remaining_points", 200)),
    )
    try:
        numbers = [r["number"] for r in recs]
        nodes = await gh.fetch_prs(owner, repo, numbers)
    finally:
        await gh.aclose()

    for r in recs:
        node = nodes.get(r["number"])
        if not node:
            r["branch_commits"] = []
            r["commit_messages"] = []
            continue
        flat = normalize_pr_node(node)
        # Prefer GitHub's exact PR-level diffstat when present.
        for k in ("additions", "deletions", "files_changed", "created_at", "merged_at", "pr_author_login"):
            if flat.get(k) is not None:
                r[k] = flat[k]
        r["branch_commits"] = flat["branch_commits"]
        r["branch_commit_count"] = flat.get("branch_commit_count")
        r["commit_messages"] = [c["message"] for c in flat["branch_commits"] if c.get("message")]


async def extract_repo(
    cfg: cfgmod.Config,
    repo_name: str,
    *,
    limit: int | None = None,
    fetch_branch_commits: bool | None = None,
) -> Path:
    rc = cfg.repo(repo_name)
    cap = int(cfg.extract.get("max_diff_chars", 50000))
    pr_regex = cfg.extract.get("pr_number_regex", r"\(#(\d+)\)\s*$")
    if fetch_branch_commits is None:
        fetch_branch_commits = bool(cfg.extract.get("fetch_branch_commits", True))

    print(f"[extract] parsing git log for {repo_name} ({rc.path}) …")
    recs = parse_git_log(rc.path, pr_regex)
    print(f"[extract] found {len(recs)} merged PRs")
    if limit:
        recs = recs[:limit]
        print(f"[extract] limited to {len(recs)}")

    print("[extract] reading diffs from local git …")
    _attach_diffs(rc.path, recs, cap)

    if fetch_branch_commits and rc.github:
        print(f"[extract] fetching branch commits via GitHub GraphQL ({rc.github}) …")
        await _attach_branch_commits(cfg, rc.github, recs)
    else:
        for r in recs:
            r["branch_commits"] = []
            r["commit_messages"] = []
        if fetch_branch_commits and not rc.github:
            print("[extract] WARNING: no github owner/repo configured; branch commits skipped")

    out = cfg.data_dir(repo_name) / "prs.jsonl"
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"[extract] wrote {len(recs)} records -> {out}")
    return out


def load_prs(cfg: cfgmod.Config, repo_name: str) -> list[dict]:
    path = cfg.data_dir(repo_name) / "prs.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `extract` first")
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # A PR number can appear on more than one commit (cherry-pick / re-land). Keep
    # the first occurrence (newest, since git log is newest-first) so every PR is
    # unique across judging, batching, and analysis.
    seen: set[int] = set()
    out = []
    for r in recs:
        n = r["number"]
        if n in seen:
            continue
        seen.add(n)
        out.append(r)
    return out
