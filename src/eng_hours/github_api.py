"""Async GitHub GraphQL client for fetching PR branch commits.

Speed-optimized for fetching commit *metadata* (authors + dates) across thousands
of merged PRs. The diffs themselves come from local git — free and fast — so the
API is only used for what git-after-squash-merge loses: the original branch
commits and their authoring dates.

Rate-limit strategy (the "optimize for speed" brief):
  * GraphQL over REST — one query fetches `graphql_batch_size` PRs at once (each
    aliased), each with up to `commits_per_pr` commits, collapsing what would be
    tens of thousands of REST calls into a few hundred queries.
  * Bounded concurrency (`max_concurrency`) keeps us under the secondary limits
    (≤100 concurrent requests, ≲900 points/min) while saturating throughput.
  * Header/`rateLimit`-aware: we read `rateLimit { remaining resetAt }` from every
    response and sleep until reset when remaining points drop below a floor; we
    honor `Retry-After` and back off on 403/429 secondary-limit responses.
"""
from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Any

import httpx

GRAPHQL_URL = "https://api.github.com/graphql"

_PR_FIELDS = """
    number
    additions
    deletions
    changedFiles
    createdAt
    mergedAt
    author { login }
    commits(first: %(commits_per_pr)d) {
      totalCount
      nodes {
        commit {
          oid
          authoredDate
          committedDate
          message
          author { email name user { login } }
        }
      }
    }
"""


def resolve_token(explicit: str | None) -> str:
    """Use an explicit token, else fall back to the `gh` CLI's stored token."""
    if explicit:
        return explicit
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        )
        tok = out.stdout.strip()
        if tok:
            return tok
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    raise RuntimeError(
        "No GitHub token. Set GITHUB_TOKEN in .env or run `gh auth login`."
    )


class GitHubGraphQL:
    def __init__(
        self,
        token: str,
        *,
        batch_size: int = 25,
        max_concurrency: int = 6,
        commits_per_pr: int = 100,
        min_remaining_points: int = 200,
    ):
        self.token = token
        self.batch_size = batch_size
        self.commits_per_pr = commits_per_pr
        self.min_remaining_points = min_remaining_points
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={
                "Authorization": f"bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "eng-hours-estimator",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_query(self, owner: str, repo: str, numbers: list[int]) -> str:
        fields = _PR_FIELDS % {"commits_per_pr": self.commits_per_pr}
        aliases = "\n".join(
            f"  pr{i}: pullRequest(number: {n}) {{{fields}}}" for i, n in enumerate(numbers)
        )
        return (
            f'query {{\n  repository(owner: "{owner}", name: "{repo}") {{\n'
            f"{aliases}\n  }}\n  rateLimit {{ remaining cost resetAt limit }}\n}}"
        )

    async def _post(self, query: str) -> dict[str, Any]:
        backoff = 2.0
        for attempt in range(8):
            async with self._sem:
                try:
                    resp = await self._client.post(GRAPHQL_URL, json={"query": query})
                except httpx.HTTPError:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue

            # Secondary / abuse / primary limit
            if resp.status_code in (403, 429):
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, 60)
                continue
            if resp.status_code >= 500:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            data = resp.json()
            rl = (data.get("data") or {}).get("rateLimit") or {}
            remaining = rl.get("remaining")
            if remaining is not None and remaining < self.min_remaining_points:
                reset_at = rl.get("resetAt")
                wait = _seconds_until(reset_at)
                if wait > 0:
                    await asyncio.sleep(min(wait + 1, 3600))

            # GraphQL transport errors (whole query failed). Per-alias nulls are fine.
            if "errors" in data and not data.get("data"):
                raise RuntimeError(f"GraphQL error: {data['errors'][:2]}")
            return data
        raise RuntimeError("GraphQL request failed after retries")

    async def fetch_prs(self, owner: str, repo: str, numbers: list[int]) -> dict[int, dict]:
        """Return {pr_number: pr_node} for every number that resolves to a PR."""
        batches = [
            numbers[i : i + self.batch_size]
            for i in range(0, len(numbers), self.batch_size)
        ]
        out: dict[int, dict] = {}

        async def run(batch: list[int]) -> None:
            data = await self._post(self._build_query(owner, repo, batch))
            repo_node = (data.get("data") or {}).get("repository") or {}
            for i, n in enumerate(batch):
                node = repo_node.get(f"pr{i}")
                if node:
                    out[n] = node

        await asyncio.gather(*(run(b) for b in batches))
        return out


def _seconds_until(iso_ts: str | None) -> float:
    if not iso_ts:
        return 0.0
    # "2026-06-16T23:04:07Z"
    try:
        t = time.strptime(iso_ts.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z")
        return max(0.0, time.mktime(t) - time.time())
    except ValueError:
        return 0.0


def normalize_pr_node(node: dict) -> dict:
    """Flatten a GraphQL PR node into the fields the extractor stores."""
    commits = []
    cnode = (node.get("commits") or {})
    for c in cnode.get("nodes") or []:
        commit = c.get("commit") or {}
        author = commit.get("author") or {}
        user = author.get("user") or {}
        commits.append(
            {
                "oid": commit.get("oid"),
                "authored_date": commit.get("authoredDate"),
                "committed_date": commit.get("committedDate"),
                "email": author.get("email"),
                "name": author.get("name"),
                "login": user.get("login"),
                "message": (commit.get("message") or "").splitlines()[0] if commit.get("message") else "",
            }
        )
    return {
        "additions": node.get("additions"),
        "deletions": node.get("deletions"),
        "files_changed": node.get("changedFiles"),
        "created_at": node.get("createdAt"),
        "merged_at": node.get("mergedAt"),
        "pr_author_login": (node.get("author") or {}).get("login"),
        "branch_commits": commits,
        "branch_commit_count": cnode.get("totalCount"),
    }
