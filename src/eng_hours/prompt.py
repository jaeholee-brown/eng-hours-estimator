"""The LLM-judge prompt.

Adapted from Amy Deng / METR's "estimating time without AI" judge prompt
(https://metr.org/notes/2026-02-17-exploratory-transcript-analysis-for-estimating-time-savings-from-coding-agents/#appendix-a-llm-judge-prompts).

Two adaptations vs. the original:

1. The original judge reads a *compressed agent transcript*. We do not have
   transcripts — we have **git history of a single merged PR**: the title, the
   description, the branch commit messages, and the full net diff. The prompt is
   rewritten to reason about that evidence instead of user/assistant turns.

2. The original asks for the time "an experienced software engineer who has full
   context would take." We make the counterfactual explicit and unambiguous: how
   long **one unassisted human software engineer, working alone, with NO AI tools
   of any kind**, would take to produce this merged change. METR's notion of "net
   successful output" still applies — a merged PR is, by definition, accepted
   output, so we estimate the time to produce exactly what landed.

The model is told to think out loud and then end with a single parseable line:
    FINAL_ESTIMATE_HOURS: <number>
We do not use any reasoning/extended-thinking mode; the "thinking" is just the
visible prose before that line.
"""
from __future__ import annotations

from textwrap import dedent
from typing import Any

SYSTEM = dedent(
    """\
    You are an expert software-engineering estimator. You read the git history of a
    single merged pull request (PR) and estimate how long it would have taken a
    human to produce it WITHOUT any AI assistance.

    THE QUESTION YOU ARE ANSWERING
    Estimate the number of engineering hours that ONE experienced software engineer,
    working ALONE and WITHOUT any AI tools whatsoever (no Copilot, no Codex, no
    Claude, no ChatGPT, no AI autocomplete — only a normal editor, compiler, docs,
    and a web browser), would need to produce the NET output of this PR: the exact
    code change that was merged.

    This is a counterfactual "time without AI" estimate. It is NOT how long the PR
    actually took, and NOT how long it would take with AI help. Picture a competent
    engineer who already has reasonable context on this codebase, doing the work by
    hand.

    WHAT TO COUNT
    - The PR was merged, so treat its final diff as accepted, successful output.
      Estimate the time to design, write, and verify exactly that change.
    - Include the realistic surrounding work an unassisted engineer cannot skip:
      understanding the relevant code, designing the approach, writing the change,
      writing/adjusting tests that are part of the diff, debugging, and iterating to
      a working, review-ready state.
    - Judge complexity from the DIFF, not just its size. Lines of code are a weak
      signal: a 500-line mechanical rename, generated file, or config/vendored bump
      is fast; a 60-line change to subtle concurrency, parsing, or algorithmic logic
      can take a day. Read what actually changed and reason about its difficulty for
      a skilled human.

    WHAT NOT TO COUNT
    - Do not add time for code review latency, CI waiting, meetings, or other
      process overhead — only hands-on engineering time.
    - Do not count work that is not in this PR. If the diff merely reverts, vendors,
      or auto-generates code, estimate the (small) human time to produce that net
      result, not the time to author what was generated.
    - Largely mechanical or trivial changes (typo fixes, dependency version bumps,
      formatting, generated lockfiles, simple config edits) should receive small
      estimates even when the diff is large.

    CALIBRATION (rough anchors for an unassisted engineer)
    - Trivial one-liner / typo / version bump: 0.1-0.5 h
    - Small, localized, well-understood change: 0.5-2 h
    - Moderate feature or fix touching a few files with some new logic: 2-8 h
    - Substantial feature, cross-cutting refactor, or subtle bug fix: 1-3 days (8-24 h)
    - Large, complex, or novel system change: multiple days (24 h+)
    Use your judgment; these are anchors, not limits.

    HOW TO RESPOND
    First think out loud, briefly: identify what the PR does, judge the difficulty of
    the actual changes, and note anything that makes it faster or slower for an
    unassisted human. Keep it concise — a few sentences to a short paragraph.

    Then, on the FINAL line and nothing after it, output exactly:
    FINAL_ESTIMATE_HOURS: <number>

    where <number> is a single positive decimal number of hours (e.g. 0.3, 2.5, 14,
    40). Do not give a range, units, or any text after the number on that line.
    """
)


def build_user_prompt(pr: dict[str, Any], *, max_diff_chars: int, max_commit_messages: int) -> str:
    """Render a PR record into the user message fed to every judge model."""
    title = pr.get("title") or pr.get("subject") or ""
    number = pr.get("number")
    body = (pr.get("body") or "").strip()

    files_changed = pr.get("files_changed")
    additions = pr.get("additions")
    deletions = pr.get("deletions")

    # Branch commit messages (subjects) give a sense of the work's shape.
    commit_msgs: list[str] = pr.get("commit_messages") or []
    shown = commit_msgs[:max_commit_messages]
    more = len(commit_msgs) - len(shown)

    diff = pr.get("diff") or ""
    truncated = False
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars]
        truncated = True

    parts: list[str] = []
    parts.append(f"# Merged PR #{number}: {title}".rstrip())

    stat_bits = []
    if files_changed is not None:
        stat_bits.append(f"{files_changed} files changed")
    if additions is not None:
        stat_bits.append(f"+{additions}")
    if deletions is not None:
        stat_bits.append(f"-{deletions}")
    if stat_bits:
        parts.append("Diffstat: " + ", ".join(stat_bits))

    if body:
        if len(body) > 4000:
            body = body[:4000] + "\n…[description truncated]…"
        parts.append("## PR description\n" + body)

    if shown:
        lines = "\n".join(f"- {m}" for m in shown)
        if more > 0:
            lines += f"\n- …and {more} more commit(s)"
        parts.append("## Branch commit messages\n" + lines)

    diff_header = "## Net diff (the merged change)"
    if truncated:
        diff_header += f"\n[diff truncated to {max_diff_chars} characters]"
    parts.append(diff_header + "\n```diff\n" + diff + "\n```")

    parts.append(
        "Now estimate the unassisted-human engineering hours for this merged PR. "
        "Think out loud briefly, then end with `FINAL_ESTIMATE_HOURS: <number>`."
    )
    return "\n\n".join(parts)
