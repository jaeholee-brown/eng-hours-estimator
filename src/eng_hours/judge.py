"""Async ensemble judging.

Runs every (PR × model) pair concurrently (bounded per-model concurrency, assuming
max-tier rate limits), parses each model's free-text answer into a final hours
number, and writes one JSONL record per (PR, model) to data/<repo>/estimates.jsonl.

Resumable: existing (number, model_key) pairs are skipped, so a re-run only fills
gaps. Each call is wrapped in exponential-backoff retry on top of the SDKs' own
retries.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from . import config as cfgmod
from .extract import load_prs
from .parse import extract_hours
from .prompt import SYSTEM, build_user_prompt
from .providers import make_provider


def _load_done(path: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Only successful estimates count as done — errored calls retry on re-run.
        if r.get("hours") is not None:
            done.add((r["number"], r["model_key"]))
    return done


async def _with_retry(coro_factory, retries: int):
    delay = 2.0
    last: Exception | None = None
    for _ in range(retries):
        try:
            return await coro_factory()
        except Exception as e:  # noqa: BLE001 — provider/transport errors vary
            last = e
            await asyncio.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 60)
    raise last  # type: ignore[misc]


async def judge_repo(
    cfg: cfgmod.Config,
    repo_name: str,
    *,
    model_keys: list[str] | None = None,
    limit: int | None = None,
) -> Path:
    prs = load_prs(cfg, repo_name)
    if limit:
        prs = prs[:limit]
    models = [m for m in cfg.models if (not model_keys or m.key in model_keys)]
    if not models:
        raise ValueError("no models selected")

    jc = cfg.judge
    max_tokens = int(jc.get("max_output_tokens", 1500))
    temperature = jc.get("temperature", 0.0)
    retries = int(jc.get("retries", 5))
    cap = int(cfg.extract.get("max_diff_chars", 50000))
    max_msgs = int(cfg.extract.get("max_commit_messages", 40))

    out_path = cfg.data_dir(repo_name) / "estimates.jsonl"
    done = _load_done(out_path)
    providers = {m.key: make_provider(m) for m in models}
    sems = {m.key: asyncio.Semaphore(m.max_concurrency) for m in models}
    prices = {m.key: (m.price_in, m.price_out) for m in models}
    lock = asyncio.Lock()
    f = out_path.open("a")

    total = sum(1 for m in models for pr in prs if (pr["number"], m.key) not in done)
    print(f"[judge] {len(prs)} PRs × {len(models)} models; {total} calls to make "
          f"({len(done)} already done)")
    if total == 0:
        f.close()
        return out_path

    from collections import defaultdict

    from .cost import token_cost

    counter = {"n": 0, "ok": 0, "fail": 0}
    # live per-model spend, accumulated from actual token usage (no reasoning,
    # so output tokens are fully counted)
    spend = {m.key: {"tin": 0, "tout": 0, "usd": 0.0} for m in models}
    start = time.time()

    async def judge_one(pr: dict[str, Any], m) -> None:
        user = build_user_prompt(pr, max_diff_chars=cap, max_commit_messages=max_msgs)
        provider = providers[m.key]
        rec: dict[str, Any] = {"number": pr["number"], "model_key": m.key}
        try:
            async with sems[m.key]:
                res = await _with_retry(
                    lambda: provider.complete(
                        SYSTEM, user,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        reasoning=m.reasoning,
                    ),
                    retries,
                )
            hours, method = extract_hours(res.text)
            rec.update(
                hours=hours,
                parse_method=method,
                tokens_in=res.tokens_in,
                tokens_out=res.tokens_out,
                raw_tail=res.text[-1500:],
            )
        except Exception as e:  # noqa: BLE001
            rec.update(hours=None, error=f"{type(e).__name__}: {str(e)[:200]}")

        async with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            counter["n"] += 1
            if rec.get("hours") is not None:
                counter["ok"] += 1
            else:
                counter["fail"] += 1
            tin = int(rec.get("tokens_in") or 0)
            tout = int(rec.get("tokens_out") or 0)
            pin, pout = prices[m.key]
            sp = spend[m.key]
            sp["tin"] += tin
            sp["tout"] += tout
            sp["usd"] += token_cost(pin, pout, tin, tout)
            if counter["n"] % 50 == 0 or counter["n"] == total:
                rate = counter["n"] / max(1e-9, time.time() - start)
                eta = (total - counter["n"]) / max(1e-9, rate)
                total_usd = sum(s["usd"] for s in spend.values())
                print(f"[judge] {counter['n']}/{total}  ok={counter['ok']} "
                      f"fail={counter['fail']}  {rate:.1f}/s  eta {eta/60:.1f}m  "
                      f"spent ${total_usd:,.2f}")

    tasks = [
        judge_one(pr, m)
        for m in models
        for pr in prs
        if (pr["number"], m.key) not in done
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        f.close()
    print(f"[judge] done: {counter['ok']} ok, {counter['fail']} failed -> {out_path}")

    # Per-model actual spend for this run (newly-made calls only).
    print("\n──────── actual cost this run (per model) ────────")
    print(f"{'model':<16}{'tok in':>14}{'tok out':>12}{'$ this run':>12}")
    print("-" * 54)
    run_total = 0.0
    for k, s in spend.items():
        run_total += s["usd"]
        print(f"{k:<16}{s['tin']:>14,}{s['tout']:>12,}{s['usd']:>12,.2f}")
    print("-" * 54)
    print(f"{'TOTAL':<16}{'':>14}{'':>12}{run_total:>12,.2f}")
    print("(cumulative spend across all runs: `eng-hours spend " + repo_name + "`)\n")
    return out_path


def load_estimates(cfg: cfgmod.Config, repo_name: str) -> list[dict]:
    path = cfg.data_dir(repo_name) / "estimates.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `judge` first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
