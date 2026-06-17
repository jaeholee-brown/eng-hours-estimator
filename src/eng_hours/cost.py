"""Precise cost estimate for a full ensemble run.

Builds the actual judge prompt for every extracted PR, counts input tokens, and
multiplies by each model's configured price. Input tokens are estimated from
character counts via a ratio that is, by default, *calibrated* against the
Anthropic tokenizer on a small live sample (`count_tokens`, which is free) — so
the estimate reflects the real prompts, not a guess. Output tokens default to a
configured average, or the measured mean from any estimates already produced.

Reports per-model and total cost for a real-time run, plus the Batch-API variant
(50% off on Anthropic/OpenAI) for reference.
"""
from __future__ import annotations

from typing import Any

from . import config as cfgmod
from .extract import load_prs
from .prompt import SYSTEM, build_user_prompt


def _calibrate_ratio(sample_prompts: list[str], default: float) -> float:
    """tokens-per-char, calibrated on Anthropic's tokenizer if a key is present."""
    key = cfgmod.env("ANTHROPIC_API_KEY")
    if not key:
        return default
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=key)
        tot_tok = tot_chars = 0
        for p in sample_prompts:
            r = client.messages.count_tokens(
                model="claude-haiku-4-5",
                system=SYSTEM,
                messages=[{"role": "user", "content": p}],
            )
            tot_tok += r.input_tokens
            tot_chars += len(SYSTEM) + len(p)
        if tot_chars:
            return tot_tok / tot_chars
    except Exception as e:  # noqa: BLE001
        print(f"[cost] calibration skipped ({type(e).__name__}); using default ratio")
    return default


def estimate_cost(
    cfg: cfgmod.Config,
    repo_name: str,
    *,
    sample: int = 30,
    output_tokens: int | None = None,
    chars_per_token: float = 3.6,
    calibrate: bool = True,
) -> dict[str, Any]:
    prs = load_prs(cfg, repo_name)
    cap = int(cfg.extract.get("max_diff_chars", 50000))
    max_msgs = int(cfg.extract.get("max_commit_messages", 40))
    n = len(prs)

    prompts = [build_user_prompt(p, max_diff_chars=cap, max_commit_messages=max_msgs) for p in prs]
    total_chars = sum(len(SYSTEM) + len(p) for p in prompts)

    default_ratio = 1.0 / chars_per_token
    if calibrate and n:
        step = max(1, n // sample)
        ratio = _calibrate_ratio(prompts[::step][:sample], default_ratio)
    else:
        ratio = default_ratio
    total_in_tokens = total_chars * ratio
    avg_in = total_in_tokens / n if n else 0

    # Output tokens: per-model measured mean if we have estimates for that model,
    # else the global measured mean, else the provided/default value.
    per_model_out = _measured_output_by_model(cfg, repo_name)
    global_out = output_tokens or _measured_output_tokens(cfg, repo_name) or 600

    print(f"[cost] {n} PRs; calibrated ratio = {ratio:.4f} tok/char "
          f"({1/ratio:.2f} char/tok); avg input ≈ {avg_in:,.0f} tok/PR\n")

    rows = []
    grand_realtime = grand_batch = 0.0
    for m in cfg.models:
        out_tok = output_tokens or per_model_out.get(m.key) or global_out
        in_cost = total_in_tokens / 1e6 * m.price_in
        out_cost = n * out_tok / 1e6 * m.price_out
        rt = in_cost + out_cost
        batch = rt * 0.5
        grand_realtime += rt
        grand_batch += batch
        rows.append({
            "model": m.key, "out_tok": out_tok, "in_usd": in_cost, "out_usd": out_cost,
            "realtime_usd": rt, "batch_usd": batch,
        })

    _print_table(rows, grand_realtime, grand_batch, n)
    return {
        "n_prs": n, "ratio": ratio, "avg_input_tokens": avg_in,
        "output_tokens": output_tokens, "per_model": rows,
        "total_realtime_usd": grand_realtime, "total_batch_usd": grand_batch,
    }


def token_cost(price_in: float, price_out: float, tin: int, tout: int) -> float:
    """USD for `tin` input + `tout` output tokens at per-1M prices."""
    return tin / 1e6 * price_in + tout / 1e6 * price_out


def actual_cost(cfg: cfgmod.Config, repo_name: str) -> dict[str, Any]:
    """Actual money spent so far, from the recorded token usage in estimates.jsonl,
    split per model. (No reasoning is used, so output tokens are fully counted.)"""
    import json

    path = cfg.data_dir(repo_name) / "estimates.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `judge` first")
    price = {m.key: (m.price_in, m.price_out) for m in cfg.models}

    agg: dict[str, dict[str, float]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = r["model_key"]
        a = agg.setdefault(k, {"calls": 0, "ok": 0, "tin": 0, "tout": 0})
        a["calls"] += 1
        if r.get("hours") is not None:
            a["ok"] += 1
        a["tin"] += int(r.get("tokens_in") or 0)
        a["tout"] += int(r.get("tokens_out") or 0)

    rows, grand = [], 0.0
    for k, a in agg.items():
        pin, pout = price.get(k, (0.0, 0.0))
        usd = token_cost(pin, pout, int(a["tin"]), int(a["tout"]))
        grand += usd
        rows.append({"model": k, "calls": int(a["calls"]), "ok": int(a["ok"]),
                     "tokens_in": int(a["tin"]), "tokens_out": int(a["tout"]),
                     "usd": usd})

    print(f"{'model':<16}{'calls':>8}{'tok in':>14}{'tok out':>12}{'actual $':>12}")
    print("-" * 62)
    for r in sorted(rows, key=lambda x: x["model"]):
        print(f"{r['model']:<16}{r['calls']:>8}{r['tokens_in']:>14,}"
              f"{r['tokens_out']:>12,}{r['usd']:>12,.2f}")
    print("-" * 62)
    print(f"{'TOTAL SPENT':<16}{'':>8}{'':>14}{'':>12}{grand:>12,.2f}\n")
    return {"per_model": rows, "total_usd": grand}


def _iter_estimates(cfg: cfgmod.Config, repo_name: str):
    import json

    path = cfg.data_dir(repo_name) / "estimates.jsonl"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _measured_output_tokens(cfg: cfgmod.Config, repo_name: str) -> float | None:
    vals = [r["tokens_out"] for r in _iter_estimates(cfg, repo_name) if r.get("tokens_out")]
    return (sum(vals) / len(vals)) if vals else None


def _measured_output_by_model(cfg: cfgmod.Config, repo_name: str) -> dict[str, float]:
    from collections import defaultdict

    sums: dict[str, list[int]] = defaultdict(list)
    for r in _iter_estimates(cfg, repo_name):
        if r.get("tokens_out"):
            sums[r["model_key"]].append(r["tokens_out"])
    return {k: sum(v) / len(v) for k, v in sums.items() if v}


def _print_table(rows, grand_rt, grand_batch, n):
    print(f"{'model':<16}{'out tok':>9}{'input $':>12}{'output $':>12}"
          f"{'real-time $':>14}{'batch $':>12}")
    print("-" * 75)
    for r in rows:
        print(f"{r['model']:<16}{int(r['out_tok']):>9}{r['in_usd']:>12,.2f}"
              f"{r['out_usd']:>12,.2f}{r['realtime_usd']:>14,.2f}{r['batch_usd']:>12,.2f}")
    print("-" * 75)
    print(f"{'ENSEMBLE TOTAL':<16}{'':>9}{'':>12}{'':>12}{grand_rt:>14,.2f}{grand_batch:>12,.2f}")
    print(f"\n≈ ${grand_rt/n:.4f} per PR across all models (real-time), {n:,} PRs.")
    print("(output tokens: per-model measured mean where available, else default; "
          "input tokens calibrated on the Anthropic tokenizer.)\n")
