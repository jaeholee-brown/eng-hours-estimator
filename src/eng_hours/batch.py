"""Batch-API path for the ensemble judge (50% cheaper, asynchronous).

Same models, same prompt, no reasoning — just submitted to each provider's Batch
API instead of called in real time. Results are written back into the SAME
data/<repo>/estimates.jsonl with the SAME schema as the real-time judge, keyed by
(number, model_key). So batch does NOT interfere with continuation: whatever was
already judged in real time is reused, and batch only fills the remaining pairs.
A later real-time `judge` run would in turn skip everything batch produced.

Two phases (batch is async):
  * submit  — compute remaining (PR, model) pairs, submit one batch per provider
              (Gemini is chunked into inline batches), record ids in batches.json
  * collect — poll each pending batch; when done, parse + append to estimates.jsonl

Per provider: Anthropic Message Batches and OpenAI Batch use a native custom_id
(= the PR number, since each batch is single-model). Gemini inline batches have no
custom_id, so we map responses to PR numbers by submission order (preserved in
`dest.inlined_responses`).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import config as cfgmod
from .extract import load_prs
from .judge import _load_done
from .parse import extract_hours
from .prompt import SYSTEM, build_user_prompt

BATCHES_FILE = "batches.json"


# ── helpers ──────────────────────────────────────────────────────────────────
def _prompt_for(cfg: cfgmod.Config, pr: dict) -> str:
    return build_user_prompt(
        pr,
        max_diff_chars=int(cfg.extract.get("max_diff_chars", 50000)),
        max_commit_messages=int(cfg.extract.get("max_commit_messages", 40)),
    )


def _remaining(cfg, repo, model_keys, limit):
    prs = load_prs(cfg, repo)
    if limit:
        prs = prs[:limit]
    done = _load_done(cfg.data_dir(repo) / "estimates.jsonl")
    models = [m for m in cfg.models if (not model_keys or m.key in model_keys)]
    by_model: dict[str, list[dict]] = {}
    for m in models:
        todo = [pr for pr in prs if (pr["number"], m.key) not in done]
        if todo:
            by_model[m.key] = todo
    return models, by_model


def _append_records(cfg, repo, records: list[dict]) -> None:
    path = cfg.data_dir(repo) / "estimates.jsonl"
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _batches_path(cfg, repo) -> Path:
    return cfg.data_dir(repo) / BATCHES_FILE


def _load_batches(cfg, repo) -> dict:
    p = _batches_path(cfg, repo)
    return json.loads(p.read_text()) if p.exists() else {"batches": []}


def _save_batches(cfg, repo, state: dict) -> None:
    _batches_path(cfg, repo).write_text(json.dumps(state, indent=2))


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# ── submit ───────────────────────────────────────────────────────────────────
def submit(cfg, repo, *, model_keys=None, limit=None, gemini_chunk=200, openai_chunk=3000) -> dict:
    models, by_model = _remaining(cfg, repo, model_keys, limit)
    if not by_model:
        print("[batch] nothing remaining to submit — all pairs already judged.")
        return _load_batches(cfg, repo)

    mt = int(cfg.judge.get("max_output_tokens", 1500))
    temp = cfg.judge.get("temperature", 0.0)
    state = _load_batches(cfg, repo)
    entries: list[dict] = state["batches"]

    for m in models:
        todo = by_model.get(m.key)
        if not todo:
            continue
        print(f"[batch] {m.key}: submitting {len(todo)} requests via {m.provider} batch …")
        if m.provider == "anthropic":
            entries.append(_submit_anthropic(cfg, m, todo, mt, temp))
        elif m.provider == "openai":
            for ch in _chunks(todo, openai_chunk):
                entries.append(_submit_openai(cfg, m, ch, mt))
        elif m.provider == "google":
            for ch in _chunks(todo, gemini_chunk):
                entries.append(_submit_gemini(cfg, m, ch, mt, temp))
        else:
            print(f"[batch] unknown provider {m.provider}; skipping")

    state["batches"] = entries
    _save_batches(cfg, repo, state)
    pending = sum(1 for e in entries if not e.get("collected"))
    print(f"[batch] submitted. {pending} batch job(s) pending. "
          f"Run `eng-hours batch {repo} --collect` once they finish "
          f"(usually < 1h; up to 24h).")
    return state


def _submit_anthropic(cfg, m, todo, max_tokens, temp) -> dict:
    from anthropic import Anthropic

    client = Anthropic(api_key=cfgmod.env("ANTHROPIC_API_KEY"))
    requests = [
        {
            "custom_id": str(pr["number"]),
            "params": {
                "model": m.model,
                "max_tokens": max_tokens,
                "temperature": temp,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": _prompt_for(cfg, pr)}],
            },
        }
        for pr in todo
    ]
    batch = client.messages.batches.create(requests=requests)
    return {"provider": "anthropic", "model_key": m.key, "batch_id": batch.id,
            "n": len(todo), "collected": False}


def _submit_openai(cfg, m, todo, max_tokens) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=cfgmod.env("OPENAI_API_KEY"))
    lines = []
    for pr in todo:
        body = {
            "model": m.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _prompt_for(cfg, pr)},
            ],
            "max_completion_tokens": max_tokens,
        }
        if m.reasoning == "off":
            body["reasoning_effort"] = "none"  # gpt-5.x: "none" disables reasoning ("minimal" is rejected)
        lines.append(json.dumps({"custom_id": str(pr["number"]), "method": "POST",
                                 "url": "/v1/chat/completions", "body": body}))
    data = ("\n".join(lines)).encode()
    fobj = client.files.create(file=("batch_input.jsonl", data), purpose="batch")
    batch = client.batches.create(input_file_id=fobj.id,
                                  endpoint="/v1/chat/completions",
                                  completion_window="24h")
    return {"provider": "openai", "model_key": m.key, "batch_id": batch.id,
            "input_file_id": fobj.id, "n": len(todo), "collected": False}


def _gemini_thinking(reasoning):
    from google.genai import types

    if reasoning != "off":
        return None
    for kwargs in ({"thinking_level": "low"}, {"thinking_budget": 0}):
        try:
            return types.ThinkingConfig(**kwargs)
        except (TypeError, ValueError):
            continue
    return None


def _submit_gemini(cfg, m, todo, max_tokens, temp) -> dict:
    from google.genai import Client, types

    client = Client(api_key=cfgmod.env("GEMINI_API_KEY"))
    thinking = _gemini_thinking(m.reasoning)
    cfg_kwargs = dict(system_instruction=SYSTEM, max_output_tokens=max_tokens,
                      temperature=temp if temp is not None else 0.0)
    if thinking is not None:
        cfg_kwargs["thinking_config"] = thinking
    gen_cfg = types.GenerateContentConfig(**cfg_kwargs)
    src = [
        types.InlinedRequest(model=m.model, contents=_prompt_for(cfg, pr), config=gen_cfg)
        for pr in todo
    ]
    job = client.batches.create(model=m.model, src=src)
    return {"provider": "google", "model_key": m.key, "batch_name": job.name,
            "numbers": [pr["number"] for pr in todo], "n": len(todo), "collected": False}


# ── collect ──────────────────────────────────────────────────────────────────
def collect(cfg, repo) -> dict:
    state = _load_batches(cfg, repo)
    entries = state.get("batches", [])
    if not entries:
        print("[batch] no submitted batches found. Run `--submit` first.")
        return state

    prices = {m.key: (m.price_in, m.price_out) for m in cfg.models}
    still_pending = 0
    for e in entries:
        if e.get("collected"):
            continue
        try:
            recs, status = _collect_one(cfg, e)
        except Exception as ex:  # noqa: BLE001
            print(f"[batch] {e['model_key']} ({e.get('batch_id') or e.get('batch_name')}): "
                  f"error polling — {type(ex).__name__}: {str(ex)[:160]}")
            still_pending += 1
            continue
        if recs is None:
            print(f"[batch] {e['model_key']}: {status} (not ready)")
            still_pending += 1
            continue
        _append_records(cfg, repo, recs)
        e["collected"] = True
        ok = sum(1 for r in recs if r.get("hours") is not None)
        print(f"[batch] {e['model_key']}: {status}; wrote {len(recs)} ({ok} ok)")
        _save_batches(cfg, repo, state)  # checkpoint after each

    _save_batches(cfg, repo, state)
    from .cost import actual_cost
    print()
    if still_pending:
        print(f"[batch] {still_pending} batch job(s) still pending — re-run --collect later.\n")
    actual_cost(cfg, repo)
    return state


def _mk_record(model_key, number, text, tin, tout):
    hours, method = extract_hours(text or "")
    return {"number": number, "model_key": model_key, "hours": hours,
            "parse_method": method, "tokens_in": tin or 0, "tokens_out": tout or 0,
            "raw_tail": (text or "")[-1500:]}


def _collect_one(cfg, e):
    if e["provider"] == "anthropic":
        return _collect_anthropic(e)
    if e["provider"] == "openai":
        return _collect_openai(e)
    if e["provider"] == "google":
        return _collect_gemini(e)
    return None, "unknown provider"


def _collect_anthropic(e):
    from anthropic import Anthropic

    client = Anthropic(api_key=cfgmod.env("ANTHROPIC_API_KEY"))
    b = client.messages.batches.retrieve(e["batch_id"])
    if b.processing_status != "ended":
        return None, b.processing_status
    recs = []
    for entry in client.messages.batches.results(e["batch_id"]):
        number = int(entry.custom_id)
        if entry.result.type == "succeeded":
            msg = entry.result.message
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            recs.append(_mk_record(e["model_key"], number, text,
                                   msg.usage.input_tokens, msg.usage.output_tokens))
        else:
            recs.append({"number": number, "model_key": e["model_key"], "hours": None,
                         "error": f"batch:{entry.result.type}"})
    return recs, "ended"


def _collect_openai(e):
    from openai import OpenAI

    client = OpenAI(api_key=cfgmod.env("OPENAI_API_KEY"))
    b = client.batches.retrieve(e["batch_id"])
    if b.status in ("validating", "in_progress", "finalizing"):
        return None, b.status
    recs = []
    # Successful requests land in the output file …
    if getattr(b, "output_file_id", None):
        for line in client.files.content(b.output_file_id).text.splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            number = int(o["custom_id"])
            resp = o.get("response") or {}
            if resp.get("status_code") == 200:
                body = resp["body"]
                content = body["choices"][0]["message"].get("content") or ""
                usage = body.get("usage", {})
                recs.append(_mk_record(e["model_key"], number, content,
                                       usage.get("prompt_tokens"), usage.get("completion_tokens")))
            else:
                recs.append({"number": number, "model_key": e["model_key"], "hours": None,
                             "error": f"batch_http:{resp.get('status_code')}"})
    # … failed requests land in the error file (record them so we don't re-poll forever).
    if getattr(b, "error_file_id", None):
        for line in client.files.content(b.error_file_id).text.splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            resp = o.get("response") or {}
            msg = ((resp.get("body") or {}).get("error") or {}).get("message", "")
            recs.append({"number": int(o["custom_id"]), "model_key": e["model_key"],
                         "hours": None, "error": f"batch_http:{resp.get('status_code')}:{msg[:120]}"})
    return recs, b.status


def _collect_gemini(e):
    from google.genai import Client

    client = Client(api_key=cfgmod.env("GEMINI_API_KEY"))
    job = client.batches.get(name=e["batch_name"])
    state = str(job.state)
    if not ("SUCCEEDED" in state or "PARTIALLY" in state or "FAILED" in state):
        return None, state
    dest = getattr(job, "dest", None)
    responses = getattr(dest, "inlined_responses", None) if dest else None
    numbers = e["numbers"]
    recs = []
    if not responses:
        for number in numbers:
            recs.append({"number": number, "model_key": e["model_key"], "hours": None,
                         "error": f"batch:{state}:no_responses"})
        return recs, state
    for number, ir in zip(numbers, responses):
        err = getattr(ir, "error", None)
        if err:
            recs.append({"number": number, "model_key": e["model_key"], "hours": None,
                         "error": f"batch:{str(err)[:120]}"})
            continue
        resp = getattr(ir, "response", None)
        text = getattr(resp, "text", "") if resp else ""
        um = getattr(resp, "usage_metadata", None) if resp else None
        tin = getattr(um, "prompt_token_count", 0) or 0
        tout = ((getattr(um, "candidates_token_count", 0) or 0)
                + (getattr(um, "thoughts_token_count", 0) or 0)) if um else 0
        recs.append(_mk_record(e["model_key"], number, text, tin, tout))
    return recs, state


def status(cfg, repo) -> None:
    state = _load_batches(cfg, repo)
    entries = state.get("batches", [])
    if not entries:
        print("[batch] no batches submitted.")
        return
    for e in entries:
        bid = e.get("batch_id") or e.get("batch_name")
        flag = "collected" if e.get("collected") else "PENDING"
        print(f"  {e['model_key']:<16} {e['provider']:<10} n={e.get('n'):<6} {flag}  {bid}")
