"""Command-line interface.

    python -m eng_hours list
    python -m eng_hours extract  <repo> [--limit N] [--no-branch-commits]
    python -m eng_hours cost     <repo> [--sample N] [--output-tokens N] [--no-calibrate]
    python -m eng_hours judge    <repo> [--limit N] [--models haiku-4.5,gpt-5.4-mini]
    python -m eng_hours spend    <repo>                  # actual $ spent so far, per model
    python -m eng_hours analyze  <repo>
    python -m eng_hours plot     <repo>
    python -m eng_hours all      <repo> [--limit N]      # extract->judge->analyze->plot
    python -m eng_hours test     <repo> [--limit 3]      # tiny live smoke test

<repo> defaults to the only configured repo when there is exactly one.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from . import config as cfgmod


def _default_repo(cfg: cfgmod.Config, given: str | None) -> str:
    if given:
        return given
    if len(cfg.repos) == 1:
        return cfg.repos[0].name
    raise SystemExit(f"specify a repo: {[r.name for r in cfg.repos]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eng_hours", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list configured repos and models")

    pe = sub.add_parser("extract", help="extract per-PR records from git + GitHub")
    pe.add_argument("repo", nargs="?")
    pe.add_argument("--limit", type=int, default=None)
    pe.add_argument("--no-branch-commits", action="store_true",
                    help="skip the GitHub API; use only squash commits")

    pc = sub.add_parser("cost", help="estimate the cost of a full ensemble run")
    pc.add_argument("repo", nargs="?")
    pc.add_argument("--sample", type=int, default=30)
    pc.add_argument("--output-tokens", type=int, default=None)
    pc.add_argument("--no-calibrate", action="store_true")

    pj = sub.add_parser("judge", help="run the ensemble judge over extracted PRs")
    pj.add_argument("repo", nargs="?")
    pj.add_argument("--limit", type=int, default=None)
    pj.add_argument("--models", default=None, help="comma-separated model keys")

    pa = sub.add_parser("analyze", help="ensemble + contributor-day aggregation")
    pa.add_argument("repo", nargs="?")

    ps = sub.add_parser("spend", help="actual cost spent so far, per model (from token usage)")
    ps.add_argument("repo", nargs="?")

    pp = sub.add_parser("plot", help="render graphs")
    pp.add_argument("repo", nargs="?")

    pall = sub.add_parser("all", help="extract -> judge -> analyze -> plot")
    pall.add_argument("repo", nargs="?")
    pall.add_argument("--limit", type=int, default=None)

    pt = sub.add_parser("test", help="tiny live smoke test (few PRs, all models)")
    pt.add_argument("repo", nargs="?")
    pt.add_argument("--limit", type=int, default=3)

    args = p.parse_args(argv)
    cfg = cfgmod.load(args.config)

    if args.cmd == "list":
        print("Repos:")
        for r in cfg.repos:
            print(f"  - {r.name}  path={r.path}  github={r.github}")
        print("Models:")
        for m in cfg.models:
            print(f"  - {m.key:<16} {m.provider}/{m.model}  "
                  f"${m.price_in}/${m.price_out} per 1M  reasoning={m.reasoning}")
        return 0

    repo = _default_repo(cfg, getattr(args, "repo", None))

    if args.cmd == "extract":
        from .extract import extract_repo
        asyncio.run(extract_repo(
            cfg, repo, limit=args.limit,
            fetch_branch_commits=not args.no_branch_commits,
        ))
    elif args.cmd == "cost":
        from .cost import estimate_cost
        estimate_cost(cfg, repo, sample=args.sample, output_tokens=args.output_tokens,
                      calibrate=not args.no_calibrate)
    elif args.cmd == "judge":
        from .judge import judge_repo
        keys = args.models.split(",") if args.models else None
        asyncio.run(judge_repo(cfg, repo, model_keys=keys, limit=args.limit))
    elif args.cmd == "spend":
        from .cost import actual_cost
        actual_cost(cfg, repo)
    elif args.cmd == "analyze":
        from .analyze import analyze
        analyze(cfg, repo)
    elif args.cmd == "plot":
        from .plots import make_plots
        make_plots(cfg, repo)
    elif args.cmd == "all":
        from .extract import extract_repo
        from .judge import judge_repo
        from .analyze import analyze
        from .plots import make_plots
        asyncio.run(extract_repo(cfg, repo, limit=args.limit))
        asyncio.run(judge_repo(cfg, repo, limit=args.limit))
        analyze(cfg, repo)
        make_plots(cfg, repo)
    elif args.cmd == "test":
        asyncio.run(_smoke_test(cfg, repo, args.limit))
    return 0


async def _smoke_test(cfg: cfgmod.Config, repo: str, limit: int) -> None:
    """Extract a few PRs and judge them with every model, printing each answer."""
    from .extract import extract_repo, load_prs
    from .judge import judge_repo, load_estimates

    print(f"=== smoke test: {repo}, {limit} PRs, all models ===")
    await extract_repo(cfg, repo, limit=limit)
    await judge_repo(cfg, repo, limit=limit)
    prs = {p["number"]: p for p in load_prs(cfg, repo)}
    by_pr: dict[int, list] = {}
    for e in load_estimates(cfg, repo):
        by_pr.setdefault(e["number"], []).append(e)
    for num, ests in sorted(by_pr.items()):
        title = (prs.get(num, {}).get("title") or "")[:70]
        print(f"\nPR #{num}: {title}")
        for e in ests:
            h = e.get("hours")
            tag = f"{h}h" if h is not None else f"ERR {e.get('error','')}"
            print(f"   {e['model_key']:<16} {tag:<10} ({e.get('parse_method','')})")


if __name__ == "__main__":
    sys.exit(main())
