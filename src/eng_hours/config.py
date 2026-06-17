"""Configuration + secret loading.

Loads `config.yaml` and `.env` from the project root. Resolves repo paths
(following symlinks) and exposes small typed accessors. Nothing here is
specific to a single repo or model — everything comes from config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = two levels up from this file (src/eng_hours/config.py -> root).
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelCfg:
    key: str
    provider: str
    model: str
    price_in: float          # USD per 1M input tokens
    price_out: float         # USD per 1M output tokens
    reasoning: str = "off"
    max_concurrency: int = 40


@dataclass
class RepoCfg:
    name: str
    path: Path
    github: str | None = None  # "owner/repo"


@dataclass
class Config:
    raw: dict[str, Any]
    repos: list[RepoCfg]
    models: list[ModelCfg]

    # convenience views
    extract: dict[str, Any] = field(default_factory=dict)
    github: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    analyze: dict[str, Any] = field(default_factory=dict)

    def repo(self, name: str) -> RepoCfg:
        for r in self.repos:
            if r.name == name:
                return r
        raise KeyError(f"repo {name!r} not in config; have {[r.name for r in self.repos]}")

    def data_dir(self, repo_name: str) -> Path:
        d = ROOT / "data" / repo_name
        d.mkdir(parents=True, exist_ok=True)
        return d


def _norm_reasoning(v: Any) -> str:
    """YAML parses bare `off`/`no`/`false` as a bool — normalize to 'off'/'on'."""
    if isinstance(v, bool):
        return "off" if v is False else "on"
    s = str(v).strip().lower()
    return "off" if s in {"off", "no", "false", "none", "0", "minimal"} else s


def _discover_repos() -> list[RepoCfg]:
    """Any git repo under repos/ becomes a target (used when config lists none)."""
    out: list[RepoCfg] = []
    repos_dir = ROOT / "repos"
    if not repos_dir.exists():
        return out
    for child in sorted(repos_dir.iterdir()):
        real = child.resolve()
        if (real / ".git").exists():
            out.append(RepoCfg(name=child.name, path=real, github=None))
    return out


def load(config_path: str | Path | None = None) -> Config:
    load_dotenv(ROOT / ".env")
    path = Path(config_path) if config_path else ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text()) or {}

    repos: list[RepoCfg] = []
    for r in raw.get("repos") or []:
        p = Path(r["path"])
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        repos.append(RepoCfg(name=r["name"], path=p, github=r.get("github")))
    if not repos:
        repos = _discover_repos()

    models = [
        ModelCfg(
            key=m["key"],
            provider=m["provider"],
            model=m["model"],
            price_in=float(m["price_in"]),
            price_out=float(m["price_out"]),
            reasoning=_norm_reasoning(m.get("reasoning", "off")),
            max_concurrency=int(m.get("max_concurrency", 40)),
        )
        for m in raw.get("models") or []
    ]

    return Config(
        raw=raw,
        repos=repos,
        models=models,
        extract=raw.get("extract", {}),
        github=raw.get("github", {}),
        judge=raw.get("judge", {}),
        analyze=raw.get("analyze", {}),
    )


def env(name: str) -> str | None:
    v = os.environ.get(name)
    return v.strip() if v else None
