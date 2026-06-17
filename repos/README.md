# `repos/` — analysis targets

Drop (or symlink) a git checkout of each repository you want to analyze in here,
one subfolder per repo, then add a matching entry to `config.yaml`.

```bash
# symlink an existing local clone (recommended — no extra disk)
ln -s /path/to/codex repos/codex

# or clone fresh
git clone https://github.com/openai/codex repos/codex
```

The tool reads the **full local git history** for diffs (free, fast) and uses the
GitHub GraphQL API only to fetch each PR's branch commits (authors + dates).

Repo contents here are gitignored — only this README is tracked.
