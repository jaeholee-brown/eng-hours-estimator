"""Robust extraction of the final hours estimate from a model's free text.

Strategy, in order of preference:
  1. The explicit `FINAL_ESTIMATE_HOURS: <number>` marker (last occurrence).
  2. A loose marker variant (e.g. "final estimate: 3 hours", "estimate = 3h").
  3. The last number that appears next to the word "hour(s)".
Returns (value, method). `value` is None if nothing parseable was found.
"""
from __future__ import annotations

import re

# 12, 12.5, .5, 1,200, optionally wrapped in ** or backticks
_NUM = r"[-+]?\$?\*?`?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+)"

_MARKER = re.compile(
    r"FINAL[_\s]*ESTIMATE[_\s]*HOURS\s*[:=]?\s*" + _NUM,
    re.IGNORECASE,
)
_LOOSE = re.compile(
    r"(?:final\s+)?(?:estimate|answer|total)\b[^\d\n]{0,40}?" + _NUM + r"\s*(?:hours?|hrs?|h)\b",
    re.IGNORECASE,
)
_NEAR_HOURS = re.compile(_NUM + r"\s*(?:hours?|hrs?|h)\b", re.IGNORECASE)


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").strip())
    except ValueError:
        return None


def extract_hours(text: str) -> tuple[float | None, str]:
    if not text:
        return None, "empty"

    matches = list(_MARKER.finditer(text))
    if matches:
        v = _to_float(matches[-1].group(1))
        if v is not None:
            return v, "marker"

    matches = list(_LOOSE.finditer(text))
    if matches:
        v = _to_float(matches[-1].group(1))
        if v is not None:
            return v, "loose"

    matches = list(_NEAR_HOURS.finditer(text))
    if matches:
        v = _to_float(matches[-1].group(1))
        if v is not None:
            return v, "near_hours"

    return None, "not_found"
