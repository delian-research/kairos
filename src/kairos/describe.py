"""Tools for programmatic and LLM callers: discover the API, read the usage
guide, and turn any kairos result into text or JSON-safe data."""

from __future__ import annotations

import datetime as dt
import inspect
import math
from collections.abc import Mapping
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

from .results import Estimate


def catalog() -> pd.DataFrame:
    """Table of every public kairos function.

    Columns: ``area`` (module), ``signature`` and ``summary`` (the first
    paragraph of the docstring), indexed by function name. Use
    ``help(kairos.<name>)`` for the full documentation of one function and
    :func:`guide` for conventions and recipes.
    """
    import kairos

    rows = []
    for name in kairos.__all__:
        obj = getattr(kairos, name)
        if not inspect.isfunction(obj):
            continue
        doc = inspect.getdoc(obj) or ""
        first = " ".join(doc.split("\n\n")[0].split())
        rows.append(
            {
                "function": name,
                "area": obj.__module__.rsplit(".", 1)[-1],
                "signature": f"{name}{inspect.signature(obj)}",
                "summary": first,
            }
        )
    out = pd.DataFrame(rows).set_index("function").sort_values(["area", "function"])
    return out


def guide() -> str:
    """The kairos usage guide as text: conventions, which function answers
    which question, recipes for securities and portfolios, and common
    mistakes. Written for LLM callers; ``print(kairos.guide())``."""
    return resources.files("kairos").joinpath("GUIDE.md").read_text(encoding="utf-8")


def _label(v: Any) -> str:
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()[:10] if _is_midnight(v) else v.isoformat()
    return str(v)


def _is_midnight(v: Any) -> bool:
    if isinstance(v, dt.datetime):
        return v.time() == dt.time(0) and v.tzinfo is None
    return True


def _plain(v: Any) -> Any:
    """Recursively convert to JSON-safe builtins. NaN and infinity become
    ``None``; dates become ISO strings."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, np.generic):
        return _plain(v.item())
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if v is pd.NaT:
        return None
    if isinstance(v, (dt.datetime, dt.date)):
        return _label(v)
    if isinstance(v, Estimate):
        return _plain(v.as_dict())
    if isinstance(v, pd.DataFrame):
        return {
            _label(c): {_label(i): _plain(x) for i, x in v[c].items()}
            for c in v.columns
        }
    if isinstance(v, pd.Series):
        return {_label(i): _plain(x) for i, x in v.items()}
    if isinstance(v, Mapping):
        return {_label(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, np.ndarray, pd.Index)):
        return [_plain(x) for x in v]
    return str(v)


def to_dict(result: Any) -> Any:
    """Convert any kairos result to JSON-safe builtins.

    * :class:`Estimate` -> its flat dict (``value``, ``n``, ``start``,
      ``end``, ``method`` and settings).
    * ``dict[str, Estimate]`` -> one such dict per key.
    * DataFrame or Series -> ``{"data": ..., "meta": ...}``, where ``data`` is
      keyed by column then row (or by row for a Series) and ``meta`` is the
      ``attrs["kairos"]`` metadata.

    Dates become ISO strings and NaN becomes ``None``, so the result can go
    straight to ``json.dumps``.
    """
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return {"data": _plain(result), "meta": _plain(result.attrs.get("kairos"))}
    return _plain(result)


def _meta_line(meta: Mapping[str, Any]) -> str:
    parts = []
    for k, v in meta.items():
        if isinstance(v, (pd.DataFrame, pd.Series)):
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.6g}")
        elif isinstance(v, (Mapping, list, tuple)):
            parts.append(f"{k}={_plain(v)}")
        else:
            parts.append(f"{k}={_label(v) if v is not None else None}")
    return "[kairos] " + ", ".join(parts)


def render(result: Any) -> str:
    """Format any kairos result as plain text that states how it was
    computed, ready to paste into a report or an LLM answer.

    An :class:`Estimate` renders as its one-line description; a dict of
    estimates as one line per key; a DataFrame or Series as a table followed
    by a ``[kairos]`` line with its metadata.
    """
    if isinstance(result, Estimate):
        return str(result)
    if isinstance(result, Mapping) and all(
        isinstance(v, Estimate) for v in result.values()
    ):
        return "\n".join(f"{k}: {v}" for k, v in result.items())
    if isinstance(result, (pd.DataFrame, pd.Series)):
        body = result.to_string(float_format=lambda x: f"{x:.6g}")
        meta = result.attrs.get("kairos")
        return f"{body}\n{_meta_line(meta)}" if meta else body
    return str(result)
