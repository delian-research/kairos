"""Self-describing result type returned by every scalar kairos function."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Estimate:
    """A computed statistic plus everything needed to say how it was computed.

    Attributes:
        value: The statistic itself.
        n: Number of observations actually used, after alignment and NaN drops.
        start: First index label used, or ``None`` if the index is not dated.
        end: Last index label used.
        method: Short name of the estimator, e.g. ``"close-to-close"``.
        params: Estimator settings and secondary outputs (ddof, return
            method, annualization, standard errors, ...).
    """

    value: float
    n: int
    start: Any
    end: Any
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.value)

    def as_dict(self) -> dict[str, Any]:
        """Flat dict form, convenient for DataFrames and JSON."""
        out = asdict(self)
        params = out.pop("params")
        out.update(params)
        return out

    def __str__(self) -> str:
        span = f"{_fmt(self.start)} to {_fmt(self.end)}"
        extras = ", ".join(f"{k}={_fmt(v)}" for k, v in self.params.items())
        tail = f"; {extras}" if extras else ""
        return f"{self.value:.6g} ({self.method}, n={self.n}, {span}{tail})"


def _fmt(v: Any) -> str:
    if isinstance(v, (dt.datetime, dt.date)):
        # pandas Timestamps are datetime subclasses
        return v.strftime("%Y-%m-%d")
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)
