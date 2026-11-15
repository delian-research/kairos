"""Tail risk: value at risk, expected shortfall and higher moments.

VaR and expected shortfall are reported as **positive losses**: a 95% VaR of
0.021 means a 2.1% loss is exceeded on about 5% of periods.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ._util import check_level, clean_returns, require_n, span
from .results import Estimate

VaRMethod = Literal["historical", "normal", "cornish-fisher"]
ESMethod = Literal["historical", "normal"]

_N = NormalDist()


def _moment(v: Any) -> float:
    """pandas-stubs type Series.skew()/kurt() broadly; they return a float."""
    return float(cast(float, v))


def _est(value: float, x: pd.Series, method: str, **params: object) -> Estimate:
    start, end = span(x.index)
    return Estimate(
        value=float(value),
        n=len(x),
        start=start,
        end=end,
        method=method,
        params=dict(params),
    )


def value_at_risk(
    r: pd.Series, level: float = 0.95, method: VaRMethod = "historical"
) -> Estimate:
    """One-period value at risk at confidence ``level``, as a positive loss.

    * ``"historical"``: the empirical ``1 - level`` quantile (linear
      interpolation).
    * ``"normal"``: ``-(mean + z * std)`` with ``z`` the normal quantile.
    * ``"cornish-fisher"``: the normal quantile adjusted for sample skew and
      excess kurtosis (Favre and Galeano, 2002).
    """
    level = check_level(level)
    x = clean_returns(r, "r")
    alpha = 1.0 - level
    if method == "historical":
        require_n(len(x), 2, "value_at_risk")
        q = float(np.quantile(x.to_numpy(), alpha))
    elif method in ("normal", "cornish-fisher"):
        require_n(len(x), 4 if method == "cornish-fisher" else 2, "value_at_risk")
        z = _N.inv_cdf(alpha)
        if method == "cornish-fisher":
            s, k = _moment(x.skew()), _moment(x.kurt())
            z = (
                z
                + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * k / 24
                - (2 * z**3 - 5 * z) * s**2 / 36
            )
        q = float(x.mean()) + z * float(x.std(ddof=1))
    else:
        raise ValueError(f"unknown VaR method {method!r}")
    return _est(-q, x, f"var-{method}", level=level)


def expected_shortfall(
    r: pd.Series, level: float = 0.95, method: ESMethod = "historical"
) -> Estimate:
    """Expected shortfall (CVaR): the mean loss beyond VaR, as a positive loss.

    * ``"historical"``: minus the mean of returns at or below the empirical
      ``1 - level`` quantile.
    * ``"normal"``: ``-(mean - std * pdf(z) / (1 - level))``.
    """
    level = check_level(level)
    x = clean_returns(r, "r")
    require_n(len(x), 2, "expected_shortfall")
    alpha = 1.0 - level
    if method == "historical":
        q = float(np.quantile(x.to_numpy(), alpha))
        tail = x[x <= q]
        value = -float(tail.mean())
        extra: dict[str, object] = {"tail_n": len(tail)}
    elif method == "normal":
        z = _N.inv_cdf(alpha)
        value = -(float(x.mean()) - float(x.std(ddof=1)) * _N.pdf(z) / alpha)
        extra = {}
    else:
        raise ValueError(f"unknown expected shortfall method {method!r}")
    return _est(value, x, f"es-{method}", level=level, **extra)


def skewness(r: pd.Series) -> Estimate:
    """Sample skewness (adjusted Fisher-Pearson, bias corrected)."""
    x = clean_returns(r, "r")
    require_n(len(x), 3, "skewness")
    return _est(_moment(x.skew()), x, "skewness", bias_corrected=True)


def excess_kurtosis(r: pd.Series) -> Estimate:
    """Sample excess kurtosis (0 for a normal distribution, bias corrected)."""
    x = clean_returns(r, "r")
    require_n(len(x), 4, "excess_kurtosis")
    return _est(_moment(x.kurt()), x, "excess-kurtosis", bias_corrected=True)


def worst_periods(r: pd.Series, n: int = 5) -> pd.Series:
    """The ``n`` lowest returns, worst first, with their dates."""
    x = clean_returns(r, "r")
    require_n(len(x), n, "worst_periods")
    return x.nsmallest(n)
