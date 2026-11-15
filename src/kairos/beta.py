"""Beta of an asset's returns against a benchmark's returns."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._util import align, require_n, span
from .results import Estimate


def newey_west_lags(n: int) -> int:
    """Rule-of-thumb HAC lag length: ``floor(4 * (n / 100) ** (2 / 9))``."""
    return max(0, int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def beta(
    asset: pd.Series,
    benchmark: pd.Series,
    *,
    hac: bool = False,
    lags: int | None = None,
) -> Estimate:
    """OLS beta from ``asset = alpha + beta * benchmark + e``.

    Both inputs must be *returns* over the same periods. They are inner-joined
    on their index and rows with any NaN are dropped.

    Args:
        hac: Report a Newey-West (Bartlett kernel) standard error, which is
            robust to heteroskedasticity and autocorrelation in the residuals.
            Otherwise the classical OLS standard error is used.
        lags: HAC lag length. ``None`` uses :func:`newey_west_lags`.

    The returned ``params`` hold ``alpha`` (per period), ``se``, ``t``, ``r2``
    and ``se_type``.
    """
    y, x = align(asset, benchmark)
    n = len(y)
    require_n(n, 3, "beta")
    xv = x.to_numpy()
    yv = y.to_numpy()
    xc = xv - xv.mean()
    yc = yv - yv.mean()
    sxx = float(np.dot(xc, xc))
    if sxx == 0.0:
        raise ValueError("benchmark returns have zero variance; beta is undefined")
    b = float(np.dot(xc, yc)) / sxx
    a = float(yv.mean() - b * xv.mean())
    u = yv - a - b * xv
    syy = float(np.dot(yc, yc))
    r2 = 1.0 - float(np.dot(u, u)) / syy if syy > 0 else float("nan")

    params: dict[str, object] = {"alpha": a}
    if hac:
        L = newey_west_lags(n) if lags is None else int(lags)
        if L < 0:
            raise ValueError(f"lags must be >= 0, got {L}")
        g = xc * u
        s = float(np.dot(g, g))
        for lag in range(1, min(L, n - 1) + 1):
            w = 1.0 - lag / (L + 1.0)
            s += 2.0 * w * float(np.dot(g[lag:], g[:-lag]))
        se = math.sqrt(s) / sxx
        params.update({"se_type": "newey-west", "lags": L})
    else:
        s2 = float(np.dot(u, u)) / (n - 2)
        se = math.sqrt(s2 / sxx)
        params["se_type"] = "ols"
    params.update({"se": se, "t": b / se if se > 0 else float("nan"), "r2": r2})

    start, end = span(y.index)
    return Estimate(value=b, n=n, start=start, end=end, method="ols", params=params)


def rolling_beta(asset: pd.Series, benchmark: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta over ``window`` aligned observations.

    Labelled by each window's last date. The first ``window - 1`` values are
    NaN, and so is any window where the benchmark has zero variance.
    """
    require_n(window, 3, "rolling_beta window")
    y, x = align(asset, benchmark)
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    out = cov / var.replace(0.0, np.nan)
    out.name = "beta"
    return out
