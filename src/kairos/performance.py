"""Risk-adjusted performance ratios.

Inputs are per-period returns. Ratios are per-period unless
``annualize=True``; mean/volatility ratios then scale by ``sqrt(periods)``.
``annualized_return`` (CAGR) is annual by definition.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from ._util import align, clean_series, is_zero_scale, require_n, rf_label, span
from .conventions import (
    DEFAULT_DDOF,
    DEFAULT_RETURN_METHOD,
    TRADING_DAYS,
    ReturnMethod,
    check_method,
)
from .drawdown import max_drawdown
from .results import Estimate
from .transforms import active_returns, excess_returns


def _log_growth(x: pd.Series, method: ReturnMethod) -> pd.Series:
    """Per-period log growth, whatever the input return definition."""
    if method == "log":
        return x
    if (x <= -1.0).any():
        raise ValueError("simple returns must be greater than -1")
    return np.log1p(x)


def _estimate(value: float, x: pd.Series, method: str, **params: object) -> Estimate:
    start, end = span(x.index)
    return Estimate(
        value=float(value),
        n=len(x),
        start=start,
        end=end,
        method=method,
        params=dict(params),
    )


def annualized_return(
    r: pd.Series,
    method: ReturnMethod = DEFAULT_RETURN_METHOD,
    *,
    periods: int = TRADING_DAYS,
) -> Estimate:
    """Compound annual growth rate (CAGR), as a **simple** annual rate.

    ``exp(sum(log growth) * periods / n) - 1``. ``method`` says what kind of
    returns ``r`` holds; the result is a simple rate either way (0.10 = 10%
    a year).
    """
    check_method(method)
    x = clean_series(r, "r")
    require_n(len(x), 1, "annualized_return")
    g = _log_growth(x, method)
    value = math.expm1(float(g.sum()) * periods / len(x))
    return _estimate(value, x, "cagr", return_method=method, periods=periods)


def sharpe_ratio(
    r: pd.Series,
    rf: float | pd.Series = 0.0,
    *,
    ddof: int = DEFAULT_DDOF,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> Estimate:
    """Mean excess return over its standard deviation.

    ``rf`` is a per-period rate (constant or Series). Raises if the excess
    returns have zero volatility.
    """
    x = excess_returns(clean_series(r, "r"), rf)
    require_n(len(x), ddof + 1, "sharpe_ratio")
    sd = float(x.std(ddof=ddof))
    if is_zero_scale(sd, float(x.mean())):
        raise ValueError("sharpe_ratio is undefined for zero volatility")
    value = float(x.mean()) / sd
    if annualize:
        value *= math.sqrt(periods)
    return _estimate(
        value,
        x,
        "sharpe",
        rf=rf_label(rf),
        ddof=ddof,
        annualized=annualize,
        periods=periods,
    )


def sortino_ratio(
    r: pd.Series,
    target: float = 0.0,
    *,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> Estimate:
    """Mean return above ``target`` over downside deviation.

    Downside deviation is ``sqrt(mean(min(r - target, 0)^2))`` taken over
    *all* periods (Sortino and Price, 1994), not only the losing ones.
    """
    x = clean_series(r, "r")
    require_n(len(x), 2, "sortino_ratio")
    d = x - target
    dd = math.sqrt(float((d.clip(upper=0.0) ** 2).mean()))
    if dd == 0:
        raise ValueError("sortino_ratio is undefined with no returns below target")
    value = float(d.mean()) / dd
    if annualize:
        value *= math.sqrt(periods)
    return _estimate(
        value,
        x,
        "sortino",
        target=target,
        downside_deviation=dd,
        annualized=annualize,
        periods=periods,
    )


def calmar_ratio(
    r: pd.Series,
    method: ReturnMethod = DEFAULT_RETURN_METHOD,
    *,
    periods: int = TRADING_DAYS,
) -> Estimate:
    """CAGR divided by the absolute maximum drawdown over the same data."""
    cagr = annualized_return(r, method, periods=periods)
    mdd = max_drawdown(clean_series(r, "r"), method)
    if mdd.value == 0:
        raise ValueError("calmar_ratio is undefined with no drawdown")
    x = clean_series(r, "r")
    return _estimate(
        cagr.value / abs(mdd.value),
        x,
        "calmar",
        return_method=method,
        periods=periods,
        cagr=cagr.value,
        max_drawdown=mdd.value,
    )


def tracking_error(
    r: pd.Series,
    benchmark: pd.Series,
    *,
    ddof: int = DEFAULT_DDOF,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> Estimate:
    """Standard deviation of active returns (asset minus benchmark)."""
    a = active_returns(r, benchmark)
    require_n(len(a), ddof + 1, "tracking_error")
    value = float(a.std(ddof=ddof))
    if annualize:
        value *= math.sqrt(periods)
    return _estimate(
        value,
        a,
        "tracking-error",
        ddof=ddof,
        annualized=annualize,
        periods=periods,
    )


def information_ratio(
    r: pd.Series,
    benchmark: pd.Series,
    *,
    ddof: int = DEFAULT_DDOF,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> Estimate:
    """Mean active return over tracking error."""
    a = active_returns(r, benchmark)
    require_n(len(a), ddof + 1, "information_ratio")
    te = float(a.std(ddof=ddof))
    if is_zero_scale(te, float(a.mean())):
        raise ValueError("information_ratio is undefined for zero tracking error")
    value = float(a.mean()) / te
    if annualize:
        value *= math.sqrt(periods)
    return _estimate(
        value,
        a,
        "information-ratio",
        ddof=ddof,
        annualized=annualize,
        periods=periods,
    )


def capture_ratio(
    r: pd.Series,
    benchmark: pd.Series,
    side: Literal["up", "down"] = "up",
    method: ReturnMethod = DEFAULT_RETURN_METHOD,
) -> Estimate:
    """Up- or down-market capture ratio.

    Uses the periods where the benchmark rose (``side="up"``) or fell
    (``"down"``), and divides the asset's geometric mean return over those
    periods by the benchmark's. An up capture of 1.2 means the asset gained
    20% more than the benchmark in up periods; a down capture below 1 means
    it lost less in down periods.
    """
    check_method(method)
    if side not in ("up", "down"):
        raise ValueError(f"side must be 'up' or 'down', got {side!r}")
    a, b = align(r, benchmark)
    mask = b > 0 if side == "up" else b < 0
    a, b = a[mask], b[mask]
    require_n(len(a), 1, f"capture_ratio ({side} periods)")
    ga = math.expm1(float(_log_growth(a, method).mean()))
    gb = math.expm1(float(_log_growth(b, method).mean()))
    return _estimate(
        ga / gb,
        a,
        f"{side}-capture",
        return_method=method,
        asset_mean=ga,
        benchmark_mean=gb,
    )


def hit_rate(r: pd.Series, threshold: float = 0.0) -> Estimate:
    """Fraction of periods with a return strictly above ``threshold``."""
    x = clean_series(r, "r")
    require_n(len(x), 1, "hit_rate")
    return _estimate(float((x > threshold).mean()), x, "hit-rate", threshold=threshold)
