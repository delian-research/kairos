"""One-call statistics table."""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd

from ._util import align, clean_returns, frame_attrs, require_n, rf_label, span
from .beta import beta
from .conventions import (
    DEFAULT_RETURN_METHOD,
    TRADING_DAYS,
    ReturnMethod,
    annualize_vol,
    check_method,
)
from .drawdown import max_drawdown
from .performance import (
    annualized_return,
    calmar_ratio,
    capture_ratio,
    hit_rate,
    information_ratio,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)
from .results import Estimate
from .tail import excess_kurtosis, expected_shortfall, skewness, value_at_risk
from .transforms import excess_returns

SUMMARY_MIN_OBS = 20


def _value(fn: Callable[[], Estimate]) -> float:
    """Value of an estimate, or NaN where the statistic is undefined
    (zero volatility, no drawdown, no down periods, ...)."""
    try:
        return fn().value
    except ValueError:
        return math.nan


def _one(
    r: pd.Series,
    benchmark: pd.Series | None,
    rf: float | pd.Series,
    method: ReturnMethod,
    periods: int,
    level: float,
) -> dict[str, float]:
    x = clean_returns(r, "r")
    require_n(len(x), SUMMARY_MIN_OBS, "summary")
    pct = f"{level * 100:g}"
    row: dict[str, float] = {
        "n": float(len(x)),
        "ann_return": annualized_return(x, method, periods=periods).value,
        "ann_vol": annualize_vol(float(x.std(ddof=1)), periods),
        "sharpe": _value(lambda: sharpe_ratio(x, rf, annualize=True, periods=periods)),
        "sortino": _value(
            lambda: sortino_ratio(
                excess_returns(x, rf), annualize=True, periods=periods
            )
        ),
        "max_drawdown": max_drawdown(x, method).value,
        "calmar": _value(lambda: calmar_ratio(x, method, periods=periods)),
        "skew": skewness(x).value,
        "excess_kurtosis": excess_kurtosis(x).value,
        f"var_{pct}": value_at_risk(x, level).value,
        f"es_{pct}": expected_shortfall(x, level).value,
        "hit_rate": hit_rate(x).value,
        "best": float(x.max()),
        "worst": float(x.min()),
    }
    if benchmark is not None:
        a, b = align(x, benchmark)
        require_n(len(a), SUMMARY_MIN_OBS, "summary (aligned with benchmark)")
        bt = beta(a, b)
        row.update(
            {
                "beta": bt.value,
                "alpha_ann": bt.params["alpha"] * periods,
                "r2": bt.params["r2"],
                "correlation": float(a.corr(b)),
                "tracking_error": tracking_error(
                    a, b, annualize=True, periods=periods
                ).value,
                "information_ratio": _value(
                    lambda: information_ratio(a, b, annualize=True, periods=periods)
                ),
                "up_capture": _value(lambda: capture_ratio(a, b, "up", method)),
                "down_capture": _value(lambda: capture_ratio(a, b, "down", method)),
            }
        )
    return row


def summary(
    r: pd.Series | pd.DataFrame,
    benchmark: pd.Series | None = None,
    *,
    rf: float | pd.Series = 0.0,
    method: ReturnMethod = DEFAULT_RETURN_METHOD,
    periods: int = TRADING_DAYS,
    level: float = 0.95,
) -> pd.DataFrame:
    """Standard statistics table: one column per asset, one row per metric.

    ``r`` holds per-period returns of kind ``method``. Every column is
    evaluated on its own non-missing data. Rows:

    * ``n``, ``ann_return`` (CAGR), ``ann_vol``
    * ``sharpe`` and ``sortino`` (both on returns in excess of the
      per-period ``rf``), ``max_drawdown``, ``calmar``; ratios are **annualized**
    * ``skew``, ``excess_kurtosis``, historical ``var_95`` and ``es_95``
      (positive losses, per period), ``hit_rate``, ``best``, ``worst``
    * with ``benchmark``: ``beta``, ``alpha_ann`` (OLS intercept on raw
      returns times ``periods``), ``r2``, ``correlation``,
      ``tracking_error`` (annualized), ``information_ratio`` (annualized),
      ``up_capture``, ``down_capture``

    A metric is NaN only where it is undefined (zero volatility, no
    drawdown, no up or down benchmark periods). Fewer than 20 observations
    raise. Settings and each column's date span are in
    ``df.attrs["kairos"]``.
    """
    check_method(method)
    cols = r if isinstance(r, pd.DataFrame) else r.to_frame(r.name or "r")
    data = {}
    spans = {}
    for c in cols.columns:
        s = cols[c]
        data[str(c)] = _one(s, benchmark, rf, method, periods, level)
        spans[str(c)] = span(s.dropna().index)
    out = pd.DataFrame(data)
    return frame_attrs(
        out,
        method="summary",
        return_method=method,
        rf=rf_label(rf),
        periods=periods,
        level=level,
        annualized_ratios=True,
        spans=spans,
        benchmark=None if benchmark is None else str(benchmark.name),
    )
