"""Volatility estimators.

All estimators return *per-period* volatility unless ``annualize=True``, in
which case they scale by ``sqrt(periods)``. The settings used are recorded in
the returned :class:`~kairos.results.Estimate`.
"""

from __future__ import annotations

import math
from typing import overload

import numpy as np
import pandas as pd

from ._util import align, clean_series, require_n, span
from .conventions import DEFAULT_DDOF, TRADING_DAYS, annualize_vol
from .results import Estimate
from .returns import last_n


@overload
def realized_vol(
    r: pd.Series,
    *,
    window: int | None = ...,
    ddof: int = ...,
    annualize: bool = ...,
    periods: int = ...,
) -> Estimate: ...
@overload
def realized_vol(
    r: pd.DataFrame,
    *,
    window: int | None = ...,
    ddof: int = ...,
    annualize: bool = ...,
    periods: int = ...,
) -> dict[str, Estimate]: ...
def realized_vol(
    r: pd.Series | pd.DataFrame,
    *,
    window: int | None = None,
    ddof: int = DEFAULT_DDOF,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> Estimate | dict[str, Estimate]:
    """Close-to-close realized volatility: the standard deviation of returns.

    Args:
        r: Returns (not prices). A DataFrame gives one estimate per column.
        window: Use only the latest ``window`` returns. Raises if fewer exist.
            ``None`` uses every non-missing return.
        ddof: Delta degrees of freedom (1 = sample standard deviation).
        annualize: Multiply by ``sqrt(periods)``.
        periods: Periods per year for annualization.
    """
    if isinstance(r, pd.DataFrame):
        return {
            str(c): realized_vol(
                r[c], window=window, ddof=ddof, annualize=annualize, periods=periods
            )
            for c in r.columns
        }
    x = clean_series(r, "r")
    if window is not None:
        x = last_n(x, window)
    require_n(len(x), ddof + 1, "realized_vol")
    sigma = float(x.std(ddof=ddof))
    if annualize:
        sigma = annualize_vol(sigma, periods)
    start, end = span(x.index)
    return Estimate(
        value=sigma,
        n=len(x),
        start=start,
        end=end,
        method="close-to-close",
        params={"ddof": ddof, "annualized": annualize, "periods": periods},
    )


def rolling_vol(
    r: pd.Series | pd.DataFrame,
    window: int,
    *,
    ddof: int = DEFAULT_DDOF,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> pd.Series | pd.DataFrame:
    """Rolling close-to-close volatility over ``window`` returns.

    The first ``window - 1`` values are NaN.
    """
    require_n(window, ddof + 1, "rolling_vol window")
    out = r.astype(float).rolling(window).std(ddof=ddof)
    return out * math.sqrt(periods) if annualize else out


def ewma_vol(
    r: pd.Series,
    halflife: float,
    *,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> Estimate:
    """Exponentially weighted volatility as of the last observation.

    Uses the RiskMetrics convention: zero mean, so the estimate is
    ``sqrt(EWMA(r**2))``. ``halflife`` is in periods. RiskMetrics' daily
    lambda of 0.94 corresponds to a halflife of about 11.2.
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be positive, got {halflife}")
    x = clean_series(r, "r")
    require_n(len(x), 2, "ewma_vol")
    var = float((x**2).ewm(halflife=halflife, adjust=True).mean().iloc[-1])
    sigma = math.sqrt(var)
    if annualize:
        sigma = annualize_vol(sigma, periods)
    start, end = span(x.index)
    return Estimate(
        value=sigma,
        n=len(x),
        start=start,
        end=end,
        method="ewma",
        params={
            "halflife": halflife,
            "demeaned": False,
            "annualized": annualize,
            "periods": periods,
        },
    )


def parkinson_vol(
    high: pd.Series,
    low: pd.Series,
    *,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> Estimate:
    """Parkinson (1980) high-low range estimator.

    ``sigma^2 = mean(ln(H/L)^2) / (4 ln 2)``. It is more efficient than
    close-to-close for a driftless diffusion but ignores overnight gaps, so it
    tends to understate the volatility of instruments that gap.
    """
    h, lo = align(high, low)
    require_n(len(h), 1, "parkinson_vol")
    if (lo <= 0).any() or (h < lo).any():
        raise ValueError("need 0 < low <= high on every row")
    hl = np.log(h / lo)
    sigma = math.sqrt(float((hl**2).mean()) / (4.0 * math.log(2.0)))
    if annualize:
        sigma = annualize_vol(sigma, periods)
    start, end = span(h.index)
    return Estimate(
        value=sigma,
        n=len(h),
        start=start,
        end=end,
        method="parkinson",
        params={"annualized": annualize, "periods": periods},
    )


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> Estimate:
    """Garman-Klass (1980) OHLC estimator.

    ``sigma^2 = mean(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2)``. Like
    Parkinson, it measures intraday variation only and excludes the
    close-to-open gap.
    """
    df = pd.concat(
        [
            open_.astype(float),
            high.astype(float),
            low.astype(float),
            close.astype(float),
        ],
        axis=1,
        join="inner",
    ).dropna()
    require_n(len(df), 1, "garman_klass_vol")
    o, h, lo, c = (df.iloc[:, i] for i in range(4))
    if (df <= 0).to_numpy().any() or (h < lo).any():
        raise ValueError("need positive prices and low <= high on every row")
    term = 0.5 * np.log(h / lo) ** 2 - (2.0 * math.log(2.0) - 1.0) * np.log(c / o) ** 2
    sigma = math.sqrt(max(float(term.mean()), 0.0))
    if annualize:
        sigma = annualize_vol(sigma, periods)
    start, end = span(df.index)
    return Estimate(
        value=sigma,
        n=len(df),
        start=start,
        end=end,
        method="garman-klass",
        params={"annualized": annualize, "periods": periods},
    )
