"""Volatility estimators.

All estimators return *per-period* volatility unless ``annualize=True``, in
which case they scale by ``sqrt(periods)``. The settings used are recorded in
the returned :class:`~kairos.results.Estimate`.
"""

from __future__ import annotations

import math
from typing import cast, overload

import numpy as np
import pandas as pd

from ._util import align, clean_series, frame_attrs, require_n, span
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


def _ohlc(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.DataFrame:
    df = pd.concat(
        [x.astype(float) for x in (open_, high, low, close)],
        axis=1,
        join="inner",
        keys=["open", "high", "low", "close"],
    ).dropna()
    if (df <= 0).to_numpy().any() or (df["high"] < df["low"]).any():
        raise ValueError("need positive prices and low <= high on every row")
    return df


def _range_estimate(
    var: float,
    df: pd.DataFrame,
    method: str,
    annualize: bool,
    periods: int,
    **extra: object,
) -> Estimate:
    sigma = math.sqrt(max(var, 0.0))
    if annualize:
        sigma = annualize_vol(sigma, periods)
    start, end = span(df.index)
    return Estimate(
        value=sigma,
        n=len(df),
        start=start,
        end=end,
        method=method,
        params={"annualized": annualize, "periods": periods, **extra},
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
    df = _ohlc(open_, high, low, close)
    require_n(len(df), 1, "garman_klass_vol")
    o, h, lo, c = (df[k] for k in ("open", "high", "low", "close"))
    term = 0.5 * np.log(h / lo) ** 2 - (2.0 * math.log(2.0) - 1.0) * np.log(c / o) ** 2
    return _range_estimate(float(term.mean()), df, "garman-klass", annualize, periods)


def _rs_terms(df: pd.DataFrame) -> pd.Series:
    o, h, lo, c = (df[k] for k in ("open", "high", "low", "close"))
    return np.log(h / c) * np.log(h / o) + np.log(lo / c) * np.log(lo / o)


def rogers_satchell_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> Estimate:
    """Rogers-Satchell (1991) OHLC estimator.

    ``sigma^2 = mean(ln(H/C) ln(H/O) + ln(L/C) ln(L/O))``. Unlike Parkinson and
    Garman-Klass it stays unbiased when prices drift, but it still ignores
    overnight gaps.
    """
    df = _ohlc(open_, high, low, close)
    require_n(len(df), 1, "rogers_satchell_vol")
    var = float(_rs_terms(df).mean())
    return _range_estimate(var, df, "rogers-satchell", annualize, periods)


def yang_zhang_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> Estimate:
    """Yang-Zhang (2000) estimator, which includes overnight gaps.

    ``sigma^2 = var(overnight) + k var(open-to-close) + (1 - k) RS`` with
    ``k = 0.34 / (1.34 + (n + 1) / (n - 1))``, where overnight is
    ``ln(O_t / C_{t-1})`` and RS is the Rogers-Satchell variance. The first
    row only supplies a previous close, so ``n`` is one less than the number
    of rows.
    """
    df = _ohlc(open_, high, low, close)
    overnight = cast(pd.Series, np.log(df["open"] / df["close"].shift(1))).iloc[1:]
    body = df.iloc[1:]
    n = len(body)
    require_n(n, 2, "yang_zhang_vol")
    intraday = np.log(body["close"] / body["open"])
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = (
        float(overnight.var(ddof=1))
        + k * float(intraday.var(ddof=1))
        + (1.0 - k) * float(_rs_terms(body).mean())
    )
    return _range_estimate(var, body, "yang-zhang", annualize, periods, k=k)


def vol_cone(
    r: pd.Series,
    windows: tuple[int, ...] = (21, 63, 126, 252),
    *,
    quantiles: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    ddof: int = DEFAULT_DDOF,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Distribution of rolling volatility for several window lengths.

    One row per window; columns are the requested quantiles of that window's
    rolling volatility history plus ``current`` (the latest value). Rolling
    windows overlap, so the quantiles are descriptive, not independent
    samples. Metadata is in ``df.attrs["kairos"]``.
    """
    x = clean_series(r, "r")
    rows = {}
    for w in windows:
        rv = rolling_vol(x, w, ddof=ddof, annualize=annualize, periods=periods)
        rv = rv.dropna()
        require_n(len(rv), 1, f"vol_cone window {w}")
        row = {f"q{q:g}": float(rv.quantile(q)) for q in quantiles}
        row["current"] = float(rv.iloc[-1])
        rows[w] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "window"
    start, end = span(x.index)
    return frame_attrs(
        out,
        method="vol-cone",
        n=len(x),
        start=start,
        end=end,
        ddof=ddof,
        annualized=annualize,
        periods=periods,
    )


def vol_percentile(r: pd.Series, window: int, *, ddof: int = DEFAULT_DDOF) -> Estimate:
    """Where the latest ``window``-period volatility sits in its own history.

    The value is the fraction of rolling-volatility observations (including
    the latest) that are less than or equal to the latest, so 1.0 means
    volatility is at its highest in the sample. ``n`` counts rolling values.
    """
    x = clean_series(r, "r")
    rv = rolling_vol(x, window, ddof=ddof).dropna()
    require_n(len(rv), 2, "vol_percentile")
    current = float(rv.iloc[-1])
    start, end = span(rv.index)
    return Estimate(
        value=float((rv <= current).mean()),
        n=len(rv),
        start=start,
        end=end,
        method="vol-percentile",
        params={"window": window, "ddof": ddof, "current_vol": current},
    )
