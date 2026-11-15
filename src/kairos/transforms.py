"""Return transforms: excess and active returns, frequency changes, and
dividend-adjusted (total return) prices."""

from __future__ import annotations

from typing import TypeVar, cast

import numpy as np
import pandas as pd

from ._util import align, clean_series
from .conventions import DEFAULT_RETURN_METHOD, ReturnMethod, check_method

PandasT = TypeVar("PandasT", pd.Series, pd.DataFrame)


def excess_returns(r: PandasT, rf: float | pd.Series) -> PandasT:
    """Returns minus a per-period risk-free rate.

    ``rf`` is either a constant per-period rate (e.g. ``0.04 / 252`` for 4% a
    year on daily data) or a Series of per-period rates. A Series is
    inner-joined to ``r`` on the index; dates missing from either side are
    dropped.
    """
    if not isinstance(rf, pd.Series):
        return cast(PandasT, r.astype(float) - float(rf))
    if isinstance(r, pd.Series):
        a, b = align(r, rf)
        return cast(PandasT, a - b)
    idx = r.index.intersection(rf.index)
    return cast(
        PandasT, r.loc[idx].astype(float).sub(rf.loc[idx].astype(float), axis=0)
    )


def active_returns(r: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Asset return minus benchmark return, on the dates both have."""
    a, b = align(r, benchmark)
    out = a - b
    out.name = "active"
    return out


def resample_returns(
    r: PandasT, freq: str, method: ReturnMethod = DEFAULT_RETURN_METHOD
) -> PandasT:
    """Aggregate returns to a lower frequency (e.g. ``"W-FRI"``, ``"ME"``).

    Log returns are summed; simple returns are compounded,
    ``prod(1 + r) - 1``. Summing simple returns is a common mistake and is
    never done here. Bins with no observations are dropped rather than
    reported as a zero return. Requires a DatetimeIndex.
    """
    check_method(method)
    if not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("resample_returns needs a DatetimeIndex")
    x = r.astype(float)
    logs = x if method == "log" else cast(PandasT, np.log1p(x))
    summed = logs.resample(freq).sum(min_count=1)
    out = summed if method == "log" else cast(PandasT, np.expm1(summed))
    return cast(PandasT, out.dropna(how="all"))


def total_return_prices(close: pd.Series, dividends: pd.Series) -> pd.Series:
    """Dividend-adjusted price series (backward adjusted).

    ``dividends`` holds cash amounts indexed by **ex-date**. On an ex-date
    the total return is ``(close_t + D_t) / close_{t-1} - 1``. The result is
    scaled so its last value equals the last close, the usual "adjusted
    close" convention, so returns computed from it are total returns.

    Every ex-date must be a date in ``close``'s index; a dividend on the first
    date cannot be applied (there is no prior close) and raises.
    """
    c = clean_series(close, "close")
    if (c <= 0).any():
        raise ValueError("close must be strictly positive")
    d = clean_series(dividends, "dividends")
    d = d[d != 0]
    missing = d.index.difference(c.index)
    if len(missing):
        raise ValueError(
            f"{len(missing)} ex-date(s) are not in the close index, first: {missing[0]}"
        )
    if len(c) and len(d) and (d.index == c.index[0]).any():
        raise ValueError("a dividend on the first date has no prior close")
    d = d.groupby(level=0).sum().reindex(c.index, fill_value=0.0)
    gross = (c + d) / c.shift(1)
    growth = gross.iloc[1:].cumprod()
    tr = pd.concat([pd.Series([1.0], index=c.index[:1]), growth])
    out = tr * (float(c.iloc[-1]) / float(tr.iloc[-1]))
    out.name = close.name
    return out
