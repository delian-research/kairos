"""Turning prices into returns, and returns back into growth paths."""

from __future__ import annotations

from typing import TypeVar, cast

import numpy as np
import pandas as pd

from .conventions import DEFAULT_RETURN_METHOD, ReturnMethod, check_method

PandasT = TypeVar("PandasT", pd.Series, pd.DataFrame)


def returns(prices: PandasT, method: ReturnMethod = DEFAULT_RETURN_METHOD) -> PandasT:
    """Period-over-period returns from a price Series or wide DataFrame.

    Args:
        prices: Prices indexed by date. A DataFrame is treated as one column
            per instrument.
        method: ``"log"`` for ``ln(p_t / p_{t-1})`` or ``"simple"`` for
            ``p_t / p_{t-1} - 1``.

    Returns:
        Same shape as ``prices`` minus the first row. ``n`` prices give
        ``n - 1`` returns.
    """
    check_method(method)
    p = prices.astype(float)
    if (p <= 0).to_numpy().any():
        raise ValueError("prices must be strictly positive")
    ratio = p / p.shift(1)
    r = cast(PandasT, np.log(ratio)) if method == "log" else ratio - 1.0
    return r.iloc[1:]


def last_n(x: PandasT, n: int) -> PandasT:
    """The latest ``n`` rows that have no missing values.

    Raises ``ValueError`` if fewer than ``n`` are available, instead of
    silently returning a shorter window. Remember that ``n`` returns need
    ``n + 1`` prices.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    clean = x.dropna()
    if len(clean) < n:
        raise ValueError(
            f"requested the last {n} observations but only {len(clean)} are "
            f"available (n returns need n + 1 prices)"
        )
    return clean.iloc[-n:]


def cumulative(r: PandasT, method: ReturnMethod = DEFAULT_RETURN_METHOD) -> PandasT:
    """Growth of 1 unit invested, from a return series.

    Returns the wealth index (starting just above or below 1.0 after the first
    period), not the cumulative return. Subtract 1 for that.
    """
    check_method(method)
    if method == "log":
        return cast(PandasT, np.exp(r.cumsum()))
    return (1.0 + r).cumprod()
