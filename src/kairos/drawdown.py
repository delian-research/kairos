"""Drawdowns: declines from the running peak of a growth path."""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd

from ._util import clean_series, require_n, span
from .conventions import DEFAULT_RETURN_METHOD, ReturnMethod, check_method
from .results import Estimate
from .returns import cumulative


def drawdown_series(
    r: pd.Series, method: ReturnMethod = DEFAULT_RETURN_METHOD
) -> pd.Series:
    """Drawdown at each date as a fraction of the running peak (0 or negative).

    The path starts at 1.0 before the first return, so a loss on day one
    counts as a drawdown.
    """
    check_method(method)
    x = clean_series(r, "r")
    wealth = cumulative(x, method)
    peak = wealth.cummax().clip(lower=1.0)
    dd = wealth / peak - 1.0
    dd.name = "drawdown"
    return dd


@overload
def max_drawdown(r: pd.Series, method: ReturnMethod = ...) -> Estimate: ...
@overload
def max_drawdown(
    r: pd.DataFrame, method: ReturnMethod = ...
) -> dict[str, Estimate]: ...
def max_drawdown(
    r: pd.Series | pd.DataFrame, method: ReturnMethod = DEFAULT_RETURN_METHOD
) -> Estimate | dict[str, Estimate]:
    """Largest peak-to-trough decline, as a negative fraction (e.g. -0.25).

    ``params`` hold ``peak`` (``None`` if the peak is the starting value before
    the first return), ``trough`` and ``recovery`` (the first date the path
    gets back to the peak, or ``None`` if it has not).
    """
    if isinstance(r, pd.DataFrame):
        return {str(c): max_drawdown(r[c], method) for c in r.columns}
    x = clean_series(r, "r")
    require_n(len(x), 1, "max_drawdown")
    wealth = cumulative(x, method)
    dd = drawdown_series(x, method)
    # Work positionally: label-based .loc is ambiguous with duplicate index
    # labels and does not type-check against an idxmin() Hashable.
    pos = int(np.argmin(dd.to_numpy()))
    trough = dd.index[pos]
    value = float(dd.iloc[pos])

    peak = None
    recovery = None
    if value < 0:
        before = wealth.iloc[: pos + 1]
        if before.max() >= 1.0:
            peak = before.index[int(np.argmax(before.to_numpy()))]
            peak_level = float(before.max())
        else:
            peak_level = 1.0
        after = wealth.iloc[pos:]
        recovered = after[after >= peak_level]
        if len(recovered):
            recovery = recovered.index[0]

    start, end = span(x.index)
    return Estimate(
        value=value,
        n=len(x),
        start=start,
        end=end,
        method="max-drawdown",
        params={
            "return_method": method,
            "peak": peak,
            "trough": trough if value < 0 else None,
            "recovery": recovery,
        },
    )
