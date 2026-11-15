"""Default conventions used across kairos.

Every function in kairos takes these as explicit keyword arguments, so a
caller can always override them. They are collected here so there is exactly
one place that says what "the default" means.
"""

from __future__ import annotations

import math
from typing import Literal

ReturnMethod = Literal["log", "simple"]

#: Trading sessions per year used for annualization.
TRADING_DAYS: int = 252

#: Default return definition. Log returns add up over time, which keeps
#: square-root-of-time annualization of volatility exact.
DEFAULT_RETURN_METHOD: ReturnMethod = "log"

#: Delta degrees of freedom for variance: 1 is the unbiased sample estimator.
DEFAULT_DDOF: int = 1


def check_method(method: str) -> ReturnMethod:
    """Validate a return method name and return it typed."""
    if method not in ("log", "simple"):
        raise ValueError(f"method must be 'log' or 'simple', got {method!r}")
    return method  # type: ignore[return-value]


def annualize_vol(sigma: float, periods: int = TRADING_DAYS) -> float:
    """Scale a per-period volatility to annual by ``sqrt(periods)``."""
    return float(sigma) * math.sqrt(periods)


def annualize_return(
    mu: float,
    method: ReturnMethod = DEFAULT_RETURN_METHOD,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualize a mean per-period return.

    Log returns scale linearly (``mu * periods``). Simple returns compound
    (``(1 + mu) ** periods - 1``).
    """
    check_method(method)
    if method == "log":
        return float(mu) * periods
    return float((1.0 + mu) ** periods - 1.0)
