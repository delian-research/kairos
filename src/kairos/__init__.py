"""kairos: price analytics with explicit conventions.

Defaults (see :mod:`kairos.conventions`):

* returns are **log** returns (``method="log"``),
* variances use the sample estimator (``ddof=1``),
* annualization uses **252** periods per year and happens only when
  ``annualize=True``.

Every scalar function returns an :class:`Estimate`. It records the value, how
many observations were used, the first and last dates, and the settings, so a
result always says how it was computed.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .beta import beta, newey_west_lags, rolling_beta
from .conventions import (
    DEFAULT_DDOF,
    DEFAULT_RETURN_METHOD,
    TRADING_DAYS,
    ReturnMethod,
    annualize_return,
    annualize_vol,
)
from .drawdown import drawdown_series, max_drawdown
from .results import Estimate
from .returns import cumulative, last_n, returns
from .volatility import (
    ewma_vol,
    garman_klass_vol,
    parkinson_vol,
    realized_vol,
    rolling_vol,
)

try:
    __version__ = version("kairos-analytics")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "DEFAULT_DDOF",
    "DEFAULT_RETURN_METHOD",
    "TRADING_DAYS",
    "Estimate",
    "ReturnMethod",
    "__version__",
    "annualize_return",
    "annualize_vol",
    "beta",
    "cumulative",
    "drawdown_series",
    "ewma_vol",
    "garman_klass_vol",
    "last_n",
    "max_drawdown",
    "newey_west_lags",
    "parkinson_vol",
    "realized_vol",
    "returns",
    "rolling_beta",
    "rolling_vol",
]
