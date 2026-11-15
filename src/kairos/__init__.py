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
from .covariance import correlation, ewma_cov, ledoit_wolf, nearest_psd
from .drawdown import drawdown_series, max_drawdown
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
from .returns import cumulative, last_n, returns
from .summary import summary
from .tail import (
    excess_kurtosis,
    expected_shortfall,
    skewness,
    value_at_risk,
    worst_periods,
)
from .transforms import (
    active_returns,
    excess_returns,
    resample_returns,
    total_return_prices,
)
from .volatility import (
    ewma_vol,
    garman_klass_vol,
    parkinson_vol,
    realized_vol,
    rogers_satchell_vol,
    rolling_vol,
    vol_cone,
    vol_percentile,
    yang_zhang_vol,
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
    "active_returns",
    "annualize_return",
    "annualize_vol",
    "annualized_return",
    "beta",
    "calmar_ratio",
    "capture_ratio",
    "correlation",
    "cumulative",
    "drawdown_series",
    "ewma_cov",
    "ewma_vol",
    "excess_kurtosis",
    "excess_returns",
    "expected_shortfall",
    "garman_klass_vol",
    "hit_rate",
    "information_ratio",
    "last_n",
    "ledoit_wolf",
    "max_drawdown",
    "nearest_psd",
    "newey_west_lags",
    "parkinson_vol",
    "realized_vol",
    "resample_returns",
    "returns",
    "rogers_satchell_vol",
    "rolling_beta",
    "rolling_vol",
    "sharpe_ratio",
    "skewness",
    "sortino_ratio",
    "summary",
    "total_return_prices",
    "tracking_error",
    "value_at_risk",
    "vol_cone",
    "vol_percentile",
    "worst_periods",
    "yang_zhang_vol",
]
