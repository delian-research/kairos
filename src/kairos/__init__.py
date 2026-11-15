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
from .describe import catalog, guide, render, to_dict
from .drawdown import drawdown_series, max_drawdown
from .events import event_study
from .inference import bootstrap_ci, probabilistic_sharpe
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
from .portfolio import (
    diversification_ratio,
    inverse_vol_weights,
    min_variance_weights,
    portfolio_returns,
    portfolio_vol,
    risk_contributions,
    risk_parity_weights,
)
from .regression import adjust_beta, dimson_beta, downside_beta, factor_regression
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
    "adjust_beta",
    "annualize_return",
    "annualize_vol",
    "annualized_return",
    "beta",
    "bootstrap_ci",
    "calmar_ratio",
    "capture_ratio",
    "catalog",
    "correlation",
    "cumulative",
    "dimson_beta",
    "diversification_ratio",
    "downside_beta",
    "drawdown_series",
    "event_study",
    "ewma_cov",
    "ewma_vol",
    "excess_kurtosis",
    "excess_returns",
    "expected_shortfall",
    "factor_regression",
    "garman_klass_vol",
    "guide",
    "hit_rate",
    "information_ratio",
    "inverse_vol_weights",
    "last_n",
    "ledoit_wolf",
    "max_drawdown",
    "min_variance_weights",
    "nearest_psd",
    "newey_west_lags",
    "parkinson_vol",
    "portfolio_returns",
    "portfolio_vol",
    "probabilistic_sharpe",
    "realized_vol",
    "render",
    "resample_returns",
    "returns",
    "risk_contributions",
    "risk_parity_weights",
    "rogers_satchell_vol",
    "rolling_beta",
    "rolling_vol",
    "sharpe_ratio",
    "skewness",
    "sortino_ratio",
    "summary",
    "to_dict",
    "total_return_prices",
    "tracking_error",
    "value_at_risk",
    "vol_cone",
    "vol_percentile",
    "worst_periods",
    "yang_zhang_vol",
]
