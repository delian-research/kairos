"""How much to trust a statistic: Sharpe ratio significance and bootstrap
confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Callable
from statistics import NormalDist
from typing import Any, SupportsFloat, cast

import numpy as np
import pandas as pd

from ._util import check_level, clean_returns, clean_series, require_n, rf_label, span
from .conventions import TRADING_DAYS
from .results import Estimate
from .transforms import excess_returns

_N = NormalDist()


def probabilistic_sharpe(
    r: pd.Series,
    benchmark_sharpe: float = 0.0,
    *,
    rf: float | pd.Series = 0.0,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> Estimate:
    """Probability that the true Sharpe ratio exceeds ``benchmark_sharpe``.

    Bailey and Lopez de Prado's (2012) probabilistic Sharpe ratio. It
    accounts for sample length, skewness and fat tails: a Sharpe of 1 over 60
    days with negative skew is much weaker evidence than over 5 years. The
    value is a probability in [0, 1]; above 0.95 is the usual bar.

    ``benchmark_sharpe`` and the reported ``params["sharpe"]`` and
    ``params["se"]`` are per period, or annualized if ``annualize=True``.
    """
    x = excess_returns(clean_returns(r, "r"), rf)
    n = len(x)
    require_n(n, 4, "probabilistic_sharpe")
    sd = float(x.std(ddof=1))
    if sd == 0:
        raise ValueError("probabilistic_sharpe is undefined for zero volatility")
    scale = math.sqrt(periods) if annualize else 1.0
    sr = float(x.mean()) / sd
    skew = float(cast(float, x.skew()))
    kurt = float(cast(float, x.kurt())) + 3.0
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if var_term <= 0:
        raise ValueError("Sharpe ratio variance is not positive for this sample")
    se = math.sqrt(var_term / (n - 1))
    z = (sr - benchmark_sharpe / scale) / se
    start, end = span(x.index)
    return Estimate(
        value=_N.cdf(z),
        n=n,
        start=start,
        end=end,
        method="probabilistic-sharpe",
        params={
            "sharpe": sr * scale,
            "benchmark_sharpe": benchmark_sharpe,
            "se": se * scale,
            "skew": skew,
            "kurtosis": kurt,
            "rf": rf_label(rf),
            "annualized": annualize,
            "periods": periods,
        },
    )


def bootstrap_ci(
    x: pd.Series | pd.DataFrame,
    statistic: Callable[[Any], SupportsFloat],
    *,
    level: float = 0.95,
    n_boot: int = 1000,
    block: int | None = None,
    seed: int | None = None,
) -> Estimate:
    """Confidence interval for any statistic by circular block bootstrap.

    Resamples blocks of consecutive observations (wrapping around the end) so
    that autocorrelation and volatility clustering inside a block are
    preserved, recomputes ``statistic`` on each resample and reports the
    percentile interval.

    Args:
        x: The data, usually returns. Pass a DataFrame for statistics of
            several series (correlation, beta): whole rows are resampled
            together, so the series stay aligned.
        statistic: Function of the data (same type as ``x``) returning a
            number or an :class:`Estimate`, e.g. ``kairos.sharpe_ratio`` or
            ``lambda d: d["a"].corr(d["b"])``.
        level: Confidence level of the interval.
        n_boot: Number of resamples.
        block: Block length. ``None`` uses ``round(n ** (1/3))``; 1 gives the
            ordinary i.i.d. bootstrap.
        seed: Seed for reproducible intervals.

    The value is the statistic on the original data; ``params`` hold
    ``lower``, ``upper`` and the standard error ``se`` (standard deviation
    across resamples).
    """
    level = check_level(level)
    s: pd.Series | pd.DataFrame
    if isinstance(x, pd.DataFrame):
        s = x.astype(float).dropna()
    else:
        s = clean_series(x, "x")
    n = len(s)
    require_n(n, 2, "bootstrap_ci")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    b = max(1, round(n ** (1.0 / 3.0))) if block is None else int(block)
    if not 1 <= b <= n:
        raise ValueError(f"block must be between 1 and {n}, got {b}")
    point = float(statistic(s))
    rng = np.random.default_rng(seed)
    vals = s.to_numpy()
    n_blocks = math.ceil(n / b)
    within = np.arange(b)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + within) % n).ravel()[:n]
        sample: pd.Series | pd.DataFrame
        if isinstance(s, pd.DataFrame):
            sample = pd.DataFrame(vals[idx], index=s.index, columns=s.columns)
        else:
            sample = pd.Series(vals[idx], index=s.index, name=s.name)
        stats[i] = float(statistic(sample))
    if not np.isfinite(stats).all():
        raise ValueError("statistic returned NaN or infinity on a resample")
    tail = (1.0 - level) / 2.0
    lower, upper = (float(v) for v in np.quantile(stats, [tail, 1.0 - tail]))
    start, end = span(s.index)
    return Estimate(
        value=point,
        n=n,
        start=start,
        end=end,
        method="bootstrap-ci",
        params={
            "lower": lower,
            "upper": upper,
            "se": float(stats.std(ddof=1)),
            "level": level,
            "n_boot": n_boot,
            "block": b,
            "seed": seed,
        },
    )
