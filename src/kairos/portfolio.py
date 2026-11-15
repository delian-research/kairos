"""Portfolio math: returns from weights, risk decomposition and simple
weighting schemes.

Weights are a mapping or Series of ``asset -> weight``. They are used as
given and never rescaled: weights that sum to less than 1 leave the rest in
cash earning zero, and a sum above 1 is leverage. Covariance matrices are
DataFrames labelled by asset on both axes, e.g. from
:func:`kairos.ledoit_wolf`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ._util import require_n, span
from .conventions import (
    DEFAULT_RETURN_METHOD,
    TRADING_DAYS,
    ReturnMethod,
    check_method,
)

WeightsLike = Mapping[str, float] | pd.Series


def _weights(weights: WeightsLike, assets: pd.Index) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    if len(w) == 0:
        raise ValueError("weights is empty")
    if w.isna().any() or not np.isfinite(w.to_numpy()).all():
        raise ValueError("weights contains NaN or infinite values")
    if w.index.has_duplicates:
        raise ValueError("weights names an asset more than once")
    missing = [a for a in w.index if a not in assets]
    if missing:
        raise ValueError(f"weights names assets that are not in the data: {missing}")
    return w


def _cov_matrix(cov: pd.DataFrame) -> np.ndarray:
    if not isinstance(cov, pd.DataFrame):
        raise TypeError("cov must be a DataFrame labelled by asset on both axes")
    if cov.shape[0] != cov.shape[1] or not cov.index.equals(cov.columns):
        raise ValueError("cov must be square with the same labels on both axes")
    m = cov.to_numpy(dtype=float)
    if not np.isfinite(m).all():
        raise ValueError("cov contains NaN or infinite values")
    return (m + m.T) / 2.0


def _w_and_cov(weights: WeightsLike, cov: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    _cov_matrix(cov)
    w = _weights(weights, cov.columns)
    sub = cov.loc[list(w.index), list(w.index)]
    return w, _cov_matrix(sub)


def portfolio_returns(
    r: pd.DataFrame,
    weights: WeightsLike,
    method: ReturnMethod = DEFAULT_RETURN_METHOD,
    *,
    rebalance: str | None = "every",
) -> pd.Series:
    """Return series of a weighted portfolio.

    Args:
        r: Asset returns of kind ``method``, one column per asset. Only the
            assets named in ``weights`` are used, on rows where all of them
            have data.
        weights: Target weights.
        method: Kind of returns in ``r``; the result is the same kind.
        rebalance: ``"every"`` (default) resets to the target weights each
            period (constant weights). ``None`` is buy-and-hold: weights
            drift with prices from the first date. A pandas frequency such as
            ``"ME"`` or ``"QE"`` resets to target at the start of each such
            period and lets weights drift inside it.

    Log returns are converted to simple returns before weighting, because a
    portfolio's simple return is the weighted sum of its assets' simple
    returns; that is not true of log returns. Settings are in
    ``result.attrs["kairos"]``.
    """
    check_method(method)
    if not isinstance(r, pd.DataFrame):
        raise TypeError("r must be a DataFrame with one column per asset")
    w = _weights(weights, r.columns)
    x = r[list(w.index)].astype(float).dropna()
    require_n(len(x), 1, "portfolio_returns")
    simple = x.to_numpy() if method == "simple" else np.expm1(x.to_numpy())
    wv = w.to_numpy()
    cash = 1.0 - float(wv.sum())

    if rebalance == "every":
        port = np.dot(simple, wv)
        if (port <= -1.0).any():
            raise ValueError("portfolio lost 100% or more in a single period")
    else:
        positions = pd.Series(np.arange(len(x)), index=x.index)
        if rebalance is None:
            groups = [positions.to_numpy()]
        else:
            if not isinstance(x.index, pd.DatetimeIndex):
                raise TypeError("a rebalance frequency needs a DatetimeIndex")
            grouped = positions.groupby(pd.Grouper(freq=rebalance))
            groups = [g.to_numpy() for _, g in grouped if len(g)]
        port = np.empty(len(x))
        for pos in groups:
            # Value of 1 unit invested at target weights at the group start.
            value = cash + np.dot(np.cumprod(1.0 + simple[pos], axis=0), wv)
            if (value <= 0).any():
                raise ValueError("portfolio value fell to zero or below")
            port[pos] = value / np.concatenate([[1.0], value[:-1]]) - 1.0

    out = pd.Series(
        port if method == "simple" else np.log1p(port), index=x.index, name="portfolio"
    )
    start, end = span(x.index)
    out.attrs["kairos"] = {
        "method": "portfolio-returns",
        "return_method": method,
        "rebalance": "buy-and-hold" if rebalance is None else rebalance,
        "weights": {str(k): float(v) for k, v in w.items()},
        "weight_sum": float(wv.sum()),
        "n": len(x),
        "start": start,
        "end": end,
    }
    return out


def portfolio_vol(
    weights: WeightsLike,
    cov: pd.DataFrame,
    *,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> float:
    """Portfolio volatility ``sqrt(w' cov w)``, in the units of ``cov``.

    Pass a per-period covariance and ``annualize=True`` to scale by
    ``sqrt(periods)``.
    """
    w, s = _w_and_cov(weights, cov)
    wv = w.to_numpy()
    sigma = math.sqrt(max(float(np.dot(wv, np.dot(s, wv))), 0.0))
    return sigma * math.sqrt(periods) if annualize else sigma


def risk_contributions(
    weights: WeightsLike,
    cov: pd.DataFrame,
    *,
    annualize: bool = False,
    periods: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Decompose portfolio volatility by asset.

    Columns: ``weight``; ``marginal`` (change in portfolio volatility per unit
    of weight, ``(cov w)_i / sigma``); ``contribution`` (``weight * marginal``,
    which sums to portfolio volatility); and ``pct`` (share of total, sums to
    1). Portfolio volatility is in ``df.attrs["kairos"]["portfolio_vol"]``.
    """
    w, s = _w_and_cov(weights, cov)
    wv = w.to_numpy()
    sw = np.dot(s, wv)
    sigma = math.sqrt(max(float(np.dot(wv, sw)), 0.0))
    if sigma == 0:
        raise ValueError("portfolio volatility is zero; nothing to decompose")
    scale = math.sqrt(periods) if annualize else 1.0
    marginal = sw / sigma
    out = pd.DataFrame(
        {
            "weight": wv,
            "marginal": marginal * scale,
            "contribution": wv * marginal * scale,
            "pct": wv * marginal / sigma,
        },
        index=w.index,
    )
    out.attrs["kairos"] = {
        "method": "risk-contributions",
        "portfolio_vol": sigma * scale,
        "annualized": annualize,
        "periods": periods,
        "weight_sum": float(wv.sum()),
    }
    return out


def diversification_ratio(weights: WeightsLike, cov: pd.DataFrame) -> float:
    """Weighted average asset volatility over portfolio volatility.

    1.0 means no diversification benefit (one asset, or perfectly correlated
    assets); higher is more diversified.
    """
    w, s = _w_and_cov(weights, cov)
    wv = w.to_numpy()
    sigma = math.sqrt(max(float(np.dot(wv, np.dot(s, wv))), 0.0))
    if sigma == 0:
        raise ValueError("portfolio volatility is zero")
    return float(np.dot(wv, np.sqrt(np.diag(s)))) / sigma


def _weight_series(values: np.ndarray, cov: pd.DataFrame, method: str) -> pd.Series:
    out = pd.Series(values, index=cov.index, name="weight")
    out.attrs["kairos"] = {"method": method, "weight_sum": float(values.sum())}
    return out


def _positive_variances(s: np.ndarray) -> np.ndarray:
    var = np.diag(s)
    if (var <= 0).any():
        raise ValueError("every asset needs a positive variance")
    return var


def inverse_vol_weights(cov: pd.DataFrame) -> pd.Series:
    """Weights proportional to ``1 / volatility``, summing to 1.

    Ignores correlations; equals risk parity only when all pairwise
    correlations are equal.
    """
    s = _cov_matrix(cov)
    inv = 1.0 / np.sqrt(_positive_variances(s))
    return _weight_series(inv / inv.sum(), cov, "inverse-vol")


def min_variance_weights(cov: pd.DataFrame, *, long_only: bool = True) -> pd.Series:
    """Fully invested minimum-variance weights (sum to 1).

    With ``long_only=False`` this is the closed form
    ``inv(cov) 1 / (1' inv(cov) 1)`` and can short. With ``long_only=True``
    (default) weights are also constrained to be non-negative, solved exactly
    by an active-set method. A singular ``cov`` raises; shrink it first, e.g.
    with :func:`kairos.ledoit_wolf`.
    """
    s = _cov_matrix(cov)
    _positive_variances(s)
    n = len(s)
    free = np.ones(n, dtype=bool)
    try:
        for _ in range(20 * n + 20):
            idx = np.flatnonzero(free)
            y = np.linalg.solve(s[np.ix_(idx, idx)], np.ones(len(idx)))
            wf = y / y.sum()
            if not long_only:
                return _weight_series(wf, cov, "min-variance")
            if (wf < -1e-12).any():
                free[idx[int(np.argmin(wf))]] = False
                continue
            w = np.zeros(n)
            w[idx] = np.clip(wf, 0.0, None)
            w /= w.sum()
            # KKT check: an excluded asset must not lower variance if added.
            grad = np.dot(s, w)
            lam = float(np.dot(w, grad))
            gap = np.where(free, 0.0, grad - lam)
            j = int(np.argmin(gap))
            if gap[j] < -1e-10 * lam:
                free[j] = True
                continue
            return _weight_series(w, cov, "min-variance-long-only")
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "cov is singular; shrink it first (e.g. kairos.ledoit_wolf)"
        ) from exc
    raise RuntimeError("min_variance_weights did not converge")


def risk_parity_weights(
    cov: pd.DataFrame,
    *,
    budget: WeightsLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> pd.Series:
    """Long-only weights whose risk contributions match a budget.

    With no ``budget`` every asset contributes equally to portfolio
    volatility (equal risk contribution). ``budget`` gives each asset's share
    of risk; it must be positive and is rescaled to sum to 1. Solved by
    cyclical coordinate descent on Spinu's (2013) convex formulation.
    """
    s = _cov_matrix(cov)
    var = _positive_variances(s)
    n = len(s)
    b: np.ndarray
    if budget is None:
        b = np.full(n, 1.0 / n)
    else:
        bs = _weights(budget, cov.columns).reindex(cov.columns)
        if bs.isna().any() or (bs <= 0).any():
            raise ValueError("budget needs a positive entry for every asset")
        b = bs.to_numpy() / float(bs.sum())
    x = 1.0 / np.sqrt(var)
    for _ in range(max_iter):
        prev = x.copy()
        for i in range(n):
            off = float(np.dot(s[i], x)) - s[i, i] * x[i]
            x[i] = (-off + math.sqrt(off * off + 4.0 * s[i, i] * b[i])) / (
                2.0 * s[i, i]
            )
        if np.abs(x - prev).max() <= tol * np.abs(x).max():
            return _weight_series(x / x.sum(), cov, "risk-parity")
    raise RuntimeError("risk_parity_weights did not converge")
