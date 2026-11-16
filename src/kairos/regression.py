"""Factor regressions and beta variants.

All inputs are returns. Standard errors are classical OLS by default, or
Newey-West (Bartlett kernel) with ``hac=True``. p-values use the normal
approximation, which is accurate for the sample sizes typical of return data
(n in the hundreds) and slightly too small for very short samples.
"""

from __future__ import annotations

import math
from dataclasses import replace
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd

from ._util import align, check_returns, frame_attrs, require_n, span
from .beta import beta, newey_west_lags
from .results import Estimate

_N = NormalDist()


def _fit(
    asset: pd.Series, factors: pd.DataFrame | pd.Series, what: str
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align, validate and solve OLS with an intercept.

    Returns the aligned frame, factor names, design matrix, coefficients,
    residuals and ``inv(X'X)``.
    """
    if not isinstance(asset, pd.Series):
        raise TypeError("asset must be a pandas Series of returns")
    f = factors.to_frame() if isinstance(factors, pd.Series) else factors
    names = [str(c) for c in f.columns]
    if "alpha" in names or len(set(names)) != len(names):
        raise ValueError("factor names must be unique and not 'alpha'")
    df = pd.concat(
        [asset.astype(float).rename("__y__"), f.astype(float)], axis=1, join="inner"
    ).dropna()
    n, k = len(df), len(names)
    require_n(n, k + 2, what)
    y = df.iloc[:, 0]
    check_returns(y, "asset")
    yv = y.to_numpy()
    x = np.column_stack([np.ones(n), df.iloc[:, 1:].to_numpy()])

    # inv() "succeeds" on nearly singular matrices and returns garbage, so
    # test the rank explicitly.
    if np.linalg.matrix_rank(x) < k + 1:
        raise ValueError("factors are perfectly collinear (or one is constant)")
    bread = np.linalg.inv(np.dot(x.T, x))
    coef = np.dot(bread, np.dot(x.T, yv))
    u = yv - np.dot(x, coef)

    return df, names, x, coef, u, bread


def factor_regression(
    asset: pd.Series,
    factors: pd.DataFrame | pd.Series,
    *,
    hac: bool = False,
    lags: int | None = None,
) -> pd.DataFrame:
    """OLS of asset returns on one or more factor return series.

    ``asset = alpha + sum_k beta_k * factor_k + e``. Rows are inner-joined on
    the index and any row with a NaN is dropped.

    Returns a DataFrame with one row for ``alpha`` (per period) and one per
    factor, and columns ``coef``, ``se``, ``t`` and ``p``.
    ``df.attrs["kairos"]`` holds ``n``, ``start``, ``end``, ``r2``,
    ``adj_r2``, ``resid_vol`` (per-period standard deviation of residuals),
    ``se_type`` and ``lags``.
    """
    df, names, x, coef, u, bread = _fit(asset, factors, "factor_regression")
    n, k = len(df), len(names)
    yv = df.iloc[:, 0].to_numpy()
    ssr = float(np.dot(u, u))
    dof = n - k - 1

    used_lags: int | None = None
    if hac:
        used_lags = newey_west_lags(n) if lags is None else int(lags)
        if used_lags < 0:
            raise ValueError("lags must be non-negative")
        xu = x * u[:, None]
        meat = np.dot(xu.T, xu)
        for lag in range(1, min(used_lags, n - 1) + 1):
            g = np.dot(xu[lag:].T, xu[:-lag])
            meat += (1.0 - lag / (used_lags + 1.0)) * (g + g.T)
        cov = np.dot(bread, np.dot(meat, bread))
    else:
        cov = bread * (ssr / dof)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    t = np.divide(coef, se, out=np.full(k + 1, np.nan), where=se > 0)
    p = np.array(
        [2.0 * (1.0 - _N.cdf(abs(v))) if np.isfinite(v) else np.nan for v in t]
    )

    centered = yv - yv.mean()
    sst = float(np.dot(centered, centered))
    r2 = 1.0 - ssr / sst if sst > 0 else math.nan
    out = pd.DataFrame(
        {"coef": coef, "se": se, "t": t, "p": p}, index=["alpha", *names]
    )
    start, end = span(df.index)
    return frame_attrs(
        out,
        method="factor-regression",
        n=n,
        start=start,
        end=end,
        r2=r2,
        adj_r2=1.0 - (1.0 - r2) * (n - 1) / dof,
        resid_vol=math.sqrt(ssr / dof),
        se_type="newey-west" if hac else "ols",
        lags=used_lags,
        p_value="normal-approximation",
    )


def residual_returns(asset: pd.Series, factors: pd.DataFrame | pd.Series) -> pd.Series:
    """The part of an asset's returns not explained by the factors.

    Residuals from ``asset = alpha + sum_k beta_k * factor_k + e``. They
    average zero (the intercept is removed) and are uncorrelated with every
    factor in sample. Use them to study what is specific to the asset, e.g.
    market-neutral returns with the market as the single factor. The fitted
    ``alpha`` and ``betas`` are in ``result.attrs["kairos"]``.
    """
    df, names, _, coef, u, _ = _fit(asset, factors, "residual_returns")
    out = pd.Series(u, index=df.index, name=asset.name)
    start, end = span(df.index)
    out.attrs["kairos"] = {
        "method": "residual-returns",
        "n": len(df),
        "start": start,
        "end": end,
        "alpha": float(coef[0]),
        "betas": {name: float(c) for name, c in zip(names, coef[1:], strict=True)},
    }
    return out


def dimson_beta(asset: pd.Series, benchmark: pd.Series, lags: int = 1) -> Estimate:
    """Dimson (1979) beta: the sum of betas on current and lagged benchmark
    returns.

    Corrects the downward bias in beta for assets that trade less often than
    the benchmark (or in another time zone), whose prices react with a delay.
    ``params["betas"]`` lists the coefficients for lag 0, 1, ..., ``lags``.
    """
    if lags < 1:
        raise ValueError("lags must be at least 1")
    a, b = align(asset, benchmark)
    cols = {f"lag{i}": b.shift(i) for i in range(lags + 1)}
    reg = factor_regression(a, pd.DataFrame(cols))
    meta = reg.attrs["kairos"]
    betas = [float(v) for v in reg["coef"].iloc[1:]]
    return Estimate(
        value=float(sum(betas)),
        n=meta["n"],
        start=meta["start"],
        end=meta["end"],
        method="dimson-beta",
        params={
            "lags": lags,
            "betas": betas,
            "alpha": float(reg["coef"].iloc[0]),
            "r2": meta["r2"],
        },
    )


def downside_beta(
    asset: pd.Series, benchmark: pd.Series, threshold: float = 0.0
) -> Estimate:
    """Beta using only the periods where the benchmark return is below
    ``threshold``: how the asset moves when the market falls."""
    a, b = align(asset, benchmark)
    mask = b < threshold
    require_n(int(mask.sum()), 3, "downside_beta (periods below threshold)")
    est = beta(a[mask], b[mask])
    return replace(
        est, method="downside-beta", params={"threshold": threshold, **est.params}
    )


def adjust_beta(
    estimate: Estimate,
    method: Literal["blume", "vasicek"] = "blume",
    *,
    prior_mean: float = 1.0,
    prior_sd: float | None = None,
) -> Estimate:
    """Shrink a beta estimate toward a prior (the market beta of 1).

    * ``"blume"``: ``2/3 * beta + 1/3 * prior_mean`` (the "adjusted beta"
      shown by data vendors).
    * ``"vasicek"``: precision-weighted average of the estimate and the
      prior, using the estimate's standard error and ``prior_sd`` (the
      cross-sectional standard deviation of betas, which you must supply).
      Noisier estimates are pulled harder toward the prior.

    Takes the :class:`Estimate` from :func:`kairos.beta` and returns a new
    one with ``raw_beta`` and ``shrink_weight`` (the weight on the prior) in
    ``params``.
    """
    raw = estimate.value
    if method == "blume":
        weight = 1.0 / 3.0
    elif method == "vasicek":
        se = estimate.params.get("se")
        if se is None or prior_sd is None or prior_sd <= 0:
            raise ValueError(
                "vasicek needs an estimate with a standard error and a "
                "positive prior_sd"
            )
        weight = se**2 / (se**2 + prior_sd**2)
    else:
        raise ValueError(f"unknown adjustment {method!r}")
    return replace(
        estimate,
        value=(1.0 - weight) * raw + weight * prior_mean,
        method=f"{estimate.method}+{method}",
        params={
            **estimate.params,
            "raw_beta": raw,
            "prior_mean": prior_mean,
            "shrink_weight": weight,
        },
    )
