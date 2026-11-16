"""Correlation and covariance matrices.

Matrix results are DataFrames that carry their settings in
``df.attrs["kairos"]``.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Literal, TypeVar

import numpy as np
import pandas as pd

from ._util import align, check_level, check_returns, frame_attrs, require_n, span
from .regression import residual_returns
from .results import Estimate

MatrixT = TypeVar("MatrixT", pd.DataFrame, np.ndarray)


def correlation(
    r: pd.DataFrame,
    *,
    min_obs: int | None = None,
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.DataFrame:
    """Pairwise correlation matrix.

    Each pair uses every date where both columns have data, so pairs can
    use different samples. A pair with fewer than ``min_obs`` common
    observations (default 2) is NaN: that is the one place kairos returns
    NaN, because the matrix still has to be square. Pairwise matrices need
    not be positive semi-definite; see :func:`nearest_psd`.
    """
    if not isinstance(r, pd.DataFrame):
        raise TypeError("correlation needs a DataFrame with one column per asset")
    mp = 2 if min_obs is None else min_obs
    out = r.astype(float).corr(method=method, min_periods=mp)
    counts = r.notna().astype(int)
    pair_n = counts.T.dot(counts)
    start, end = span(r.index)
    return frame_attrs(
        out,
        method=f"correlation-{method}",
        min_obs=mp,
        start=start,
        end=end,
        pair_n=pair_n,
    )


def _corr(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation that raises, instead of warning and returning NaN,
    when either series is constant."""
    if float(np.ptp(x.to_numpy())) == 0.0 or float(np.ptp(y.to_numpy())) == 0.0:
        raise ValueError("correlation is undefined: a series is constant")
    return float(x.corr(y))


def pair_correlation(
    a: pd.Series,
    b: pd.Series,
    *,
    controls: pd.Series | pd.DataFrame | None = None,
    level: float = 0.95,
) -> Estimate:
    """Pearson correlation between two return series, with a confidence
    interval and optional controls.

    With ``controls`` (e.g. the market's returns) this is the **partial
    correlation**: both series are first regressed on the controls and the
    residuals are correlated. Two stocks with high market betas look
    correlated even when nothing links them directly; controlling for the
    market isolates the relationship specific to the pair.

    ``params`` hold ``lower`` and ``upper`` (Fisher z interval at ``level``)
    and, with controls, ``raw_correlation``. The Fisher interval assumes
    independent observations; for daily returns
    ``kairos.bootstrap_ci(r[[x, y]], lambda d: d[x].corr(d[y]))`` is more
    robust.
    """
    level = check_level(level)
    names: list[str] = []
    raw: float | None = None
    if controls is None:
        x, y = align(a, b)
    else:
        c = controls.to_frame() if isinstance(controls, pd.Series) else controls
        names = [str(n) for n in c.columns]
        df = pd.concat(
            [a.astype(float).rename("__a__"), b.astype(float).rename("__b__"), c],
            axis=1,
            join="inner",
        ).dropna()
        raw = _corr(df["__a__"], df["__b__"])
        ctl = df.iloc[:, 2:]
        x = residual_returns(df["__a__"], ctl)
        y = residual_returns(df["__b__"], ctl)
    check_returns(x, "a")
    check_returns(y, "b")
    n = len(x)
    dof = n - 3 - len(names)
    require_n(dof, 1, "pair_correlation (observations beyond 3 + controls)")
    rho = _corr(x, y)
    if abs(rho) >= 1.0:
        lower = upper = rho
    else:
        z = math.atanh(rho)
        half = NormalDist().inv_cdf(0.5 + level / 2.0) / math.sqrt(dof)
        lower, upper = math.tanh(z - half), math.tanh(z + half)
    params: dict[str, object] = {
        "lower": lower,
        "upper": upper,
        "level": level,
        "interval": "fisher-z",
    }
    if controls is not None:
        params.update({"controls": names, "raw_correlation": raw})
    start, end = span(x.index)
    return Estimate(
        value=rho,
        n=n,
        start=start,
        end=end,
        method="partial-correlation" if names else "correlation-pearson",
        params=params,
    )


def rolling_correlation(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation of two return series over ``window``
    periods. The first ``window - 1`` values are NaN."""
    x, y = align(a, b)
    check_returns(x, "a")
    check_returns(y, "b")
    require_n(window, 3, "rolling_correlation window")
    require_n(len(x), window, "rolling_correlation")
    out = x.rolling(window).corr(y)
    out.name = "rolling_correlation"
    return out


def ewma_cov(r: pd.DataFrame, halflife: float, *, demean: bool = False) -> pd.DataFrame:
    """Exponentially weighted covariance as of the last date.

    Uses rows where every column has data. Weights decay by half every
    ``halflife`` periods and are normalized to sum to 1 (the same weights as
    ``pandas.ewm(halflife=..., adjust=True)``). With ``demean=False``
    (RiskMetrics, the default) the diagonal's square root equals
    :func:`kairos.ewma_vol` for each column.
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be positive, got {halflife}")
    x = r.astype(float).dropna()
    t = len(x)
    require_n(t, 2, "ewma_cov")
    decay = 0.5 ** (1.0 / halflife)
    w = decay ** np.arange(t - 1, -1, -1, dtype=float)
    w /= w.sum()
    a = x.to_numpy()
    if demean:
        a = a - np.dot(w, a)
    cov = np.dot((a * w[:, None]).T, a)
    out = pd.DataFrame(cov, index=x.columns, columns=x.columns)
    start, end = span(x.index)
    return frame_attrs(
        out,
        method="ewma-cov",
        n=t,
        start=start,
        end=end,
        halflife=halflife,
        demeaned=demean,
    )


def ledoit_wolf(r: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf (2004) shrinkage covariance toward a scaled identity.

    ``(1 - s) * S + s * mu * I``, where ``S`` is the sample covariance
    (``ddof=0``), ``mu`` the mean variance and ``s`` the closed-form optimal
    shrinkage intensity, recorded as ``df.attrs["kairos"]["shrinkage"]``.
    Uses rows where every column has data. The result is positive definite
    whenever ``s > 0``, even with more assets than observations.
    """
    x = r.astype(float).dropna()
    n, p = x.shape
    require_n(n, 2, "ledoit_wolf")
    a = x.to_numpy()
    a = a - a.mean(axis=0)
    s = np.dot(a.T, a) / n
    mu = float(np.trace(s)) / p
    # Distance of S from the target, and the estimation noise in S.
    delta = float(((s - mu * np.eye(p)) ** 2).sum()) / p
    a2 = a**2
    beta_bar = float(np.dot(a2.T, a2).sum() / n - (s**2).sum()) / (n * p)
    beta = min(beta_bar, delta)
    shrink = 0.0 if delta == 0 else beta / delta
    cov = (1.0 - shrink) * s + shrink * mu * np.eye(p)
    out = pd.DataFrame(cov, index=x.columns, columns=x.columns)
    start, end = span(x.index)
    return frame_attrs(
        out,
        method="ledoit-wolf",
        n=n,
        start=start,
        end=end,
        shrinkage=shrink,
        target="scaled-identity",
        ddof=0,
    )


def nearest_psd(cov: MatrixT, *, eps: float = 0.0, keep_diag: bool = True) -> MatrixT:
    """Repair a symmetric matrix so it is positive semi-definite.

    Symmetrizes, clips eigenvalues below ``eps`` to ``eps`` and, with
    ``keep_diag=True``, rescales so the diagonal (variances, or the ones of a
    correlation matrix) is unchanged. This is eigenvalue clipping, a fast
    approximation, not Higham's (2002) exact nearest correlation matrix.
    """
    m = np.asarray(cov, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("cov must be a square matrix")
    if not np.isfinite(m).all():
        raise ValueError("cov contains NaN or infinite values")
    m = (m + m.T) / 2.0
    vals, vecs = np.linalg.eigh(m)
    fixed = np.dot(vecs * np.clip(vals, eps, None), vecs.T)
    if keep_diag:
        d0, d1 = np.diag(m), np.diag(fixed)
        if (d0 < 0).any():
            raise ValueError("cov has a negative diagonal entry")
        scale = np.sqrt(np.divide(d0, d1, out=np.zeros_like(d0), where=d1 > 0))
        fixed = fixed * np.outer(scale, scale)
    fixed = (fixed + fixed.T) / 2.0
    if isinstance(cov, pd.DataFrame):
        return pd.DataFrame(fixed, index=cov.index, columns=cov.columns)
    return fixed
