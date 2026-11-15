"""Correlation and covariance matrices.

Matrix results are DataFrames that carry their settings in
``df.attrs["kairos"]``.
"""

from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np
import pandas as pd

from ._util import frame_attrs, require_n, span

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
