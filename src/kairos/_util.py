"""Internal helpers shared by the estimator modules."""

from __future__ import annotations

from typing import Any

import pandas as pd


def clean_series(x: pd.Series, name: str = "series") -> pd.Series:
    if not isinstance(x, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(x).__name__}")
    return x.astype(float).dropna()


def require_n(n: int, minimum: int, what: str) -> None:
    if n < minimum:
        raise ValueError(f"{what} needs at least {minimum} observations, got {n}")


def span(index: pd.Index) -> tuple[Any, Any]:
    if len(index) == 0:
        return None, None
    return index[0], index[-1]


def align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Inner-join two Series on their index and drop rows with any NaN."""
    df = pd.concat([a.astype(float), b.astype(float)], axis=1, join="inner").dropna()
    return df.iloc[:, 0], df.iloc[:, 1]
