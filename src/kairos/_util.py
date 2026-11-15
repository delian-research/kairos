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


def check_level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be strictly between 0 and 1, got {level}")
    return float(level)


def rf_label(rf: float | pd.Series) -> float | str:
    """How a risk-free argument is recorded in ``Estimate.params``."""
    return "series" if isinstance(rf, pd.Series) else float(rf)


def frame_attrs(df: pd.DataFrame, **meta: Any) -> pd.DataFrame:
    """Attach kairos metadata to a DataFrame result (``df.attrs["kairos"]``)."""
    df.attrs["kairos"] = meta
    return df


def is_zero_scale(scale: float, location: float) -> bool:
    """True if a dispersion is zero up to floating-point noise.

    The sample std of a constant series is ~1e-18, not exactly 0.
    """
    return scale <= 1e-12 * max(abs(location), 1e-300) or scale == 0.0
