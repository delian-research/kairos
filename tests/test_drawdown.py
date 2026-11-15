from __future__ import annotations

import pandas as pd
import pytest

import kairos as k


def _path(levels: list[float]) -> pd.Series:
    """Simple returns that trace the given wealth levels starting from 1.0."""
    idx = pd.bdate_range("2024-01-01", periods=len(levels))
    wealth = pd.Series(levels, index=idx)
    prev = wealth.shift(1).fillna(1.0)
    return wealth / prev - 1.0


def test_known_peak_trough_recovery() -> None:
    r = _path([1.1, 1.2, 0.9, 1.0, 1.25, 1.3])
    est = k.max_drawdown(r, method="simple")
    assert est.value == pytest.approx(0.9 / 1.2 - 1.0)
    assert est.params["peak"] == r.index[1]
    assert est.params["trough"] == r.index[2]
    assert est.params["recovery"] == r.index[4]


def test_unrecovered_and_loss_from_start() -> None:
    r = _path([0.95, 0.8, 0.85])
    est = k.max_drawdown(r, method="simple")
    assert est.value == pytest.approx(-0.2)
    assert est.params["peak"] is None  # peak was the starting value
    assert est.params["recovery"] is None


def test_no_drawdown() -> None:
    r = _path([1.01, 1.02, 1.03])
    est = k.max_drawdown(r, method="simple")
    assert est.value == 0.0
    assert est.params["trough"] is None


def test_drawdown_series_nonpositive() -> None:
    dd = k.drawdown_series(_path([1.1, 0.99, 1.2]), method="simple")
    assert (dd <= 0).all()
    assert dd.iloc[1] == pytest.approx(0.99 / 1.1 - 1)


def test_log_and_simple_agree_on_same_prices() -> None:
    px = pd.Series([100.0, 120.0, 90.0, 130.0])
    a = k.max_drawdown(k.returns(px, "log"), method="log").value
    b = k.max_drawdown(k.returns(px, "simple"), method="simple").value
    assert a == pytest.approx(b) == pytest.approx(-0.25)
