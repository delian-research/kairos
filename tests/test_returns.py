from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import kairos as k


def test_simple_and_log_returns() -> None:
    px = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2024-01-01", periods=3))
    simple = k.returns(px, method="simple")
    log = k.returns(px, method="log")
    assert len(simple) == 2
    assert simple.iloc[0] == pytest.approx(0.10)
    assert simple.iloc[1] == pytest.approx(-0.10)
    assert log.iloc[0] == pytest.approx(math.log(1.1))
    assert simple.index[0] == px.index[1]


def test_default_method_is_log() -> None:
    px = pd.Series([1.0, 2.0])
    assert k.returns(px).iloc[0] == pytest.approx(math.log(2.0))


def test_dataframe_shape() -> None:
    px = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [10.0, 10.0, 5.0]})
    r = k.returns(px, method="simple")
    assert r.shape == (2, 2)
    assert r["b"].tolist() == pytest.approx([0.0, -0.5])


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        k.returns(pd.Series([1.0, 0.0]))
    with pytest.raises(ValueError):
        k.returns(pd.Series([1.0, 2.0]), method="pct")  # type: ignore[arg-type]


def test_last_n_takes_latest_and_raises_when_short() -> None:
    r = pd.Series([np.nan, 1.0, 2.0, 3.0])
    assert k.last_n(r, 2).tolist() == [2.0, 3.0]
    with pytest.raises(ValueError, match="only 3 are available"):
        k.last_n(r, 4)


def test_cumulative_round_trips_prices() -> None:
    px = pd.Series([100.0, 105.0, 95.0, 120.0])
    for method in ("log", "simple"):
        wealth = k.cumulative(k.returns(px, method=method), method=method)
        assert wealth.tolist() == pytest.approx((px.iloc[1:] / 100.0).tolist())
