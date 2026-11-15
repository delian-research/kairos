from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos as k


def test_excess_returns_constant_and_series(dates, rng):
    r = pd.Series(rng.normal(0, 0.01, 10), index=dates[:10])
    assert np.allclose(k.excess_returns(r, 0.001), r - 0.001)
    rf = pd.Series(0.0002, index=dates[2:12])
    ex = k.excess_returns(r, rf)
    assert list(ex.index) == list(dates[2:10])
    assert np.allclose(ex, r.loc[dates[2:10]] - 0.0002)


def test_excess_returns_dataframe(dates):
    df = pd.DataFrame({"a": 0.01, "b": 0.02}, index=dates[:5])
    rf = pd.Series(0.005, index=dates[1:5])
    out = k.excess_returns(df, rf)
    assert out.shape == (4, 2)
    assert np.allclose(out["b"], 0.015)


def test_active_returns(dates):
    a = pd.Series([0.02, 0.01, 0.0], index=dates[:3])
    b = pd.Series([0.01, 0.01, 0.01], index=dates[:3])
    assert np.allclose(k.active_returns(a, b), [0.01, 0.0, -0.01])


def test_resample_log_sums_and_simple_compounds(dates, rng):
    simple = pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)
    log = np.log1p(simple)
    m_simple = k.resample_returns(simple, "ME", "simple")
    m_log = k.resample_returns(log, "ME", "log")
    assert np.allclose(np.log1p(m_simple), m_log)
    first = simple[simple.index.to_period("M") == simple.index[0].to_period("M")]
    assert m_simple.iloc[0] == pytest.approx((1 + first).prod() - 1)


def test_resample_drops_empty_bins():
    idx = pd.to_datetime(["2024-01-02", "2024-03-04"])
    out = k.resample_returns(pd.Series([0.01, 0.02], index=idx), "ME")
    assert len(out) == 2  # February has no data and is not a zero return


def test_resample_needs_datetime_index():
    with pytest.raises(TypeError):
        k.resample_returns(pd.Series([0.1, 0.2]), "ME")


def test_total_return_prices(dates):
    close = pd.Series(100.0, index=dates[:5])
    divs = pd.Series([1.0], index=[dates[2]])
    tr = k.total_return_prices(close, divs)
    assert tr.iloc[-1] == pytest.approx(100.0)
    r = tr.pct_change().dropna()
    assert r.loc[dates[2]] == pytest.approx(0.01)
    assert r.drop(dates[2]).abs().max() == pytest.approx(0.0)


def test_total_return_prices_rejects_unknown_ex_date(dates):
    close = pd.Series(100.0, index=dates[:5])
    with pytest.raises(ValueError, match="not in the close index"):
        k.total_return_prices(close, pd.Series([1.0], index=[dates[10]]))
    with pytest.raises(ValueError, match="first date"):
        k.total_return_prices(close, pd.Series([1.0], index=[dates[0]]))
