from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos as k


def test_summary_matches_individual_functions(dates, rng):
    b = pd.Series(rng.normal(0.0003, 0.01, len(dates)), index=dates, name="bench")
    a = 1.2 * b + pd.Series(rng.normal(0, 0.005, len(dates)), index=dates)
    df = pd.DataFrame({"a": a, "b": b})
    s = k.summary(df, benchmark=b, rf=0.0001)
    assert list(s.columns) == ["a", "b"]
    assert s.loc["sharpe", "a"] == pytest.approx(
        k.sharpe_ratio(a, 0.0001, annualize=True).value
    )
    assert s.loc["beta", "a"] == pytest.approx(k.beta(a, b).value)
    assert s.loc["beta", "b"] == pytest.approx(1.0)
    assert s.loc["var_95", "a"] == pytest.approx(k.value_at_risk(a).value)
    meta = s.attrs["kairos"]
    assert meta["annualized_ratios"] is True
    assert meta["spans"]["a"] == (dates[0], dates[-1])


def test_summary_series_and_undefined_metric(dates):
    r = pd.Series(np.full(30, 0.001), index=dates[:30], name="up_only")
    s = k.summary(r)
    assert list(s.columns) == ["up_only"]
    assert np.isnan(s.loc["calmar", "up_only"])  # no drawdown
    assert "beta" not in s.index


def test_summary_needs_enough_data(dates):
    with pytest.raises(ValueError, match="at least 20"):
        k.summary(pd.Series(0.01, index=dates[:5]))
