from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import kairos as k


@pytest.fixture
def r(dates, rng):
    return pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates)


def test_cagr_constant_return(dates):
    g = 0.001
    simple = pd.Series(g, index=dates[:252])
    est = k.annualized_return(simple, "simple")
    assert est.value == pytest.approx((1 + g) ** 252 - 1)
    assert k.annualized_return(np.log1p(simple), "log").value == pytest.approx(
        est.value
    )


def test_sharpe(r):
    est = k.sharpe_ratio(r, 0.0001)
    x = r - 0.0001
    assert est.value == pytest.approx(x.mean() / x.std(ddof=1))
    ann = k.sharpe_ratio(r, 0.0001, annualize=True)
    assert ann.value == pytest.approx(est.value * math.sqrt(252))
    assert est.params["rf"] == 0.0001


def test_sharpe_zero_vol_raises(dates):
    with pytest.raises(ValueError, match="zero volatility"):
        k.sharpe_ratio(pd.Series(0.01, index=dates[:10]))


def test_sortino_by_hand():
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    dd = math.sqrt((0.01**2 + 0.02**2) / 4)
    assert k.sortino_ratio(r).value == pytest.approx(r.mean() / dd)


def test_calmar(r):
    est = k.calmar_ratio(r)
    cagr = k.annualized_return(r).value
    mdd = k.max_drawdown(r).value
    assert est.value == pytest.approx(cagr / abs(mdd))


def test_tracking_error_and_information_ratio(r, rng):
    b = r - pd.Series(rng.normal(0.0001, 0.002, len(r)), index=r.index)
    a = r - b
    assert k.tracking_error(r, b).value == pytest.approx(a.std(ddof=1))
    assert k.information_ratio(r, b).value == pytest.approx(a.mean() / a.std())


def test_capture_ratio_of_leveraged_series(dates, rng):
    b = pd.Series(rng.normal(0, 0.01, 500), index=dates[:500])
    up = k.capture_ratio(b * 2, b, "up", "simple")
    down = k.capture_ratio(b * 2, b, "down", "simple")
    assert 1.9 < up.value < 2.1
    assert 1.9 < down.value < 2.1
    assert k.capture_ratio(b, b, "up").value == pytest.approx(1.0)


def test_hit_rate():
    assert k.hit_rate(pd.Series([0.01, -0.01, 0.0, 0.02])).value == 0.5
