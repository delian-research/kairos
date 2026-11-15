from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import kairos as k


def test_realized_vol_matches_numpy(dates: pd.DatetimeIndex, rng) -> None:
    r = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    est = k.realized_vol(r)
    assert est.value == pytest.approx(np.std(r.to_numpy(), ddof=1))
    assert est.n == len(r)
    assert est.start == dates[0] and est.end == dates[-1]
    assert est.method == "close-to-close"
    assert est.params["ddof"] == 1 and est.params["annualized"] is False


def test_window_and_annualization(dates: pd.DatetimeIndex, rng) -> None:
    r = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    daily = k.realized_vol(r, window=252)
    annual = k.realized_vol(r, window=252, annualize=True)
    assert daily.n == 252 and daily.start == dates[-252]
    assert annual.value == pytest.approx(daily.value * math.sqrt(252))
    assert daily.value == pytest.approx(np.std(r.to_numpy()[-252:], ddof=1))


def test_window_too_long_raises(rng) -> None:
    r = pd.Series(rng.normal(0, 0.01, 100))
    with pytest.raises(ValueError):
        k.realized_vol(r, window=252)


def test_constant_prices_have_zero_vol() -> None:
    r = k.returns(pd.Series([50.0] * 30))
    assert k.realized_vol(r).value == 0.0


def test_dataframe_gives_dict(dates, rng) -> None:
    df = pd.DataFrame(
        {"a": rng.normal(0, 0.01, len(dates)), "b": rng.normal(0, 0.02, len(dates))},
        index=dates,
    )
    out = k.realized_vol(df)
    assert set(out) == {"a", "b"}
    assert out["b"].value > out["a"].value


def test_rolling_vol(dates, rng) -> None:
    r = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    rv = k.rolling_vol(r, 20)
    assert rv.iloc[:19].isna().all()
    assert rv.iloc[-1] == pytest.approx(np.std(r.to_numpy()[-20:], ddof=1))


def test_ewma_vol_tracks_recent_regime(dates, rng) -> None:
    calm = rng.normal(0, 0.005, 400)
    wild = rng.normal(0, 0.03, 200)
    r = pd.Series(np.concatenate([calm, wild]), index=dates)
    est = k.ewma_vol(r, halflife=10)
    assert est.value == pytest.approx(0.03, rel=0.35)
    assert est.params["demeaned"] is False


def _gbm_ohlc(rng, sigma: float, n_days: int, steps: int = 390):
    """Simulate intraday GBM paths with no drift and no overnight gap."""
    dt = 1.0 / steps
    shocks = rng.normal(0, sigma * math.sqrt(dt), (n_days, steps))
    logpath = np.cumsum(shocks, axis=1)
    open_ = np.ones(n_days)
    paths = np.exp(np.concatenate([np.zeros((n_days, 1)), logpath], axis=1))
    idx = pd.RangeIndex(n_days)
    return (
        pd.Series(open_, index=idx),
        pd.Series(paths.max(axis=1), index=idx),
        pd.Series(paths.min(axis=1), index=idx),
        pd.Series(paths[:, -1], index=idx),
    )


def test_range_estimators_recover_sigma(rng) -> None:
    sigma = 0.02
    o, h, lo, c = _gbm_ohlc(rng, sigma, n_days=2000)
    # Discrete sampling biases range estimators slightly low, hence the tolerance.
    assert k.parkinson_vol(h, lo).value == pytest.approx(sigma, rel=0.08)
    assert k.garman_klass_vol(o, h, lo, c).value == pytest.approx(sigma, rel=0.08)


def test_range_estimators_validate() -> None:
    with pytest.raises(ValueError):
        k.parkinson_vol(pd.Series([1.0]), pd.Series([2.0]))
