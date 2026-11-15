from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos as k


@pytest.fixture
def market(dates, rng):
    return pd.Series(rng.normal(0.0003, 0.01, len(dates)), index=dates)


def test_recovers_injected_jump(market, rng):
    asset = 1.1 * market + rng.normal(0, 0.003, len(market))
    events = [market.index[i] for i in (200, 300, 400, 500)]
    for d in events:
        asset.loc[d] += 0.05
    es = k.event_study(asset, market, events, window=(-3, 3))
    assert list(es.index) == [-3, -2, -1, 0, 1, 2, 3]
    assert es.loc[0, "mean_ar"] == pytest.approx(0.05, abs=0.005)
    assert es.loc[3, "car"] == pytest.approx(0.05, abs=0.01)
    assert abs(es.loc[-1, "car"]) < 0.01
    assert es.loc[0, "t"] > 5
    assert (es["n_events"] == 4).all()
    meta = es.attrs["kairos"]
    assert meta["t_stat"] == "cross-sectional"
    assert len(meta["event_car"]) == 4


def test_non_trading_day_moves_to_next_session(market):
    asset = market.copy()
    saturday = pd.Timestamp("2025-03-01")
    assert saturday not in market.index
    es = k.event_study(asset, market, [saturday], window=(0, 1))
    assert es.attrs["kairos"]["events_used"] == [pd.Timestamp("2025-03-03")]
    assert es.attrs["kairos"]["t_stat"].startswith("time-series")


def test_edge_events_are_skipped_with_reasons(market):
    events = [market.index[5], market.index[300], market.index[-1]]
    es = k.event_study(market * 1.0, market, events, window=(-2, 2))
    meta = es.attrs["kairos"]
    assert meta["events_used"] == [market.index[300]]
    reasons = [why for _, why in meta["events_skipped"]]
    assert any("history" in r for r in reasons) and any("end" in r for r in reasons)


def test_no_usable_events_raises(market):
    with pytest.raises(ValueError, match="no usable events"):
        k.event_study(market * 1.0, market, [market.index[3]])
    assert np.isfinite(market).all()
