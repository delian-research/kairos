from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import kairos as k


def test_catalog_lists_every_public_function():
    cat = k.catalog()
    functions = {n for n in k.__all__ if callable(getattr(k, n)) and n[0].islower()}
    assert set(cat.index) == functions
    assert (cat["summary"].str.len() > 10).all()
    assert cat.loc["sharpe_ratio", "area"] == "performance"
    assert cat.loc["beta", "signature"].startswith("beta(asset")


def test_guide_mentions_every_function_family():
    text = k.guide()
    for name in ("summary", "portfolio_returns", "risk_contributions", "event_study"):
        assert name in text
    assert "Prices are not returns" in text


def test_to_dict_is_json_safe(dates, rng):
    r = pd.DataFrame(
        rng.normal(0, 0.01, (300, 2)), index=dates[:300], columns=["a", "b"]
    )
    r.iloc[:5, 1] = np.nan
    results = [
        k.realized_vol(r["a"]),
        k.realized_vol(r),
        k.max_drawdown(r["a"]),
        k.summary(r, benchmark=r["a"].rename("bench")),
        k.correlation(r, min_obs=500),  # NaN off-diagonal, nested DataFrame meta
        k.portfolio_returns(r.dropna(), {"a": 0.5, "b": 0.5}),
        k.event_study(r["a"], r["a"] * 0.5, [dates[200]]),
    ]
    for res in results:
        text = json.dumps(k.to_dict(res), allow_nan=False)
        assert "NaN" not in text
    d = k.to_dict(k.realized_vol(r["a"]))
    assert d["start"] == "2024-01-01" and d["method"] == "close-to-close"
    c = k.to_dict(k.correlation(r, min_obs=500))
    assert c["data"]["a"]["b"] is None
    assert c["meta"]["pair_n"]["a"]["b"] == 295


def test_render(dates, rng):
    r = pd.Series(rng.normal(0, 0.01, 100), index=dates[:100], name="x")
    assert "close-to-close" in k.render(k.realized_vol(r))
    text = k.render(k.summary(r))
    assert "ann_vol" in text and "[kairos]" in text and "return_method=log" in text
    both = k.render(k.realized_vol(r.to_frame().assign(y=r * 2)))
    assert both.splitlines()[1].startswith("y: ")


def test_prices_passed_as_returns_raise(dates):
    prices = pd.Series(np.linspace(100, 120, 50), index=dates[:50])
    for fn in (
        k.realized_vol,
        k.sharpe_ratio,
        k.max_drawdown,
        k.value_at_risk,
        k.summary,
    ):
        with pytest.raises(ValueError, match="looks like prices"):
            fn(prices)
    with pytest.raises(ValueError, match="looks like prices"):
        k.beta(prices, prices * 2)
    k.realized_vol(k.returns(prices))  # the suggested fix works
