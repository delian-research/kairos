# Kairos

Kairos is a small Python package for price analytics: returns, volatility,
beta and drawdown. Every result says how it was computed, so the conventions
behind a number are never ambiguous.

## Installation

```bash
pip install kairos-analytics        # or: uv add kairos-analytics
```

The distribution is named `kairos-analytics` on PyPI, but the package is
imported as `kairos`:

```python
import kairos as k
```

Kairos supports Python 3.10+ and depends only on `numpy` and `pandas`. It does
no I/O: you pass in prices or returns as pandas objects indexed by date.

## Conventions

| Setting | Default | Override |
|---|---|---|
| Return definition | **log**, `ln(p_t / p_{t-1})` | `method="simple"` |
| Variance estimator | sample, `ddof=1` | `ddof=0` |
| Annualization | off; when on, `× sqrt(252)` | `annualize=True`, `periods=` |
| Window | all non-missing data | `window=n` (raises if fewer than `n` exist) |
| Alignment | inner-join on index, drop NaN rows | — |

Too few observations raise `ValueError`; kairos never silently returns NaN or
a shorter window than you asked for.

## Example

```python
import kairos as k

r = k.returns(prices)                  # log returns; n prices -> n - 1 returns
r = k.last_n(r, 252)                   # exactly the latest 252 returns

vol = k.realized_vol(r, annualize=True)
print(vol)
# 0.137384 (close-to-close, n=252, 2025-11-13 to 2026-11-13; ddof=1, annualized=True, periods=252)

vol.value, vol.n, vol.start, vol.params["ddof"]
```

Every scalar function returns an `Estimate` with `value`, `n`, `start`, `end`,
`method` and `params`. `Estimate.as_dict()` flattens it for DataFrames or JSON.
Passing a DataFrame (one column per instrument) to `realized_vol` or
`max_drawdown` returns a `dict[str, Estimate]`.

## API

| Function | What it computes |
|---|---|
| `returns(prices, method)` | Log or simple returns from a Series or wide DataFrame |
| `last_n(x, n)` | Latest `n` non-missing rows; raises if fewer exist |
| `cumulative(r, method)` | Wealth index (growth of 1) |
| `realized_vol(r, window, ddof, annualize)` | Close-to-close standard deviation |
| `rolling_vol(r, window)` | Rolling close-to-close volatility |
| `ewma_vol(r, halflife)` | RiskMetrics-style zero-mean EWMA volatility |
| `parkinson_vol(high, low)` | High-low range estimator |
| `garman_klass_vol(open, high, low, close)` | OHLC range estimator |
| `beta(asset, benchmark, hac=False)` | OLS beta with alpha, R², SE and t; optional Newey-West SE |
| `rolling_beta(asset, benchmark, window)` | Rolling OLS beta |
| `drawdown_series(r)` / `max_drawdown(r)` | Drawdown path; worst drawdown with peak, trough and recovery dates |

Range estimators (Parkinson, Garman-Klass) measure intraday variation only and
ignore overnight gaps, so they tend to read lower than close-to-close
volatility for instruments that gap.

## Development

```bash
uv sync
uv run ruff check . && uv run mypy src && uv run pytest
```

See [RELEASING.md](RELEASING.md) for publishing to PyPI.
