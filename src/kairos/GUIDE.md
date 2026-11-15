# kairos guide

kairos computes risk and performance statistics from price and return data.
It does no I/O: load data elsewhere (for example with agora), pass pandas
objects in, and get results that say how they were computed.

```python
import kairos as k

k.catalog()  # table of every function: area, signature, summary
help(k.sharpe_ratio)  # full documentation of one function
```

## Rules that prevent wrong numbers

1. **Prices are not returns.** Almost every function takes *returns*. Convert
   first: `r = k.returns(prices)`. Passing prices raises an error.
2. **Say which kind of return you have.** `k.returns` gives **log** returns by
   default (`method="simple"` for simple). Functions that compound
   (`annualized_return`, `max_drawdown`, `calmar_ratio`, `capture_ratio`,
   `cumulative`, `resample_returns`, `portfolio_returns`, `summary`) take the
   same `method` argument, and it must match your data. Their default is
   also `"log"`, so the default pipeline is consistent.
3. **Nothing is annualized unless you ask.** Volatility and ratios are per
   period unless `annualize=True` (uses `periods=252`; pass `periods=52` for
   weekly data, `12` for monthly). Exceptions: `annualized_return` is always
   annual, and `summary` annualizes and says so in its metadata.
4. **Risk-free rates are per period.** For 4% a year on daily data pass
   `rf=0.04 / 252`, or a Series of per-period rates.
5. **VaR and expected shortfall are positive losses.** `0.021` means a 2.1%
   loss.
6. **n returns need n + 1 prices.** `k.last_n(r, 252)` gives exactly the
   latest 252 returns and raises if there are fewer.
7. **Errors mean the answer is undefined or the data is too short.** kairos
   raises `ValueError` instead of returning NaN. Report the reason; do not
   retry with made-up inputs.
8. **Price returns are not total returns.** Returns from raw closes exclude
   dividends. Use `k.total_return_prices(close, dividends)` first when
   income matters (long windows, high-yield names, bond funds).

## Reading results

* Single numbers come back as an `Estimate`: `.value`, `.n`, `.start`,
  `.end`, `.method`, `.params`. `float(est)` gives the value and `str(est)`
  a one-line description.
* Tables come back as DataFrames with the same metadata in
  `df.attrs["kairos"]`.
* `k.render(result)` formats anything as text with its provenance.
  `k.to_dict(result)` gives JSON-safe data (dates as ISO strings, NaN as
  null).

When you report a number, state the window, the observation count and the
conventions (log or simple, annualized or not). They are all in the result.

## Which function answers which question

| Question | Call |
|---|---|
| Full picture of one or more securities | `k.summary(r, benchmark=b, rf=rf)` |
| How volatile is it? | `k.realized_vol(r, annualize=True)`; OHLC data: `k.yang_zhang_vol(o, h, l, c)` |
| Is volatility high right now? | `k.vol_percentile(r, 21)`, `k.vol_cone(r)` |
| Recent volatility, weighting new data more | `k.ewma_vol(r, halflife=11)` |
| What did it return per year? | `k.annualized_return(r)` |
| Risk-adjusted return | `k.sharpe_ratio(r, rf, annualize=True)`, `k.sortino_ratio`, `k.calmar_ratio` |
| Is that Sharpe statistically meaningful? | `k.probabilistic_sharpe(r)`, `k.bootstrap_ci(r, k.sharpe_ratio)` |
| Worst loss from a peak | `k.max_drawdown(r)`; the path: `k.drawdown_series(r)` |
| How bad is a bad day? | `k.value_at_risk(r, 0.95)`, `k.expected_shortfall(r, 0.95)`, `k.worst_periods(r)` |
| Sensitivity to the market | `k.beta(r, b, hac=True)`; over time: `k.rolling_beta(r, b, 63)` |
| Beta of an illiquid or foreign stock | `k.dimson_beta(r, b)` |
| Beta in falling markets | `k.downside_beta(r, b)` |
| Exposure to several factors | `k.factor_regression(r, factors_df, hac=True)` |
| Performance against a benchmark | `k.tracking_error`, `k.information_ratio`, `k.capture_ratio`, `k.active_returns` |
| Reaction to events (earnings, index adds) | `k.event_study(r, b, dates)` |
| How do assets move together? | `k.correlation(r_df)`; for portfolio math: `k.ledoit_wolf(r_df)` |
| Portfolio return series | `k.portfolio_returns(r_df, weights)` |
| Where does portfolio risk come from? | `k.risk_contributions(weights, cov)` |
| Suggested weights | `k.inverse_vol_weights`, `k.min_variance_weights`, `k.risk_parity_weights` |
| Weekly or monthly returns from daily | `k.resample_returns(r, "W-FRI")`, `"ME"` |

## Recipes

### One security

```python
r = k.last_n(k.returns(close), 252)  # latest 252 log returns
print(k.render(k.summary(r, rf=0.04 / 252)))
print(k.max_drawdown(r))  # includes peak, trough, recovery dates
```

### Security against a benchmark

```python
r = k.returns(prices_df)  # columns: asset and benchmark closes
a, b = r["NVDA"], r["SPY"]
print(k.beta(a, b, hac=True))  # value, alpha, se, t, r2 in params
print(k.render(k.summary(a, benchmark=b)))
```

### Portfolio

```python
r = k.returns(prices_df)  # one column per holding
w = {"SPY": 0.6, "TLT": 0.4}
port = k.portfolio_returns(r, w)  # constant weights; rebalance=None for buy-and-hold
print(k.render(k.summary(port, benchmark=r["SPY"])))

cov = k.ledoit_wolf(r[list(w)])  # per-period covariance, safe to invert
print(k.render(k.risk_contributions(w, cov, annualize=True)))
print(k.risk_parity_weights(cov))
```

`portfolio_returns` uses the rows where every weighted asset has data, so a
holding with a short history shortens the whole sample. Weights are used as
given: a sum below 1 leaves the rest in cash at zero return.

### Event study

```python
es = k.event_study(r["AAPL"], r["SPY"], earnings_dates, window=(-1, 5))
print(k.render(es))  # mean abnormal return, CAR and t by day
es.attrs["kairos"]["events_skipped"]  # events too near the edge of the data
```

## Choosing a covariance estimator

* `ledoit_wolf`: the default for portfolio math; always invertible.
* `ewma_cov`: recent data weighted more; use for current risk.
* `correlation` with `min_obs`: for reading relationships when histories
  differ. Pairwise matrices can be invalid for portfolio math; pass them
  through `nearest_psd` first.

## Pitfalls

* A Sharpe ratio from a few months of data is mostly noise. Check
  `probabilistic_sharpe` or a bootstrap interval before drawing conclusions.
* Beta, correlation and volatility change over time. Prefer a stated window
  (`last_n`) and look at the rolling versions before generalizing.
* Parkinson, Garman-Klass and Rogers-Satchell ignore overnight gaps and read
  low for stocks that gap; Yang-Zhang includes them.
* Compare like with like: align windows before comparing two securities
  (`summary` on a DataFrame evaluates each column on its own data and
  records each span in the metadata).
* kairos has no calendar. Windows are counted in rows, so missing dates in
  the input silently stretch a window in calendar time.
