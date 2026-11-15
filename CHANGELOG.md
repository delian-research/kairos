# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Estimate` result type that records value, observation count, date span,
  method and settings.
- Returns: `returns`, `last_n`, `cumulative` (log returns by default).
- Volatility: `realized_vol`, `rolling_vol`, `ewma_vol`, `parkinson_vol`,
  `garman_klass_vol`.
- Beta: `beta` (OLS, optional Newey-West standard error), `rolling_beta`,
  `newey_west_lags`.
- Drawdown: `drawdown_series`, `max_drawdown`.
- Conventions module: `TRADING_DAYS`, `DEFAULT_RETURN_METHOD`, `DEFAULT_DDOF`,
  `annualize_vol`, `annualize_return`.
- Transforms: `excess_returns`, `active_returns`, `resample_returns`,
  `total_return_prices`.
- Performance: `annualized_return` (CAGR), `sharpe_ratio`, `sortino_ratio`,
  `calmar_ratio`, `tracking_error`, `information_ratio`, `capture_ratio`,
  `hit_rate`.
- Tail risk: `value_at_risk` (historical, normal, Cornish-Fisher),
  `expected_shortfall`, `skewness`, `excess_kurtosis`, `worst_periods`.
- Volatility: `rogers_satchell_vol`, `yang_zhang_vol`, `vol_cone`,
  `vol_percentile`.
- Covariance: `correlation`, `ewma_cov`, `ledoit_wolf`, `nearest_psd`.
- `summary`: one-call statistics table with metadata in `df.attrs["kairos"]`.

[Unreleased]: https://github.com/delian-research/kairos/commits/main
