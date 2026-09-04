# Third-party notices

This repository uses the following third-party software. All packages are
installed unmodified from public package indexes (pip/PyPI); this
repository vendors no third-party source files. Cached CSV bars under
`galahad-futures/data/cache/` are factual market data retrieved from
venue public endpoints, and `galahad-futures/data/fixtures/` is synthetic
— neither is third-party code.

## NautilusTrader

- Version: `nautilus_trader==1.231.0` (pinned exact; final stable 1.x
  line — see `galahad-futures/docs/nautilus-v2-migration.md`)
- Copyright (c) Nautech Systems Pty Ltd
- License: LGPL-3.0-only — https://www.gnu.org/licenses/lgpl-3.0.html
- Source: https://github.com/nautechsystems/nautilus_trader
- Use: optional, pip-installed dependency of `galahad-futures`, imported
  dynamically at runtime behind the `nautilus` package extra
  (`pip install galahad-futures[nautilus]`). The library is used
  **unmodified** as an event-driven backtest execution backend; no
  NautilusTrader source is copied into, or distributed with, this
  repository.
- As the LGPL requires, users may replace the pinned build with a
  different NautilusTrader version (or a modified build) of their own;
  the dynamic-import boundary keeps that substitution straightforward.
  Note that versions other than the pin are unsupported by this project
  until the pin formally moves (parity-run gated).

## galahad-futures runtime dependencies

| Package | License | Purpose |
|---|---|---|
| pandas | BSD-3-Clause | Bar frames and time-series plumbing across data, decision, and report layers |
| numpy | BSD-3-Clause | Numerical kernels; direct use in `galahad_futures.statistics` (DSR/PBO) |
| pyarrow | Apache-2.0 | Parquet cache tier (offline-readable bar cache); imported lazily |
| PyYAML | MIT | Session and engine configuration loading |
| pytest | MIT | Test runner for the package suite |

## quantkit

Runtime dependencies (declared in `quantkit/pyproject.toml`), all used
unmodified:

| Package | License | Purpose |
|---|---|---|
| numpy | BSD-3-Clause | Array numerics |
| pandas | BSD-3-Clause | Panels, books, and reporting frames |
| scipy | BSD-3-Clause | Statistics and optimization primitives |
| matplotlib | Matplotlib license (BSD-compatible, PSF-based) | Research plotting |

Optional extras:

| Package | License | Purpose |
|---|---|---|
| vectorbt (pinned `==1.0.0`) | Apache-2.0 with Commons Clause ("fair-code") | Vectorized parameter sweeps (`quantkit[sweep]`). Commons Clause restricts resale as a standalone product: internal research use only, per `quantkit/pyproject.toml` |
| lightgbm | MIT | Gradient-boosted models (`quantkit[ml]`) |
| xgboost | Apache-2.0 | Gradient-boosted models (`quantkit[ml]`) |
| scikit-learn | BSD-3-Clause | ML utilities (`quantkit[ml]`) |

The full research environment (data providers such as akshare, yfinance,
ccxt, baostock; notebook and reporting tooling) is enumerated in
`quantkit/requirements.txt`; those packages are likewise installed
unmodified from PyPI and carry their own licenses.

## quant-desk

`quant-desk` ships scripts and configuration (no package manifest). Its
scripts build on the in-repo `quantkit` and additionally import:

| Package | License | Purpose |
|---|---|---|
| baostock | BSD License (per PyPI metadata) | China A-share market-data retrieval for the `ashare` configs |

plus numpy/pandas/PyYAML as covered above.

## researchkit

`researchkit` declares **no third-party runtime dependencies**
(standard library only); the `test` extra pulls pytest (MIT, covered
above). The bundled JSON Schema documents under `researchkit/schemas/`
are original work referencing the JSON Schema draft 2020-12 dialect.
