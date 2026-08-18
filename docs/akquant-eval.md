# akquant evaluation (2026-08-18)

**Verdict: WATCH.** Good Rust-core engineering, but it is not a data leg and
the current A-share track has no execution need it fills. Revisit when the
A-share track needs a high-throughput backtest/execution substrate or a
sanctioned execution channel appears.

## What it is

[akquant](https://github.com/akfamily/akquant) (akfamily) is a quantitative
backtest/trading framework: a Rust core (~31k lines, pyo3 0.29, maturin
wheels on PyPI — no Rust toolchain needed by users) wrapped by a Python
interface (~43k lines, 131 files). Event-pipeline engine (data → strategy →
execution → statistics processors), `rust_decimal` money math, risk/margin/
settlement managers with liquidation audits, zero-copy NumPy bar ingestion,
103 TA-Lib indicators, a Polars factor-expression engine (Alpha101-style),
an ML walk-forward validation framework (train/test/rolling windows for
sklearn/torch), and multi-process grid search.

## Verified facts (GitHub API, 2026-08-18)

| Fact | Value |
|---|---|
| License | MIT (API, Cargo.toml, LICENSE all agree) |
| Stars / forks / open issues | 2,044 / 267 / 0 |
| Created | 2026-01-30 (~6.5 months old) |
| Last push / release | 2026-08-18 (same day) / v0.3.41 on PyPI (2026-08-16); Cargo at 0.3.42 |
| Cadence | rapid churn: 0.3.14 → 0.3.41 within one month; API renames still landing (test_api_rename_* files) |
| Data model | consumes pandas DataFrames; akshare is the canonical feeder (`fetch_akshare_symbol`, qfq/hfq) but is only a docs-extra dependency, not required |

## Assessment

- **Architecture**: sound — real Rust core, event-driven pipeline, decimal
  money, locking discipline documented in code comments, 183 test files.
- **A-share data via akshare**: yes as a feeder, but akquant is a consumer,
  not a provider — it owns no data leg.
- **Statistics**: equity/cash/margin curves and liquidation audits only; no
  DSR/PBO/purged-CV machinery, so no overlap with `quantkit.validation`.
- **Maintenance**: MIT, same house as akshare, extremely active, but young
  and fast-moving; pre-1.0 churn is real.

## Interaction with quantkit

- **No data-leg role**: the A-share data path stays akshare/tushare →
  `quantkit.data` → `quantkit.cn_market` masks. akquant neither replaces nor
  requires any of it; it would sit downstream as a backtest/execution engine
  over frames already validated by quantkit (one-way boundary).
- **Doctrine conflict**: akquant's Strategy API places orders (`buy` /
  `close_position` in callbacks); GALAHAD doctrine (roadmap, risk #2) is
  strategies emit target positions, execution layer places orders. Adoption
  would need an adapter that feeds only our targets into akquant's simulated
  execution.
- **A-share execution** is semi-automated (signal → human) until a sanctioned
  channel exists, so no near-term execution need.

## Recommendation

**Watch** — do not adopt now, do not skip. Reasons to adopt later: real Rust
quality, MIT, active maintenance, no competition with our validation stack.
Reasons not to adopt now: it is not a data leg (the question it was evaluated
for); strategy-places-orders API conflicts with our doctrine (adapter
required); pre-1.0 churn would cost rework; low-frequency A-share research
has no throughput bottleneck that Rust buys. Watch triggers: stable 1.x API,
a sanctioned A-share execution channel, or a measured backtest-speed
bottleneck.
