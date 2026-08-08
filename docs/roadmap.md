# Roadmap: From Sector Deep-Dives to Automated Execution (2026-08-04)

## 1. Architecture: research → selection → trading

Four decoupled layers with explicit I/O contracts. News lives only at the top;
the execution layer never sees a headline.

```
L0  Attention    Daily digest pipeline (scrape → machine-readable snapshots)
                 ↓ catalyst/theme extraction — which sectors keep igniting
L1  Research     Sector deep-dives: industry pages, broker research archive
                 Fundamentals: akshare/tushare (A-shares), SEC EDGAR/yfinance (US),
                 venue APIs (crypto)
                 ↓ sector universe + fundamentals scoring
L2  Signals      Factor screening + timing: local factor library, statistically
                 rigorous evaluation (purged k-fold with embargo, walk-forward,
                 DSR/PBO, block bootstrap)
                 ↓ target positions / add-rules
                 (e.g., price X% below MA AND fundamentals score ≥ Y → add Z)
L3  Execution    paper: exchange testnet / broker paper → small live capital
                 Risk controls up front: per-order and daily limits, portfolio
                 drawdown breaker, kill switch (manual, one action)
```

The sector deep-dive loop, concretely: a theme recurs in the digest → L1 builds
the sector universe with financials and valuation data → industry structure
research on top → L2 runs factor scoring and backtests over that universe →
a watchlist falls out. A human approval gate sits between the watchlist and L3;
rule-based pass-through is only considered at P3.

## 2. Staged plan

| Phase | Scope | Acceptance criteria |
|---|---|---|
| **P0 Data foundation** | Venue WS feed + REST backfill + persistence (klines/ticker → parquet); equity dailies reuse existing pipelines | 7 consecutive unattended days; self-healing reconnects; message-latency p99 within budget |
| **P0b Strategy doctrine** | Academic spine for strategy families + gates (see `strategy_foundations.md`) | Dual-MA/TSMOM as null; funding + liquidation first-class; DSR/PBO policy written |
| **P1 Signals & backtests** | One deliberately naive strategy to start — dual-MA / trend-filtered rules; full statistical evaluation | DSR > 0; PBO within bounds (policy: flag if >0.5); no out-of-sample collapse |
| **P2 Paper trading** | Run P1 on perp/spot paper; full telemetry (fills, slippage, disconnects, funding) | ≥4 weeks; deviation between fills and backtest assumptions is explainable |
| **P3 Small live capital** | Real money at a lose-it-all-tolerant size; kill switch validated first | Manual acceptance checklist, line by line; limits hard-coded |

US equities run as a parallel track after P2: Alpaca paper → IBKR. A-share
execution stays semi-automated (signal → human) unless a sanctioned channel
becomes available.

## 3. Risk doctrine (precedes any strategy code)

1. Every execution path defaults to OFF; enabling one is an explicit human act
2. Strategies emit **target positions**, never orders; the execution layer is
   the only component allowed to place orders, and it is hard-capped
3. Every strategy ships with a pre-written invalidation condition — the
   circumstances under which it must stop — and stops when it trips
4. Live position limits are hard-coded in execution-layer configuration;
   changing them is a code change with an audit trail
5. News/digest layers never trigger trades (no L0→L3 edge)

## 4. Private implementation tracks (not in this repo)

Production code stays private. Complementary local tracks include:

- **Spot / broker paper** — venue testnet and broker paper APIs (P2).
- **Futures / USDT-M paper** — a separate private substrate for signed
  positions, leverage, maintenance margin, liquidation, and a risk gate that
  is the only path from strategy **targets** to fills. Public GALAHAD does not
  ship that ledger; the research constraint still holds: paper ≥4 weeks before
  any live capital, and strategies never emit orders.

These tracks are allowed to coexist with OpenAlice (agent workspace) and
TradingAgents (LLM multi-agent research): those systems may inform human
judgment; they do not replace deterministic execution and margin accounting.

## 5. Open questions (next research round)

- P0 persistence format and directory conventions
- 24h disconnect profile of a WS long-connection over the local egress
  gateway — this sets the feasibility ceiling for the crypto real-time link
- Invalidation conditions for the trend-filtered DCA strategy
- Alpaca vs. IBKR: account opening and funding feasibility (user-side action)
- Funding-rate realism and multi-symbol margin portfolio for the futures paper track
