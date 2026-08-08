# Data-Stack Evaluation (2026-08-04)

## 1. Where a daily digest sits in the information hierarchy

A scraped-and-curated daily market digest is an **attention-allocation layer**,
not a trading input.

| Dimension | What it provides | What it cannot provide |
|---|---|---|
| Latency | Once daily, post-close synthesis | Intraday timeliness; executable price precision |
| Content | Event awareness, cross-market sentiment, sector-theme momentum | Fundamentals, valuation, order book, flow detail |
| Fundamental analysis | The WSJ-front-page function: what happened, what narrative the market is pricing | Fundamental analysis proper — that requires structured filings and financial data |
| Trading | Post-hoc context; hypothesis generation | Any trigger — trading off it is trading off a newspaper |

The analogy holds: a daily digest serves a value investor the way a WSJ
subscription does — background awareness and lead discovery. No disciplined
strategy converts headlines into orders. Its correct role in a quant stack is a
**catalyst/sentiment corpus** (machine-readable daily snapshots, amenable to
event extraction and theme tracking): it feeds the research layer, never the
execution layer.

## 2. Infrastructure for automated rebalancing: Bloomberg vs. TradingView vs. APIs

Taking US equities as the example: automated position adds / rebalancing have
modest data requirements (minute-level suffices; tick data is unnecessary). The
real constraints are **execution reliability and brokerage access**.

### Bloomberg Terminal (~$27k/yr)

- An institutional workstation for humans — data, news, analytics — not an
  automation substrate
- Programmatic access means B-PIPE, an enterprise feed at enterprise pricing;
  not a retail proposition
- Verdict: oversized and misshaped for a personal quant stack. **Pass**

### TradingView (~$15–60/mo)

- Best-in-class charting and alerting; webhooks can forward alerts to an external
  executor, broker integrations are limited
- Display-grade SLA, not trading-grade; the alert chain
  (TV → webhook → your server → broker) has too many moving parts
- Verdict: acceptable as a **prototype-stage alert trigger**, disqualified as a
  production base layer

### Broker / market-data APIs (the correct answer)

| Layer | US-equities candidates | Notes |
|---|---|---|
| Execution | **IBKR TWS/Gateway API**, Alpaca | IBKR has the widest venue coverage (US/HK/options); Alpaca is commission-free with a clean API and free paper trading |
| Market data | Polygon.io / Alpaca Data / IBKR subscriptions | Minute-level rebalancing runs fine on Alpaca's free IEX feed; full SIP via Polygon starts around ~$29/mo |
| News & fundamentals | In-house digest pipeline + SEC EDGAR (free) | EDGAR is the primary source for US fundamentals |

**The truth about "reliability":** it is a function of hop count and per-hop SLA,
not subscription price. Owning the data and execution hops (an Alpaca/IBKR
monostack) removes one failure domain versus "TradingView alert → custom
executor", and costs three orders of magnitude less than Bloomberg.

### Hard constraints on CN/HK venues (stated plainly)

- A-shares: no sanctioned retail programmatic channel; broker-side QMT/PTrade
  terminals carry capital thresholds and lock to their own clients. Free data
  (akshare/tushare) is adequate; **execution is either QMT behind a capital
  threshold, or semi-automated (signal → human confirms the order)**
- HK: no retail API; IBKR covers HKEX and is the only realistic automation path
- This is why **crypto is the correct first venue for automation** (below)

## 3. Crypto real-time: how to actually guarantee it

Crypto is the only 24/7 venue with free APIs, no account minimums, arbitrary
position sizing, and a full-featured testnet — the natural first target for an
end-to-end automation build.

**Real-time means the venue's native WebSocket feed — not page scraping, not
REST polling.**

| Component | Selection | Network path (this host) |
|---|---|---|
| Live market data | Binance spot WebSocket (trade/kline/depth streams) | WS endpoints unreachable on the direct route → via the local market-data egress gateway |
| REST backfill / reconciliation | Binance public REST mirror (klines etc.) | Direct route, no gateway |
| Historical bulk | Binance public data dumps (daily/monthly klines for backtests) | Direct route |
| Execution (paper phase) | Binance Spot **Testnet** (feature-parity with the live spot API, paper money) | Same path as WS |

Engineering notes (all hard-won common sense):

1. WS connections need heartbeats and automatic reconnect; maintain local
   last-price / order-book state, use REST only for gap backfill
2. Clock sync (NTP) — signed requests are sensitive to clock drift
3. Egress stability beats latency: pin the egress route; no mid-session
   switching during trading windows
4. Define "real-time" acceptance metrics up front: WS downtime, message-latency
   distribution (p50/p99), gap-replay counts — and wire them into standing
   monitoring

## 4. Conclusions

- Daily digest = WSJ-grade reference: valuable (awareness, hypotheses),
  disqualified as a trading input. Retained as the catalyst corpus
- No Bloomberg; TradingView at most as a prototype alerter
- US-equity automation = IBKR or Alpaca monostack; A-share execution stays
  semi-automated absent a sanctioned channel
- Crypto first: Binance WS + public REST mirror + Testnet paper — the P0 data
  foundation
