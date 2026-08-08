# Architecture (v0.1)

```
┌─────────────┐     targets (signed leverage)
│  Strategy   │ ──────────────────────────┐
│  dual_ma    │                           │
└─────────────┘                           ▼
┌─────────────┐                     ┌───────────┐     allowed target
│  Data       │  bars (OHLCV)       │ RiskGate  │ ──────────────────┐
│ fixture/REST│ ───────────────────▶│ caps+kill │                   │
└─────────────┘                     └───────────┘                   ▼
                                                              ┌──────────────┐
                                                              │ FuturesBook  │
                                                              │ L/S margin   │
                                                              │ MTM + liq    │
                                                              └──────┬───────┘
                                                                     │
                                                                     ▼
                                                              output/journal
```

- **Strategy** never imports the book for fills.
- **RiskGate** is the only path from target → `apply_target`.
- **FuturesPaperBook** owns wallet, positions, fees, funding hook, liquidation.
- **mode=paper** default; live requires kill_switch off + enable_live.
