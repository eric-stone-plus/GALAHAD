# Trading system architecture

How the public repositories compose into one trading-research system,
from raw market data to a reviewed, evidence-bundled decision. Every
layer is fail-closed and paper-first: **no component enables a live
order path by default.**

```
┌─────────────────────────── research layer ───────────────────────────┐
│ GALAHAD/galahad-crypto   backtests, execution-RL studies             │
│ GALAHAD/quantkit         factors, validation gates (walk-forward,    │
│                          DSR/PBO, block bootstrap), sizing           │
│ GALAHAD/researchkit      offline artifact-graph validation           │
└──────────────┬───────────────────────────────────────────────────────┘
               │ decision artifacts (signals, schedules, sizing)
┌──────────────▼─────────────────────────── execution layer ───────────┐
│ GALAHAD/galahad-futures  USDT-M paper/testnet substrate              │
│                          (NautilusTrader; margin book, funding,      │
│                          drawdown force-flat, kill switch)           │
│ GALAHAD/galahad-security US-equities paper substrate (Alpaca paper)  │
└──────────────┬───────────────────────────────────────────────────────┘
               │ session summary + journal (schema-validated)
┌──────────────▼──────────────────── orchestration & review layer ─────┐
│ STAMMTISCH                 event-sourced pipelines; gates are code;  │
│                            adapters own all product contact;         │
│                            offline-verifiable evidence bundles       │
│ QUINTE                     A2A v1.0 review host (LLM seats);         │
│                            reviews a closed-schema Brief, never      │
│                            originates direction or repairs numbers   │
│ HIGHBALL                   rules-plane carrier: decision verdicts    │
│                            and residual blockers                     │
└──────────────────────────────────────────────────────────────────────┘
```

## Layer contracts

1. **Research → execution.** Research tools emit decision artifacts
   (JSON, schema-validated). They never touch a venue. The backtest
   workbench (`galahad-crypto`) is deterministic and offline by
   default; pinned public data makes results reproducible by a
   stranger.
2. **Execution → orchestration.** A paper/testnet session produces a
   summary and a fill journal under a documented JSON contract
   (`--json` stdout is machine-parsable; human logs go to stderr).
   Reconciliation is part of the session: expected vs. actual
   positions must match or the session fails closed
   (`position_mismatch`).
3. **Orchestration → review.** STAMMTISCH runs pipelines as event
   logs (events are the authority, manifests are projections). A
   review stage sends a doctrine Brief to QUINTE over A2A and records
   content-addressed receipts. Gates are quantified thresholds over
   typed artifact fields — never model judgment. A model verdict is
   an opinion attached to the evidence, not a gate.
4. **Evidence.** Every completed run exports an offline-verifiable
   bundle (digest-bound manifests, receipts, artifacts). Verdicts
   from the rules plane (HIGHBALL) and reviewer opinions are carried
   inside the bundle, so a third party can re-verify the chain
   without any service access.

## Operating principles

- **Paper-first.** Testnet/paper venues only; live enablement would
  require explicit config and is not shipped.
- **Fail closed.** Corrupt state, digest drift, missing metrics, or a
  non-terminal review task halt the run with a durable record; there
  is no best-effort fallback.
- **Numerical conclusions stand on validated data alone.** LLM review
  exists to find confounders, contradictions, and invalidations; one
  material unresolved blocker means abstain/block.
- **Credentials never live in repos.** Venue and API keys are
  environment-only (`.env`, mode 600, gitignored; `token_env`
  indirection in pipeline configs).

## Where to read next

- `docs/exchange-integration.md` — venue/adapter matrix and testnet
  availability for the execution layer.
- `galahad-crypto/README.md` — research workbench quickstart and
  recorded (including negative) results.
- `galahad-futures/docs/` — paper engine contracts and evidence
  records.
- STAMMTISCH `docs/architecture.md` — the normative pipeline/evidence
  spec; STAMMTISCH `docs/fullstack-quickstart.md` — end-to-end setup
  of all four repositories.
