# LangAlpha → STAMMTISCH / GALAHAD evaluation

Object of this verdict: **[`ginlix-ai/LangAlpha`](https://github.com/ginlix-ai/LangAlpha)** at commit [`15107b8`](https://github.com/ginlix-ai/LangAlpha/tree/15107b869a146b42f72b65bb126a7eb718da40fb), fetched 2026-08-14 from the live public repo. The pinned source set is its [README](https://github.com/ginlix-ai/LangAlpha/blob/15107b869a146b42f72b65bb126a7eb718da40fb/README.md), [`skills/` tree](https://github.com/ginlix-ai/LangAlpha/tree/15107b869a146b42f72b65bb126a7eb718da40fb/skills), [LICENSE](https://github.com/ginlix-ai/LangAlpha/blob/15107b869a146b42f72b65bb126a7eb718da40fb/LICENSE), [`dcf-model/SKILL.md`](https://github.com/ginlix-ai/LangAlpha/blob/15107b869a146b42f72b65bb126a7eb718da40fb/skills/dcf-model/SKILL.md), and [`THIRD-PARTY-NOTICES.md`](https://github.com/ginlix-ai/LangAlpha/blob/15107b869a146b42f72b65bb126a7eb718da40fb/skills/THIRD-PARTY-NOTICES.md).
Steering: professional finance skills for **GALAHAD**, not a vibe-investing app inside STAMMTISCH.

## Disambiguation

| Repo | What it is | Role here |
|---|---|---|
| **ginlix-ai/LangAlpha** | “Claude Code for Financial Market” / “vibe investing agent harness”. Apache-2.0. Skills + LangGraph/PTC + FastAPI/web + MCP. | **The merge candidate.** |
| **[`Chen-zexi/LangAlpha`](https://github.com/Chen-zexi/LangAlpha)** | “Multi-Agent Financial Research workflow”. Jupyter-heavy, **no SPDX license**, topics `langraph`/`trading-strategies`. Homepage also `langalpha.ai`. | **Name clash only.** Do not clone it by accident. |

---

## 1. Three-layer inventory (from live source)

### 1.1 Financial skills (`skills/`)

README claims **23 pre-built financial research skills** (Agent Skills Spec). Its five-category table counts 23 items, while the pinned `skills/` tree contains 33 top-level skill directories because it also holds harness, workbench, and general-purpose tooling. Directories present:

**Valuation & modeling (research-grade, Anthropic-derived):**

`dcf-model`, `comps-analysis`, `3-statements`, `model-update`, `check-model`

**Equity research:**

`initiating-coverage` (30–50 page report), `earnings-preview`, `earnings-analysis`, `thesis-tracker`

**Market intelligence:**

`morning-note`, `catalyst-calendar`, `sector-overview`, `competitive-analysis`, `idea-generation`, plus `market-watch`, `x-api`

**Document generation:**

`pdf`, `docx`, `pptx`, `xlsx`, `html-report`

**Operations / harness (not research doctrine):**

`automation`, `secretary`, `user-profile`, `onboarding`, `run-workflow`, `self-improve`, `check-deck`

**Workbench-integrated and general-purpose support:**

`chart-annotation`, `inline-widget`, `interactive-dashboard`, `ui-design`, `web-scraping`

`skills/THIRD-PARTY-NOTICES.md` states the valuation/research skills are **derived from `anthropics/financial-services-plugins` (Apache-2.0)**, modified to swap Bloomberg/FactSet/Daloopa for LangAlpha’s FMP-backed MCP + native tools.

A representative skill (`dcf-model/SKILL.md`) is a **prompt workflow**, not a library: it tells an agent to call `fundamentals` / `macro` MCP, generate the workbook with `openpyxl`, and run `.agents/skills/xlsx/scripts/recalc.py`. It does not itself provide a standalone offline DCF implementation.

### 1.2 Agent harness

README self-description: **“A vibe investing agent harness.”**

- **LangGraph ReAct** main agent + `Task()` swarm (parallel async subagents, isolated contexts, checkpoint resume).
- **Programmatic Tool Calling (PTC):** the model writes Python; a **Daytona** (or Docker) sandbox runs it against MCP wrappers instead of dumping raw series into the LLM context.
- **Flash vs PTC** dual mode; multi-provider LLM layer (OpenAI / Anthropic / Gemini / DeepSeek / Kimi / GLM / …) with OAuth + BYOK.
- Middleware: skill loading, plan mode, multimodal, compaction, live steering, provenance.
- Surfaces: **React 19 web workbench**, `libs/ptc-cli` TUI, Slack/Discord/Feishu/Telegram (hosted).
- Automations: cron + **price-triggered** live WebSocket (ginlix-data; hosted-only in beta).

This layer **owns prompts, model endpoints, tool loops, and a GUI/daemon**.

### 1.3 Runtime infra

From README architecture + Getting Started:

| Piece | Role |
|---|---|
| FastAPI (`server.py`, `src/server/`) | Threads, workspaces, market data, OAuth, automations, skills |
| PostgreSQL (dual pool) | App data + LangGraph checkpointer |
| Redis | SSE replay (150k events), cache, steering queue |
| Daytona / Docker sandbox | Isolated code execution |
| FMP API | Fundamentals, statements, macro, options |
| ginlix-data | Real-time WS (Polygon/Massive) |
| Yahoo / yfinance | Keyless fallback |
| SEC EDGAR, Tavily/Serper/Exa, Firecrawl | Filings / search / crawl |
| `docker-compose.yml`, `deploy/` | Hosted-service shape |

`make up` starts Postgres, Redis, backend, and frontend. The repository is a **self-hostable full-stack research workbench with web and CLI/TUI clients**, plus a separate hosted offering; it is not a thin standalone CLI.

**License:** Apache License 2.0, Copyright 2025 Ginlix AI (`LICENSE`), with critical path-level exceptions. Apache-covered files can coexist in an MIT-licensed repository, but copied or modified files do not thereby become MIT-only: preserve the Apache license and copyright notices, mark modifications, and retain the Anthropic lineage recorded in `skills/THIRD-PARTY-NOTICES.md`.

[`skills/docx/LICENSE.txt`](https://github.com/ginlix-ai/LangAlpha/blob/15107b869a146b42f72b65bb126a7eb718da40fb/skills/docx/LICENSE.txt),
[`skills/pdf/LICENSE.txt`](https://github.com/ginlix-ai/LangAlpha/blob/15107b869a146b42f72b65bb126a7eb718da40fb/skills/pdf/LICENSE.txt),
[`skills/pptx/LICENSE.txt`](https://github.com/ginlix-ai/LangAlpha/blob/15107b869a146b42f72b65bb126a7eb718da40fb/skills/pptx/LICENSE.txt), and
[`skills/xlsx/LICENSE.txt`](https://github.com/ginlix-ai/LangAlpha/blob/15107b869a146b42f72b65bb126a7eb718da40fb/skills/xlsx/LICENSE.txt)
each expressly prohibit extracting or retaining copies outside Anthropic
services, reproduction, derivative works, distribution, sublicensing, and
transfer. The root Apache license must not be used to infer permission for
those four directories. **Do not copy, vendor, adapt, translate, or derive
GALAHAD implementations from them.** Replace their functionality with
independently authored workflows based on public file-format specifications and
permissively licensed libraries.

**Root tree captured:** `.dockerignore`, `.env.example`, `.gitattributes`, `.github/`, `.gitignore`, `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `DESIGN.md`, `Dockerfile.sandbox`, `LICENSE`, `Makefile`, `README.md`, `agent_config.yaml`, `alembic.ini`, `config.yaml`, `deploy/`, `docker-compose.yml`, `docs/`, `libs/`, `mcp_servers/`, `migrations/`, `pyproject.toml`, `scripts/`, `server.py`, `skills/`, `src/`, `tests/`, `uv.lock`, `web/`, `workflows/`.

---

## 2. Mapping onto STAMMTISCH and GALAHAD

### 2.1 STAMMTISCH (normative)

STAMMTISCH [`docs/architecture.md` §1](https://github.com/eric-stone-plus/STAMMTISCH/blob/8a5bda58988ac89ce8456e8c17739a2df3706b20/docs/architecture.md#1-positioning):

> **STAMMTISCH is an orchestrator of *products*, not an agent runtime. It does not call model endpoints, does not own prompts, and does not reimplement review logic.**

Non-goals (same section):

> **No GUI, no daemon, no hosted service. A CLI and a state root.**
>
> **No agent framework (no prompt chains, no tool-calling loops of its own).**
>
> **No live capital / production deployment actions.**

STAMMTISCH already treats **GALAHAD as the doctrine/quant protagonist** (`doctrine/examples/galahad/`, `src/adapters/galahad.rs` → `galahad-futures` paper session, live capital refused). Thin platform, thick products (P7).

| LangAlpha layer | Fits STAMMTISCH? |
|---|---|
| Whole harness (LangGraph, PTC, web, SSE, swarm) | **No.** It *is* an agent runtime + GUI + daemon. |
| FastAPI / Postgres / Redis / Daytona | **No.** Host infra is explicitly “not part of the composition” (§1). |
| Price-triggered automations | **No as core.** They require an always-on event service; that is hosted infrastructure outside STAMMTISCH's composition. The source does not show that these automations execute trades. |
| Skill *texts* as doctrine/briefs consumed by GALAHAD | **Maybe**, as **product content**, not as STAMMTISCH core. |

A whole-repo merge would make STAMMTISCH the thing §1 says it is not.

### 2.2 GALAHAD (constraint)

GALAHAD [`AGENTS.md`](../AGENTS.md):

> public quantitative-trading research notebook + reusable toolkit. Everything committed must be **reproducible by a stranger**: no personal machine paths, no internal infrastructure references, no credentials or endpoints tied to a specific host.
>
> keep the **paper-first** discipline — **no live-order path enabled by default**.
>
> MIT License.

What GALAHAD already is:

- `galahad-futures`: paper engine, walk-forward, risk, fixtures (`scripts/run_paper.py`).
- `quantkit`: OHLCV, factors, gates, backtest — **code**, not an LLM skill.
- `quant-desk`: select → trade → evaluate → review → gates.
- Roadmap L0–L3 (`docs/roadmap.md`): digest → **sector deep-dives / fundamentals** → factor signals → paper then small live.

LangAlpha skills sit naturally at **roadmap L1 (Research)** — DCF, coverage, earnings, sector overview — where GALAHAD's roadmap calls for sector deep-dives and fundamentals. They do **not** replace L2 (DSR/PBO, walk-forward) or L3 (paper/live).

| Skill cluster | GALAHAD fit |
|---|---|
| DCF, 3-statement, comps, model-update, check-model | **Partial fit.** Real IB procedure. Must be rebound to fixture/akshare/quantkit data; cannot require FMP/Daytona for stranger repro. |
| Initiating coverage, earnings preview/analysis, thesis tracker, morning note, sector overview, competitive analysis | **Partial fit.** Matches L1 “sector deep-dives / broker research”. Outputs should be **gated artifacts** (JSON + cited numbers), not chat. |
| Idea generation, catalyst calendar, market-watch, X research | **Weak as a mandatory baseline.** Their live and non-deterministic sources conflict with offline stranger-reproducibility; they may only be optional, environment-gated inputs. |
| secretary, automation, user-profile, onboarding, chart-annotation, widgets, ui-design, web-scraping | **Reject.** Harness/UI/live channel. |
| Price automations / live WS | **Reject from the skill pack.** They are hosted event/daemon infrastructure, not professional-finance research doctrine; the cited source does not establish that they submit orders. |
| `docx`, `pdf`, `pptx`, `xlsx` | **Reject categorically.** Their directory-level Anthropic licenses prohibit retained copies, reproduction, derivatives, and distribution. |

DCF skill as shipped is **not** a GALAHAD module: it is instructions for an agent that already has MCP + sandbox. Dropping the folder into GALAHAD unchanged would add Markdown instructions, not a fixture-backed executable DCF component that GALAHAD can validate with `pytest`.

---

## 3. Three-part verdict

### (a) Whole-repo merge into STAMMTISCH — **NO**

LangAlpha is a full agent runtime that owns model calls, prompts, a React GUI, FastAPI, Postgres, Redis, and sandboxes. STAMMTISCH §1 forbids exactly that class of merge. Swallowing the harness would also import host infra STAMMTISCH refuses to narrate, and would turn the public CLI orchestrator into a “vibe investing” app. **Do not merge the repo.**

### (b) Extract-skills-only (or adapter-wrap) — **CONDITIONAL YES**

- **Yes, extract** the Anthropic-derived *research procedure* skills (valuation + equity research + morning/sector/competitive) into **GALAHAD**, not into `stammtisch` core.
- **Conditions:**
  1. For permissively licensed paths only, keep the Apache-2.0 license and copyright/change notices plus `THIRD-PARTY-NOTICES.md` / Anthropic attribution; track copied or modified files as Apache-covered rather than relabeling them MIT-only. Exclude `docx`, `pdf`, `pptx`, and `xlsx` completely under their directory-level restrictions.
  2. Rewrite tool bindings: FMP/Daytona/MCP → `quantkit` + fixtures + optional env-gated vendors. A stranger `pytest` must pass offline.
  3. Promote checklists into **quantified gates** (doctrine `gates.json` style), not “the model feels done.”
  4. English human-facing skill instructions and artifacts; source-language raw
     fields require documented English mappings. No ginlix/Daytona host paths.
  5. **Do not** vendor `src/ptc_agent`, `web/`, `server.py`, compose stack, or MCP servers.
- **Adapter-wrap of the live LangAlpha process:** only if someone later wants a *separate* product CLI that emits a receipt STAMMTISCH can hash. That is a new product, not a STAMMTISCH merge.

### (c) Suitability as GALAHAD professional finance skills — **PARTIAL**

The skill *content* is the right kind of professional finance (IB DCF, comps, initiating coverage, earnings) and is the only layer worth taking. It directly addresses GALAHAD's planned L1 sector and fundamentals research layer while leaving its L2/L3 signal and execution layers separate.

It is **not** a drop-in skill pack: execution assumes LangAlpha's agent and sandbox plus external data tools. Yahoo is a keyless fallback, but the DCF workflow calls LangAlpha MCP tools and the README recommends FMP for stronger coverage; several other skills are UI, secretary, or live-trigger oriented. Suitability = **partial**: lift the research procedures, rebind them to reproducible paper-first data, and reject the harness.

Licenses: LangAlpha Apache-2.0 (Ginlix, 2025) + Anthropic plugin lineage; STAMMTISCH MIT; GALAHAD MIT. Extraction from the permissively licensed paths is feasible as a mixed-license distribution if the Apache conditions above are met and all path-level exceptions are excluded. Offline reproducibility remains the harder engineering constraint.

---

## 4. Operator action

1. Do **not** add LangAlpha as a STAMMTISCH subtree or docker-compose dependency.
2. If GALAHAD wants these skills next: prefer the permissively licensed
   Anthropic upstream, create a selected first-party set under
   `.agents/skills/`, rewrite MCP/Daytona bindings, add fixture-backed tests,
   and wire outputs into typed evidence/gate schemas. Do not copy LangAlpha's
   four restricted Office skill directories.
3. Ignore `Chen-zexi/LangAlpha` unless someone pastes the wrong clone URL.

---

## Source excerpts (captured 2026-08-14 at `15107b8`)

README: “A vibe investing agent harness”; 23 skills; LangGraph + PTC + Daytona; `make up` starts PostgreSQL, Redis, backend, frontend; data tiers ginlix-data / FMP / Yahoo; some skills adapted from `anthropics/financial-services-plugins`.

LICENSE: Apache License 2.0, Copyright 2025 Ginlix AI.

`dcf-model/SKILL.md` front matter: derived from anthropics/financial-services-plugins (Apache-2.0); tools are fundamentals/macro MCP + `get_company_overview`.
