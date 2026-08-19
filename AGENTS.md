# AGENTS.md — GALAHAD

- Positioning: public quantitative-trading research notebook + reusable toolkit. Everything committed must be reproducible by a stranger: no personal machine paths, no internal infrastructure references, no credentials or endpoints tied to a specific host.
- Language: English, financial-engineering register.
- `docs/` changes before code does; roadmap and validation doctrine live in `docs/roadmap.md`.
- Code changes require tests (`pytest` per component); keep the paper-first discipline — no live-order path enabled by default.
- Secrets and environment specifics belong in env vars (`QUANT_DESK_PROXY`, API keys via `.env`, never committed).
- Preserve contributor identity: agent-authored commits use the agent's GitHub-linked Git author identity rather than the human operator or only a co-author trailer; human-authored commits retain the human author.
- Apache-2.0 License (see `LICENSE` and `NOTICE`).
