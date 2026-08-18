# GALAHAD Researchkit

Dependency-free v1 artifact compatibility, strict v2 artifact-graph validation,
deterministic SEC CompanyFacts normalization, and financial-statement, DCF, and
trading-comparables calculations. All core paths run offline; data acquisition,
credentials, and MCP transports stay outside this package.

Use `validate_artifact_graph` for new v2 evidence packs. It verifies snapshot,
artifact, and result hashes as distinct values; resolves RFC 6901 pointers;
checks cutoff/entity/lineage/reachability; and replays the installed calculation
methods. `validate_artifact` supports legacy v1 and individual v2 envelopes for
component diagnostics, but does not prove cross-artifact closure.

Run the fixture-backed tests from this directory:

```bash
uv run --with 'pytest>=8.3,<10' python -m pytest -q
```
