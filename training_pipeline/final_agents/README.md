# Final Agents

This folder is for agents that are intended to be runnable under the real
deployment contract.

Rules:

- Only use the deploy-time observation contract: `{"depth", "state"}`.
- No privileged simulator labels.
- No training-only wrappers.
- Keep the entrypoint importable from [`../drone_agent.py`](../drone_agent.py)
  during repo development.
- For final packaged submission, remember that the top-level
  [`../drone_agent.py`](../drone_agent.py) remains the authoritative entrypoint.
  Any final learned export must still work from that top-level file plus
  allowed model artifacts.

Current contents:

- `drone_agent.py`: cleaned hardcoded baseline.
