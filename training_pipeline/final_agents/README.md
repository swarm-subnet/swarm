# Final Agents

This directory is for deployable outputs only.

Rules:

- Consume only `{"depth", "state"}` at runtime.
- Do not depend on privileged labels.
- Do not depend on training-only wrappers.
- Keep the exported controller small enough to pass normal Swarm submission
  checks.
