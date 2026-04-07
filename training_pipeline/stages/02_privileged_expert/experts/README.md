# Specialist Experts

This directory is the phase-02 control plane for multi-expert development.

Rules:

- Every specialist inherits from the shared base config in `shared/base_config.json`.
- Every specialist uses the shared mode vocabulary in `shared/mode_vocabulary.json`.
- Every specialist must keep the same 5D action contract as the validator-facing actor.
- Every specialist has its own folder so it can be tuned, evaluated, and quality-gated independently.
- Phase 03 should only accept a specialist when its evaluation summary passes the category gate in that folder.

Layout:

- `shared/`
  - common base config
  - common mode vocabulary
- `registry.json`
  - map-category to specialist assignment
- `specialists/<map_category>/`
  - `config_override.json`
  - `quality_gate.json`

Workflow:

1. Start from `shared/base_config.json`.
2. Tune only the category override in `specialists/<map_category>/config_override.json`.
3. Evaluate that specialist on the same map category in phase 02.
4. Record the latest `expert_eval_summary.json` path in `registry.json`.
5. Enable `--require-quality-gates` in phase 03 so only accepted specialists can generate data.

Current categories:

- `city_static`
- `city_dynamic`
- `open_static`
- `open_dynamic`
- `mountain_static`
- `mountain_dynamic`
- `village_static`
- `village_dynamic`
- `warehouse_static`
- `forest_static`
