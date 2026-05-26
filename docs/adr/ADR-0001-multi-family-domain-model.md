# ADR-0001: Multi-Family Domain Model and Naming

- Status: Accepted
- Date: 2026-05-25

## Context

The current benchmark stack mixes legacy map-centric wording, `map_type`,
`challenge_type`, and bare `family` language for different concepts. That is workable
for one SAR benchmark, but it is ambiguous once the platform supports multiple
challenge families across the subnet, backend, and website.

The ambiguity shows up in two places:

1. The top-level benchmark grouping and the per-environment grouping use the same word.
2. Legacy transport fields (`challenge_type`, `map_type`, `typeN_*` benchmark groups)
   do not expose the intended domain boundaries.

## Decision

Adopt the following canonical terms across repos.

### `challenge_family`

The top-level benchmark product line.

- Stable ID shape: `cf_*`
- Owns the benchmark rules, scoring policy, emissions policy, and supported
  environment types.
- Initial challenge family IDs:
  - `cf_autopilot`
  - `cf_search_and_rescue`

### `challenge_instance`

A single generated evaluation unit inside a challenge family.

- Stable ID shape: `ci_*`
- Natural key dimensions: `challenge_family`, `environment_type`,
  `benchmark_version`, `seed`
- Represents one concrete benchmarkable instance, not the family definition itself.

### `skill`

The capability a challenge family is intended to measure.

- Stable ID shape: `sk_*`
- Initial skill IDs:
  - `sk_autonomous_navigation`
  - `sk_search_and_rescue`

### `market_vertical`

The deployment market represented by a challenge family.

- Stable ID shape: `mv_*`
- Initial market vertical IDs:
  - `mv_general_autonomy`
  - `mv_public_safety`

### `environment_type`

The concrete environment category evaluated within a challenge family.

- Enum values for the current SAR challenge family:
  - `city`
  - `open`
  - `mountain`
  - `village`
  - `warehouse`
  - `forest`
- `environment_type` replaces older map-centric benchmark-doc wording.

### `family_state`

Lifecycle state for a challenge family definition.

- Enum values:
  - `incubating`
  - `active`
  - `archived`

### `emissions_state`

Lifecycle state for whether a challenge family contributes to emissions.

- Enum values:
  - `incubating`
  - `active`
  - `saturated`
  - `archived`
  - `regression`

## Legacy Compatibility

The current system still exposes several legacy names. They remain valid implementation
details for now, but they map to the canonical model above.

- `challenge_type` is the legacy numeric transport field for `environment_type`.
- `map_type` is a legacy API field name whose value set is the `environment_type` enum.
- Benchmark group names such as `type1_city` are legacy runner aliases that map to
  `environment_type`.

## Consequences

- Benchmark architecture docs should use `challenge_family` for the top-level grouping
  and `environment_type` for city/open/mountain/village/warehouse/forest.
- New abstractions should be challenge-family-aware instead of SAR-only or
  environment-type-specific.
- The machine-readable contract for IDs, enums, and legacy aliases lives in
  `swarm/domain_model/benchmark_domain_model.schema.json`.
