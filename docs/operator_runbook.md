# Operator Runbook

Internal guide for the team running the Swarm backend and validators. It covers the knobs that change network behavior: family rollout, the registry copies, validator trust, version gating, the screening switch, and the epoch freeze. Miner- and validator-facing setup lives in [miner.md](miner.md) and [validator.md](validator.md).

---

## Family Rollout Controls

The backend decides which challenge families each validator may evaluate. Four env vars drive it (`swarm-backend/app/rollout.py`); none of them appear in `.env.example`, so an unset production `.env` runs the defaults.

| Env var | Values | Default | Effect |
|---------|--------|---------|--------|
| `SWARM_ROLLOUT_MODE` | `single_family`, `staged`, `multi_family` | `multi_family` (anything else normalizes to it) | Selects the rollout mode |
| `SWARM_ROLLOUT_DEFAULT_FAMILY_ID` | any known family id | `cf_autopilot` (`DEFAULT_CHALLENGE_FAMILY_ID`; unknown ids fall back too) | The family used when a validator can't be trusted with the full set |
| `SWARM_ENABLED_CHALLENGE_FAMILIES` | comma-separated family ids | unset = all families in `CHALLENGE_FAMILY_IDS` | Restricts the enabled set. Unknown ids are dropped; if none survive, only the default family stays. The default family is always included; it is inserted at position 0 only if it is not already in the requested list. In `single_family` mode the set is forced to just the default family |
| `SWARM_ENFORCE_VALIDATOR_CONTRACT` | `true` / anything else | `false` (only the literal `true` enables) | In `multi_family` mode, reject validators that don't present the contract |

### The validator contract

`VALIDATOR_CONTRACT_VERSION = 'family_sync.v1'` is hardcoded identically in the backend (`app/rollout.py`) and the validator (`swarm/validator/backend_api.py`). The validator sends it on every signed request as `X-Swarm-Validator-Contract`, alongside `X-Validator-Hotkey` / `-Signature` / `-Nonce` / `-Timestamp` and `X-Code-Version`.

### Compatibility matrix (`get_validator_rollout_compatibility`)

| Mode | Contract sent | Compatible | `safe_family_ids` | Reason |
|------|--------------|------------|-------------------|--------|
| `single_family` | any | yes | default family only | `single_family_rollout` |
| any | `family_sync.v1` | yes | all enabled families | `validator_contract_accepted` |
| `staged` | missing / other | yes | default family only | `staged_rollout_default_family_only` |
| `multi_family` | missing, enforce off | yes | all enabled families | `validator_contract_optional` |
| `multi_family` | missing, enforce on | **no** (`upgrade_required=True`) | empty | `validator_contract_upgrade_required` |

`POST /validators/tasks/authorize` denies any family outside the validator's `safe_family_ids`: `authorized=False`, `reason_code='upgrade_required'` (when upgrade is required) or `'rollout_restricted'`, `requeue_policy='permanent'`.

Two things to know when operating this:

- The full rollout payload (`{mode, default_family_id, enabled_family_ids, required_validator_contract, enforce_validator_contract}`) is attached to task-authorize / next-task responses and to the public `GET /families/metadata` as `payload['rollout']`. Validators persist it (plus their own `validator_compatibility`) into `swarm/state/runtime_state.json`, so you can inspect what a validator last saw.
- The enabled set gates **validator task authorization only**. Miner submission ingestion (chain scanner) validates family existence and visibility against the registry, not the rollout enabled set: `is_rollout_enabled_family` exists in `rollout.py` but has no callers.

---

## Family Registry: Three Copies, One Source of Truth

The registry that defines the families, their states, policies, and emission allocations lives in three places:

| Repo | Path | Role |
|------|------|------|
| swarm | `swarm/domain_model/benchmark_domain_model.schema.json` | **Source of truth** (schema_version `2026-05-25`) |
| swarm-backend | `app/family_registry.json` | Byte-identical copy, loaded by `app/domain_model.py` |
| Swarm-Website | `src/family_registry.json` | Byte-identical copy |

**Sync is manual.** After editing the schema in the swarm repo, run:

```bash
python3 swarm/scripts/sync_family_registry.py
```

The script reads the swarm-repo schema and overwrites both copies. It assumes the three repos are sibling directories under one workspace root. There is no CI hook: the only automated guard is the backend test `test_registry_copies_are_in_sync_across_repos`, and it **skips** when the sibling checkouts are absent, so divergence is only caught when backend tests run in a full three-repo workspace.

Divergence matters: the backend derives `CHALLENGE_FAMILY_IDS`, `FAMILY_STATES`, and `EMISSIONS_STATES` from **its** copy at import time, DB `challenge_family` rows are seeded and repaired from that copy, and the chain scanner rejects any submission naming a family the backend copy doesn't know.

### Current registry state

Five families, all `family_state='active'` and `emissions_state='active'`:

| Family | `emission_allocation` |
|--------|----------------------|
| `cf_autopilot` | 0.10 |
| `cf_search_and_rescue` | 0.10 |
| `cf_swarm_autopilot` | 0.10 |
| `cf_swarm_sar` | 0.10 |
| `cf_interceptor` | 0.10 |

Allocations sum to 0.50: the unallocated half of the pool burns during the rollout and is raised manually as the tasks prove out.

State enums: `family_state` ∈ {incubating, active, archived}; `emissions_state` ∈ {incubating, active, saturated, archived, regression}; `visibility` ∈ {public, private}.

### Per-family runtime controls

Persisted per-family state lives in the DB table `challenge_family` (`family_state`, `emissions_state`, `emission_allocation`, `solve_threshold`, `solved_at`, `solved_score`) and is operated through:

- `GET /admin/families`: lists persisted states plus derived allocation shares.
- `POST /admin/families/{family_id}`: updates `family_state` / `emissions_state` / `emission_allocation` / `solve_threshold` (admin session required, at least one field, 400 otherwise). Setting `family_state='archived'` implicitly archives emissions unless an explicit `emissions_state` is given. `emission_allocation` and `solve_threshold` must be in [0.0, 1.0].

Allocation semantics are **absolute**: `allocation_share = emission_allocation × status_multiplier`, never normalized. All base weights are 1.0 except archived (0.0). If included shares sum past 1.0 a warning logs, miner weights scale down pro-rata, and the burn share drops to 0. A solved family keeps paying for `SOLVED_PAYOUT_DURATION` = 7 days after `solved_at`, then `archive_expired_solved_families` auto-archives its emissions and the family's slice burns from that point on.

---

## Validator Trust

Two independent layers, checked in this order on every validator request.

### 1. Signature + stake (`app/auth.py`)

Every validator request is signed by the hotkey over `timestamp:nonce:METHOD:path:body_sha256` (timestamp skew limit 60 s) and must carry on-chain stake ≥ `MIN_VALIDATOR_STAKE` (env, default **4096**) on `METAGRAPH_NETWORK` (default `finney`), netuid `METAGRAPH_NETUID` (default `124`). DB-cached stake counts as fresh for `DB_STAKE_FRESHNESS_SECONDS` (default 1800).

### 2. Coldkey whitelist: `TRUSTED_VALIDATOR_COLDKEYS`

Comma-separated coldkey whitelist, checked by `require_trusted_validator` on the 8 main evaluation endpoints in `api_validators.py` (task authorize, next-task, seed-scores, epoch publish, and friends).

| Whitelist state | Behavior |
|-----------------|----------|
| Empty / unset | **Fail-closed**: every evaluation endpoint returns 403 and heartbeats tell validators to stop — nothing is evaluated until the whitelist is configured |
| Configured | **Fail-closed**: coldkey resolved from the DB (`validatoronchain`) then the metagraph; unresolved or unlisted coldkeys get 403 with "Contact the team to be added" |

`require_strict_trusted_validator` (private artifact bytes) applies the same fail-closed rule.

> **Required before anything scores.** The whitelist gates all evaluation, so a fresh deploy must set `TRUSTED_VALIDATOR_COLDKEYS` before validators can take tasks. The private families (`cf_search_and_rescue`, `cf_swarm_sar`) additionally serve artifacts only through the strict gate and need those validators on `>= 5.0.0`. Set the whitelist and roll out 5.0.0 validators **before** relying on evaluation.

---

## Version Gating & Cutover

Two version headers travel on every validator request, gated separately.

### Code version: `X-Code-Version`

Validators send `swarm.__version__` (currently `5.0.0`). The backend dependency `require_current_code_version` rejects with **HTTP 426** `validator_code_version_below_minimum` when the header is missing or below `MIN_VALIDATOR_CODE_VERSION` (env, default `5.0.0`). Private-track endpoints use `require_private_code_version` against `PRIVATE_MIN_VALIDATOR_CODE_VERSION` (env, default `5.0.0`). To cut old validators off after a release: bump these env vars and restart the backend. No code change needed.

### Benchmark version: `X-Benchmark-Version`

Validators send `BENCHMARK_VERSION` (the first 3 components of `swarm.__version__`). The backend fetches the official version from GitHub raw `swarm/__init__.py` on branch `SWARM_VERSION_REF` (env, default `main`), overridable wholesale via `SWARM_VERSION_URL`, cached 900 s. A submission counts as coming from an old validator only when its reported version parses **below** the official one: a *missing* header is accepted, not dropped. Old-validator seed scores are silently dropped (`recorded=0`).

When the official version changes, the version-transition job (checked every 5 minutes) expires all pending models to `VERSION_EXPIRED` and queues every champion for re-evaluation. This means **merging a version bump to `main` (or whatever `SWARM_VERSION_REF` points at) is the cutover trigger**: the backend picks it up in up to about 20 minutes (the 15-minute cache TTL plus the next 5-minute transition-job tick), without a deploy.

---

## The Screening Switch

`SCREENING_ENABLED = False` is a hardcoded module **constant** in `benchmark_logic.py`, not an env var. Flipping it means a code change plus a backend redeploy.

With screening off (the current and default behavior):

- Fresh submissions enter as `PENDING_BENCHMARK` and the benchmark phase covers seed indices [0, 1100) in one pass (`BENCHMARK_SEEDS_PER_RUN`, env default 1100).
- Screening-phase authorization always returns `(False, 'screening disabled')`.
- A stray `PENDING_SCREENING` head is healed in place to `PENDING_BENCHMARK` (audit event `screening_disabled_promotion`), so the queue head can never be pinned by a phase that doesn't run.

Flipping it to `True` gates new models on a screening pass first: seeds [0, 300) (`SCREENING_SEEDS_PER_RUN`, env default 300), pass bar = champion score + screening improvement floor, then the benchmark phase continues over [300, 1100).

**The flip is self-healing.** `apply_screening_mode_fixups` runs at every backend startup (lifespan, under `LOCK_BENCHMARK_SCORES`):

| Direction | Fixup |
|-----------|-------|
| Screening off | All `PENDING_SCREENING` models moved to `PENDING_BENCHMARK` |
| Screening on | `PENDING_BENCHMARK` models with `screening_score IS NULL` moved back to `PENDING_SCREENING` |

Both directions cancel stale-phase tasks/batches, keep already-recorded seed scores, and reconcile batches for both phases (audit event `screening_mode_fixup`). So the operational procedure is simply: change the constant, deploy, restart: the DB converges on its own.

---

## Epochs & Freeze

- Epoch anchor: **2026-03-30 16:00:00 UTC** (a Monday); duration 7 × 86400 s. `epoch = floor((now − anchor) / duration) + 1`, derived purely from wall clock (no state to manage). The same anchor/duration constants exist validator-side.
- `EPOCH_FREEZE_SECONDS = 5400` (1.5 h, hardcoded): `is_in_freeze_window()` is true in the last 5400 s of an epoch. Its only production caller is the **chain scanner**, which skips scanning new chain commitments during the freeze. Commitments made in the window are picked up after rollover. The same constant exists in `swarm/constants.py` but nothing in the validator consumes it.
- The epoch-transition job runs every 5 minutes; on rollover it sets all `PENDING_SCREENING`/`PENDING_BENCHMARK` models submitted before the new epoch to `EPOCH_EXPIRED` and queues every current champion for re-evaluation on the fresh seeds.
- Seeds: each validator generates its own 1100 seeds per epoch per family from OS entropy (`random.SystemRandom`), stored under `state/epoch_seeds/`, and publishes them to the backend only **after** the epoch ends (`POST /validators/epoch/publish`, trusted-validator gated).

---

## Environment Variables by Service

### Backend: required (`swarm-backend/.env.example`)

| Var | Example / default |
|-----|-------------------|
| `DATABASE_URL` | postgres (in-network in production compose, bound 127.0.0.1:5433) |
| `WANDB_API_KEY` | none |
| `ADMIN_API_KEY` | fail-closed for admin auth |
| `MODEL_FILES_PATH` | `/app/data/models` |
| `SWARM_VIDEOS_DIR` | `/app/data/videos` |
| `EPOCH_STATE_FILE` | `/app/data/last_processed_epoch.txt` |
| `LOG_LEVEL` | `INFO` |

The production docker-compose additionally sets `PYTHONPATH=/app` and loads `../../.env`; the backend is bound to 127.0.0.1:8000.

### Backend: rollout & trust (NOT in `.env.example`, add to the live `.env` explicitly)

| Var | Default when unset | Meaning |
|-----|--------------------|---------|
| `SWARM_ROLLOUT_MODE` | `multi_family` | Rollout mode |
| `SWARM_ROLLOUT_DEFAULT_FAMILY_ID` | `cf_autopilot` | Rollout default family |
| `SWARM_ENABLED_CHALLENGE_FAMILIES` | all registry families | Enabled family set |
| `SWARM_ENFORCE_VALIDATOR_CONTRACT` | `false` | Reject contract-less validators in multi_family |
| `TRUSTED_VALIDATOR_COLDKEYS` | unset → nothing evaluates | Coldkey whitelist (fail-closed; required before any scoring) |

### Backend: version gates

| Var | Default |
|-----|---------|
| `MIN_VALIDATOR_CODE_VERSION` | `5.0.0` |
| `PRIVATE_MIN_VALIDATOR_CODE_VERSION` | `5.0.0` |
| `STAKE_WEIGHTED_FROM_VERSION` | `5.0.0` |
| `SWARM_VERSION_REF` | `main` |
| `SWARM_VERSION_URL` | unset (overrides the GitHub raw URL wholesale) |

### Backend: operational knobs

| Var | Default | Var | Default |
|-----|---------|-----|---------|
| `MIN_VALIDATOR_STAKE` | 4096 | `BATCH_SIZE` | 50 |
| `METAGRAPH_NETWORK` | `finney` | `BATCH_LEASE_TTL_SECONDS` | 120 |
| `METAGRAPH_NETUID` | 124 | `REAPER_INTERVAL_SECONDS` | 30 |
| `DB_STAKE_FRESHNESS_SECONDS` | 1800 | `MAX_BATCH_ATTEMPTS` | 3 |
| `SCREENING_SEEDS_PER_RUN` | 300 | `MODEL_STUCK_SCREENING_SECONDS` | 3600 |
| `BENCHMARK_SEEDS_PER_RUN` | 1100 | `MODEL_STUCK_BENCHMARK_SECONDS` | 7200 |
| `PENDING_UPLOAD_TIMEOUT_SECONDS` | 21600 | | |
| `SCANNER_MIN_REGISTRATION_BLOCK` | 0 (off) | `HEARTBEAT_STALL_SECONDS` | 1800 |
| `PRIVATE_MODEL_FILES_PATH` | `/app/data/private_models` | `ACTIVE_HEARTBEAT_SECONDS` | 120 |
| `VERSION_STATE_FILE` | `/app/data/last_processed_version.txt` | `NEW_FLOW_RECENCY_SECONDS` | 3600 |
| `PUBLIC_RATE_LIMIT` | 60 per `PUBLIC_RATE_WINDOW`=60 s | `NEXT_TASK_LONG_POLL_SECONDS` | 25 |
| `ADMIN_SESSION_TTL_SECONDS` | 43200 | `SSE_MAX_STREAM_LIFETIME_SEC` | 21600 |

### Validator (swarm repo)

| Var | Default | Meaning |
|-----|---------|---------|
| `SWARM_BACKEND_API_URL` | required (no code default; the `swarm` CLI defaults it to the public backend API) | Backend URL |
| `WANDB_API_KEY` | optional | wandb logging |
| `VALIDATOR_NAME` | `validator-{uid}` | Display name |
| `SWARM_WEIGHT_SETTER_POLL_SEC` | 60 | Weight-setter poll interval |
| `SWARM_WEIGHT_SETTER_RETRY_SEC` | 300 | Retry after a failed weight set |
| `SWARM_WEIGHT_REFRESH_SEC` | 300 | Weight map refresh |

`swarm/.env.example` lists exactly the first three.

Advanced tuning (all optional, `swarm/config/runtime.py`): `SWARM_DOCKER_THREAD_CAPS` (default false), `SWARM_TORCH_NUM_THREADS`, `SWARM_TORCH_INTEROP_THREADS`, `SWARM_DOCKER_WORKER_CPUS_OVERRIDE`, `SWARM_DOCKER_WORKER_MEMORY_OVERRIDE`, `SWARM_DOCKER_WORKER_CPUSETS`, `SWARM_DOCKER_WORKER_CPUSET_CPUS_{i}`, `SWARM_HOST_WORKER_MEMORY_MB`, `SWARM_HOST_WORKER_CPUSETS`, `SWARM_BATCH_TIMEOUT_MULT` (1.0), `SWARM_BATCH_TIMEOUT_HARD_CAP_SEC` (0), `SWARM_BATCH_TIMEOUT_EXTEND_ON_PROGRESS` (false), `SWARM_BATCH_TIMEOUT_EXTEND_SEC` (30), `SWARM_BATCH_TIMEOUT_MAX_TOTAL_SEC` (0), `SWARM_LOG_RPC_TRACE` (false), `SWARM_TERRAIN_CACHE_DIR`; burn validator: `BURN_VALIDATOR_HEARTBEAT_SEC` (5), `BURN_VALIDATOR_STALL_TIMEOUT_SEC` (900).

---

## Quick Reference: Where to Act

| I want to… | Do this |
|------------|---------|
| Restrict validators to one family | `SWARM_ROLLOUT_MODE=single_family` (+ `SWARM_ROLLOUT_DEFAULT_FAMILY_ID`), restart backend |
| Roll out a new family gradually | `SWARM_ROLLOUT_MODE=staged`: contract-less validators get the default family only |
| Force validators to upgrade for multi-family | `SWARM_ENFORCE_VALIDATOR_CONTRACT=true` in `multi_family` mode |
| Add/change a family or its policy | Edit `swarm/swarm/domain_model/benchmark_domain_model.schema.json`, run `python3 swarm/scripts/sync_family_registry.py`, commit all three copies |
| Change a family's state or emission share live | `POST /admin/families/{family_id}` |
| Allow validators to evaluate | Set `TRUSTED_VALIDATOR_COLDKEYS` (fail-closed; empty means nothing scores) |
| Cut off old validator code | Bump `MIN_VALIDATOR_CODE_VERSION` / `PRIVATE_MIN_VALIDATOR_CODE_VERSION`, restart backend |
| Trigger a benchmark-version cutover | Merge the `swarm/__init__.py` version bump to the `SWARM_VERSION_REF` branch: backend follows within up to about 20 minutes |
| Turn screening on/off | Flip `SCREENING_ENABLED` in `benchmark_logic.py`, deploy, restart: startup fixups migrate the queue |
