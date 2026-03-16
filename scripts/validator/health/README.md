# Validator Health Scripts

This folder contains the validator health checks used to verify whether a validator is still setting weights on chain.

Files:
- `check_validator_health.py`: prints a 10-row table with one health check per hour for the last 10 hours by default.
- `check_current_epoch_weights.py`: returns `true` or `false` for the current in-progress epoch.

Recommended commands:

```bash
python3 scripts/validator/health/check_validator_health.py
```

```bash
python3 scripts/validator/health/check_validator_health.py --single-check --network finney
```

```bash
python3 scripts/validator/health/check_validator_health.py --last-epochs 10
```

```bash
python3 scripts/validator/health/check_current_epoch_weights.py \
  --netuid 124 \
  --hotkey 5FF6pxRem43f7wCisfXevqYVURZtxxnC4kYTx4dnNAWqi9vg \
  --network finney
```

Defaults:
- `netuid=124`
- `hotkey=5FF6pxRem43f7wCisfXevqYVURZtxxnC4kYTx4dnNAWqi9vg`
- `network=archive`

Modes:
- Default: last 10 hours table, one health check per hour.
- `--single-check`: one-line `OK` / `ERROR` result for the latest completed epoch, or a chosen `--block`.
- `--last-epochs N`: prints `true` if all of the last `N` completed epochs had a weight update, otherwise `false`.

Python usage:

```python
from scripts.validator.health.check_validator_health import were_last_epochs_healthy

print(were_last_epochs_healthy(10))
```

Notes:
- Use `archive` for historical checks and last-epochs checks. Public `finney` endpoints discard older state.
- `check_current_epoch_weights.py` remains parameterized, but you can keep using the same fixed validator values above.
