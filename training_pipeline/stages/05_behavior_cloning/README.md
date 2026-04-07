# 05 Behavior Cloning

Goal:

- Pretrain the student on expert data before any DAgger or RL.

Build here:

- supervised action imitation loss
- auxiliary perception losses
- sequence batching for recurrent training
- validation metrics by map type
- live progress logging
- early stopping and best-checkpoint selection

Done when:

- the student can navigate simple maps from expert demonstrations alone
- you can measure failures by type before moving to DAgger

Current implementation notes:

- training reads Phase-3 episode files directly or a weighted sampling manifest
- losses are logged as both `total` and `action-only` for train and validation
- early stopping watches validation total loss
- the best checkpoint is saved even if later epochs regress
