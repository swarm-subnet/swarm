# 05 Behavior Cloning

Goal:

- Pretrain the student on expert data before any DAgger or RL.

Build here:

- supervised action imitation loss
- auxiliary perception losses
- sequence batching for recurrent training
- validation metrics by map type

Done when:

- the student can navigate simple maps from expert demonstrations alone
- you can measure failures by type before moving to DAgger
