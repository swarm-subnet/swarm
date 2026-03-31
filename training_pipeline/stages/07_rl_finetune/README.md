# 07 RL Fine-Tune

Goal:

- Improve beyond imitation with benchmark-aligned reward.

Build here:

- recurrent PPO training loop
- actor on deployable observations only
- critic on privileged labels plus deployable observations
- checkpoint evaluation and early stopping

Important:

- this stage comes after behavior cloning and DAgger
- asymmetric critic support usually means custom training code

Done when:

- RL improves score rather than just destabilizing a good imitation policy
