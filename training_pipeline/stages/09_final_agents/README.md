# 09 Final Agents

Goal:

- Promote only deployable artifacts into `final_agents`.

What belongs here:

- cleaned hardcoded baselines
- distilled student policies ready for packaging
- final deploy-time wrappers

What does not belong here:

- privileged experts
- dataset builders
- DAgger collectors
- RL training code

Promotion rule:

- if it requires `info["privileged"]`, it is not a final agent
- if it requires arbitrary training helper Python modules at packaging time, it
  is not yet a valid final submission
