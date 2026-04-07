# 02 Privileged Expert

Goal:

- Build the first reliable controller with "godmode" information.
- Build it progressively, not one-shot against the full validator distribution.

Build here:

- stage-02 custom curricula for easier-than-benchmark task bands
- simple privileged teachers for early landing curriculum stages
- search-center controller
- local avoidance controller
- platform intercept controller
- landing controller
- controller arbitration or mode logic

Inputs:

- `info["privileged"]`
- deploy-time observation if useful

Outputs:

- expert action
- optional expert mode label

Progressive order:

- `open_r1_5_static`
- `open_r5_10_static`
- `city_r5_10_static`
- `forest_r5_10_static`
- `warehouse_r5_10_static`
- `village_r10_20_static`
- `mountain_r10_20_static`
- moving-platform contact curricula after static landing is reliable

Current implementation notes:

- `build_privileged_expert.py` now supports `--task-source custom_radius`
- `progressive_curriculum.py` defines early easy stages
- `experts/` now scaffolds specialist teachers by map category, all inheriting from one shared base config and one shared mode vocabulary
- the validator manifest remains the benchmark source of truth, but stage 02 may use easier custom curricula to develop the teacher incrementally
- `experts/registry.json` is the assignment point from `map_category` to `teacher_id`, config override, and quality gate
- specialist quality should be judged per category, not only by one aggregate success number

Done when:

- the expert solves static maps reliably
- the expert tracks moving platforms well enough to produce good labels
- weak categories can be iterated independently without breaking strong ones
