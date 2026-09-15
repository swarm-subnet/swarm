# Simulation engine switches

How the validator's simulation is steered from this repository: the per-family options, the environment variables and the constants that turn engine features on, which call carries them into the engine, and the steps a new map follows to get the full picture. The engine itself is the `swarm-bullet3` wheel, our fork of PyBullet; every switch it offers is listed in the fork's `docs/swarm_api.md`, and this page maps our side onto it.

The wheel version is pinned in `requirements.txt`. A family can only ask for a switch the pinned wheel carries; the environment checks with `hasattr(pybullet, "<constant>")` and raises a clear error otherwise. Tests behind such a check skip on an older wheel, which means CI proves nothing about them until the pin moves.

## Family options

Class attributes on `ChallengeFamilyRuntime` in `swarm/challenge_families/base.py`. Every registered family is on the defaults today, so every family renders and flies exactly as before the options existed. Turning one on changes that family's observations or flights, so it needs a version bump.

| Option | Default | What it turns on | Engine switch it uses |
|---|---|---|---|
| `render_backend` | `"tiny"` | `"raycast"` sends the drone depth cameras through the Embree ray caster, single and batched call alike; the office family stays on TinyRenderer because it observes colour; on the ray caster the 35 m visual hide of the distance cull is skipped (a ray caster does not get slower with more in view), the 50 m collision cull stays | `ER_SWARM_RAYCAST` |
| `seeded_sun` | `False` | One sun per seed from a real arc (`swarm/core/daylight.py`): the seed draws the sun's height and heading, the height sets its colour and strength; `lightDirection`, `lightColor`, `lightAmbientCoeff` and `lightDiffuseCoeff` on the observation and on-demand colour renders come from it (a recorded video frame takes only the sky) | the upstream light arguments |
| `night_share` | `0.0` | With `seeded_sun`, this share of seeds gets a moon instead: 10 to 60 degrees up, cool blue-grey, dimmer than any sun, with its own dark sky gradient | `skyHorizonColor`, `skyZenithColor` |
| `sky_from_sun` | `False` | With `seeded_sun` and a daytime sun, the renderer computes the sky from that sun; a night seed keeps the moon's sky | `ER_SWARM_SKY_SUN` |
| `sky_clouds` | `False` | With `seeded_sun` and `sky_from_sun` on a daytime sun, the map seed seeds a cloud layer into that sky | `skyCloudSeed` |
| `sky_colors(task)` | returns `None` | A hook a family overrides to return `(horizon, zenith)` RGB triples for a fixed or seed-dependent two-colour sky; wins over the moon's sky | `skyHorizonColor`, `skyZenithColor` |
| `daylight` | `False` | With `render_backend = "raycast"`, `seeded_sun` and `sky_from_sun`: every colour frame of a daytime seed uses the renderer's daylight model with the picture flags and `shadow=1`, lit by the seed's sun at several times the sky, with the exposure opened up for a low sun as a camera does (`swarm/core/daylight.py`, `daylight_render_kwargs`), under a photograph from the worlds package sky pack picked by the seed among the skies whose sun stands about as high, turned so its sun shares the seed's heading (`swarm/core/sky_pack.py`); the photo is loaded into the engine once per world; a night seed keeps the moon | `ER_SWARM_DAYLIGHT` with `ER_SWARM_SHADOW_MAP`, `ER_SWARM_MOVER_SHADOW`, `ER_EDGE_ANTIALIAS`, `ER_ALPHA_CUTOUT`, `ER_TEXTURE_FILTER`, `ER_SPECULAR_GLINT`, `ER_SWARM_LINEAR_LIGHT`; `exposure`, `hazeDistance`, `shadowCoreRadius`, `skyTextureId`, `skyYaw` |
| `physics_mode` | `"pyb"` | A `gym_pybullet_drones` `Physics` value; `"pyb_gnd_drag_dw"` adds the URDF's air drag, ground effect and downwash; the ground effect height is read from the altitude ray, not world z, so it works on a map whose floor is not at zero; with wind on, the library's still-air drag is skipped because the wind force already applies it on the relative air | none, drone model only |

Measured on the open map at 50 Hz: the aerodynamic terms cost +0.05 ms per substep and shorten a 10 s cruise at 1.5 m/s by 2.6 %.

## Per-map options

| Table | Where | What it does |
|---|---|---|
| `WIND_BY_MAP[(family_id, challenge_type)] = {"max_mps": ..., "turbulence": ..., "gusts": ...}` | `swarm/constants.py`, ships empty | Seeded wind for that map: a steady vector at 40 to 67 % of the cap, Dryden low-altitude turbulence, a few gust bumps peaking at 1.5 x the mean, all drawn from the map seed and clamped to the cap; applied as rotor drag on the relative air every physics substep. `task_gen` copies the entry into the `MapTask` fields `wind_max_mps`, `wind_turbulence`, `wind_gusts`, which default to zero so old blobs unpack unchanged. A 4 m/s cap drifts a hovering drone 0.8 m in 10 s and costs 72 to 92 us per control step |
| `GEOM_CONCAVE_BVH_CACHE` on the concave shapes | mountain terrain (`swarm/core/mountain_generator_parts/terrain.py`) and the office pieces (`swarm/core/maps/office/builder.py`), through `getattr` so an older wheel ignores it | The engine saves each collision tree to the cache folder and loads it for the same triangles at the same scale; about 1.7 s less per mountain seed from the second miner on |

## Environment variables

| Variable | Default | Read by | What it does |
|---|---|---|---|
| `SWARM_RENDER_BACKEND` | unset | the environment | Overrides every family's `render_backend` for an experiment: `raycast` or `tiny` |
| `SWARM_RENDER_THREADS` | 2 (in the engine) | the engine, once per process | Render threads; the bytes do not depend on it. Workers are sized at `DOCKER_WORKER_CPUS` |
| `SWARM_BATCH_DEPTH` | `1` | the environment | `0` renders multi-drone depth one camera at a time instead of through `getDepthImagesBatch` |
| `SWARM_BVH_CACHE_DIR` | set by the seed manager | the engine | The seed manager points it at `state/bvh_cache/epoch_<n>` when it loads or generates an epoch's seeds, creates that folder and deletes the older epochs' folders; the host workers inherit it. Unset it and every shape and tree builds fresh |
| `SWARM_TERRAIN_CACHE_DIR` | a per-uid folder under `state` | the mountain generator | Where the generated terrain meshes are kept |
| `SWARM_DOCKER_PREWARM` | on | the benchmark and validator workers | `0` restores the serial container start; on, the next seed's container starts while the current seed flies, at the lowest CPU weight and a half-core quota until it is adopted. The batch plan puts one seed per container (`_batch_indices` in `swarm/benchmark/engine_parts/seeds.py`), so one clean container per seed stays the rule |

## Constants that steer the engine

All in `swarm/constants.py`.

| Group | Constants |
|---|---|
| Physics | `SIM_DT` (50 Hz), `SOLVER_ITERATIONS` (4), `SOLVER_MIN_ISLAND_SIZE` (128), `SPEED_LIMIT` |
| Cameras | `CAMERA_FOV_BASE` and `CAMERA_FOV_VARIANCE`, `DEPTH_FAR`, `SAR_DEPTH_RES`, `SAR_RGB_RES` |
| Distance cull | `CULL_VISUAL_RADIUS` (35 m), `CULL_PHYSICS_RADIUS` (50 m), `CULL_INTERVAL_STEPS`, `CULL_MIN_AABB_SPAN`, `CULL_MIN_FACES`, `CULL_MIN_TOTAL_FACES` |
| Old seeded light | `LIGHT_RANDOMIZATION_ENABLED`: the light direction every family uses today, a point on a circle that half the time sits below the ground |
| Seeded sun and moon | `SUN_SEED_OFFSET`, `SUN_LATITUDE_DEG`, `SUN_DECLINATION_DEG`, `SUN_MIN_ELEVATION_DEG`, `SUN_DIFFUSE_MAX`, `SUN_EXTINCTION`, `SUN_AMBIENT_RANGE`, `MOON_ELEVATION_RANGE_DEG`, `MOON_COLOR`, `MOON_DIFFUSE_RANGE`, `MOON_AMBIENT_RANGE` |
| Daylight | `DAYLIGHT_SUN_DIFFUSE_MAX`, `DAYLIGHT_AMBIENT`, `DAYLIGHT_EXPOSURE`, `DAYLIGHT_EXPOSURE_GAIN_MAX`, `DAYLIGHT_SKY_DUSK_SHARE`, `DAYLIGHT_HAZE_M`, `DAYLIGHT_SHADOW_CORE_M`, `DAYLIGHT_SKY_SEED_OFFSET`, `DAYLIGHT_SKY_ELEVATION_TOLERANCE_DEG` |
| Wind | `WIND_BY_MAP`, `WIND_SEED_OFFSET`, `WIND_MEAN_FRACTION`, `WIND_TURB_SIGMA_XY`, `WIND_TURB_SIGMA_Z`, `WIND_TURB_TAU_XY_SEC`, `WIND_TURB_TAU_Z_SEC`, `WIND_GUST_PEAK`, `WIND_GUST_DURATION_SEC` |
| Workers | `DOCKER_WORKER_CPUS` |

## What the environment sets on the engine

Two things every world gets whatever the family, from `swarm/utils/env_factory.py` and the environment's reset in `swarm/core/moving_drone.py`:

- `setPhysicsEngineParameter(numSolverIterations=SOLVER_ITERATIONS, minimumSolverIslandSize=SOLVER_MIN_ISLAND_SIZE)`: four solver iterations instead of PyBullet's fifty, and islands merged up to 128 bodies, both for speed.
- Every static map body goes into collision group 2 with mask 1, so terrain never collides with terrain and only the drone (group 1) meets it; the distance cull restores that pair when it re-enables a body.

## The render calls

`MovingDroneAviary` in `swarm/core/moving_drone.py` makes four kinds of engine render call. Every switch above lands in one of them.

| Call | Who | Flags | Arguments from the options |
|---|---|---|---|
| Drone observation, `getCameraImage` | every family, one drone | `ER_NO_SEGMENTATION_MASK`, plus `ER_DEPTH_ONLY` for depth families, plus the backend flag, the sky flag and, for a colour observer under `daylight`, the daylight flags | office: the episode light colour; with a sun: `sun_render_kwargs`; the sky kwargs; under `daylight`: `daylight_render_kwargs` and the sky photo |
| Multi-drone depth, `getDepthImagesBatch` | swarm families | the backend flag | none |
| On-demand RGB, `getCameraImage` | search and rescue | `ER_NO_SEGMENTATION_MASK` plus the sky flag, plus the daylight flags under `daylight` | `sun_render_kwargs`, the sky kwargs; under `daylight`: `daylight_render_kwargs` and the sky photo, with `shadow=1` |
| Video frame, `getCameraImage` | recordings only | `ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX` plus the sky flag, `shadow=1`, plus the daylight flags under `daylight` | the sky kwargs; under `daylight`: the light direction, `daylight_render_kwargs` and the sky photo |

Scored renders run with `shadow=0` and `renderer=ER_TINY_RENDERER`; the ray caster is selected by the flag, not by the renderer argument.

## From seed to light to sky

```text
seed
 |- seeded_sun(seed, night_share)           swarm/core/daylight.py
 |     day:   height drawn on the arc, hour read back, colour and strength from the height
 |     night: moon 10 to 60 degrees up, dark sky gradient
 |- apply_seeded_sun(env, ...)              sets _light_direction, _light_color, _sun
 |- sun_render_kwargs(sun)                  lightColor, lightAmbientCoeff, lightDiffuseCoeff
 |- _apply_sun_sky(seed)                    sky_from_sun and a day sun: ER_SWARM_SKY_SUN, sky_clouds: skyCloudSeed = seed
 |- _sky_kwargs()                           family sky_colors, else the moon's sky, plus the cloud seed
 '- _apply_daylight(seed)                   daylight and a day sun: ER_SWARM_DAYLIGHT with the picture flags, pick_sky(seed, sun, pack) for the photo and its turn
```

Same seed, same sun, same sky on every validator: every value is rounded to fixed decimals, and the renderer computes the sky from plain double arithmetic with its own polynomials, no maths library.

## A new map with the full picture

1. Export it with `validator/scripts/check_blender_export.py` clean: the engine loads a broken export without a word.
2. Load the pieces with `VISUAL_SHAPE_MATERIALS_FROM_MTL` if they carry several materials, `VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY` on thin shapes, `VISUAL_SHAPE_RENDER_TREE_CACHE` on the big static pieces, `GEOM_CONCAVE_BVH_CACHE` on the concave collision shapes, and `specularColor=[0, 0, 0]` on matte pieces if the glint will be on.
3. On the family runtime: `render_backend = "raycast"`, `seeded_sun = True`, `sky_from_sun = True`, `daylight = True` for a colour camera, and `sky_clouds`, `night_share`, `physics_mode` as the challenge wants; a `WIND_BY_MAP` entry if it wants wind.
4. For a colour camera on the ray caster, add the picture flags to its render call: `ER_SWARM_SHADOW_MAP | ER_SWARM_MOVER_SHADOW` with `shadow=1`, `ER_EDGE_ANTIALIAS`, `ER_ALPHA_CUTOUT`, `ER_TEXTURE_FILTER`, `ER_SPECULAR_GLINT`, `ER_SWARM_LINEAR_LIGHT`, and `shadowLightCoeff` below 0.8 for real shadows. Measured together on the solar-park slice: about 30 ms per 256 px frame at 2 threads.
5. Pin the frames: a test like `validator/tests/test_sky_sun.py` that renders one frame of the map at 1, 2 and 4 threads and compares it to committed hashes.
6. Bump the version: the family's pixels and flights are new.

## Tools and tests

| Tool | What it proves |
|---|---|
| `validator/scripts/verify_render_identity.py` | Orbit and episode hashes of 7 scenes across the families; run on the wheel before and after an engine change, with `SWARM_RENDER_BACKEND=raycast` for the other path |
| `validator/scripts/compare_render_masks.py` | Share of pixels with the same object id between TinyRenderer and the ray caster, per scene (99.4 to 99.9999 % today; the rest sit on object edges) |
| `validator/scripts/check_blender_export.py` | Every importer rule a Blender export can break, per file, with the fix |

| Test | Covers |
|---|---|
| `validator/tests/test_render_backend.py` | the backend switch, agreement with TinyRenderer, ray-cast frames pinned to hashes at 1, 2, 4 and 8 threads |
| `validator/tests/test_sky_sun.py` | the sun sky frames pinned to hashes on both paths |
| `validator/tests/test_sky_colors.py` | `sky_colors`, `sky_from_sun`, `sky_clouds` and the camera kwargs |
| `validator/tests/test_daylight.py` | the sun arc, the moon, determinism per seed |
| `validator/tests/test_daylight_family.py` | the `daylight` option: off everywhere today, its render arguments, the sky photo pick and turn per seed, the flags and the photo load in the environment |
| `validator/tests/test_wind.py` | the wind model, the cap, the off path |
| `validator/tests/test_family_physics_mode.py` | the physics mode, the ray-height ground effect, wind replacing the still-air drag |
| `validator/tests/test_bvh_cache.py` | the collision cache flag and the epoch folder |
| `validator/tests/test_moving_drone_cull.py` | the isolated collision group after re-enable, the visual hide skipped on the ray caster |
| `validator/tests/test_container_prewarm.py` | the overlapped container start |
| `validator/tests/test_check_blender_export.py` | every export rule fires once, with the right line |
