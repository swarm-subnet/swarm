from ._shared import *


def get_global_scale(seed: int) -> float:
    rng = random.Random(seed + TYPE_3_SCALE_SEED_OFFSET)
    return rng.uniform(TYPE_3_SCALE_MIN, TYPE_3_SCALE_MAX)


def _make_noise_params(seed: int, gs: float) -> List[dict]:
    rng = random.Random(seed + 9999)
    params = []
    for i in range(TERRAIN_N_OCTAVES):
        params.append(
            {
                "amp": 1.0 / (1.6**i),
                "fx": rng.uniform(0.008, 0.022) / gs * (1.4**i),
                "fy": rng.uniform(0.008, 0.022) / gs * (1.4**i),
                "px": rng.uniform(0, 2 * math.pi),
                "py": rng.uniform(0, 2 * math.pi),
            }
        )
    return params


def _sample_terrain_height(
    x: float, y: float, noise_params: List[dict], amplitude: float
) -> float:
    h = 0.0
    for o in noise_params:
        h += (
            o["amp"] * math.sin(x * o["fx"] + o["px"]) * math.cos(y * o["fy"] + o["py"])
        )
    return h * amplitude


def get_terrain_z(x: float, y: float, seed: int, gs: float) -> float:
    noise = _make_noise_params(seed, gs)
    amp_rng = random.Random(seed + 12345)
    amp = amp_rng.uniform(2.0 * gs, 5.0 * gs)

    mesh_size = 1500.0 * gs
    res = TERRAIN_RESOLUTION
    half = mesh_size / 2.0
    step = mesh_size / (res - 1)

    gx = (x + half) / step
    gy = (y + half) / step
    gx = max(0.0, min(res - 1.001, gx))
    gy = max(0.0, min(res - 1.001, gy))

    c0, r0 = int(gx), int(gy)
    c1, r1 = min(c0 + 1, res - 1), min(r0 + 1, res - 1)
    tx, ty = gx - c0, gy - r0

    x0 = -half + c0 * step
    x1 = -half + c1 * step
    y0 = -half + r0 * step
    y1 = -half + r1 * step

    h00 = round(_sample_terrain_height(x0, y0, noise, amp), 4)
    h10 = round(_sample_terrain_height(x1, y0, noise, amp), 4)
    h01 = round(_sample_terrain_height(x0, y1, noise, amp), 4)
    h11 = round(_sample_terrain_height(x1, y1, noise, amp), 4)

    return (
        h00 * (1 - tx) * (1 - ty)
        + h10 * tx * (1 - ty)
        + h01 * (1 - tx) * ty
        + h11 * tx * ty
    )


# ---------------------------------------------------------------------------
# SECTION 3: Shape cache
# ---------------------------------------------------------------------------
class _ShapeCache:
    def __init__(self):
        self._cache: Dict = {}

    def clear(self):
        self._cache = {}

    def get(self, cli: int, path: str, scale_vec: list, rgba: Optional[list] = None):
        key = (cli, path, tuple(scale_vec), tuple(rgba) if rgba else None)
        if key in self._cache:
            return self._cache[key]
        kw = {"fileName": path, "meshScale": scale_vec}
        if rgba:
            kw["rgbaColor"] = rgba
            kw["specularColor"] = [0, 0, 0]
        vis = p.createVisualShape(p.GEOM_MESH, physicsClientId=cli, **kw)
        col = p.createCollisionShape(
            p.GEOM_MESH, fileName=path, meshScale=scale_vec, physicsClientId=cli
        )
        self._cache[key] = (vis, col)
        return vis, col


_cache = _ShapeCache()


# ---------------------------------------------------------------------------
# SECTION 4: Terrain OBJ generation & spawning
# ---------------------------------------------------------------------------
def _generate_terrain_tiles(
    obj_dir: str,
    size: float,
    res: int,
    noise_params: List[dict],
    amplitude: float,
    tiles: int,
) -> Tuple[Dict, float, List[str]]:
    half = size / 2.0
    step = size / (res - 1)
    heights: Dict[Tuple[int, int], float] = {}

    for r in range(res):
        for c in range(res):
            x = -half + c * step
            y = -half + r * step
            heights[(r, c)] = _sample_terrain_height(x, y, noise_params, amplitude)

    cells_per_tile = (res - 1) // tiles
    tile_paths: List[str] = []

    for tr in range(tiles):
        for tc in range(tiles):
            r_start = tr * cells_per_tile
            c_start = tc * cells_per_tile
            r_end = (tr + 1) * cells_per_tile if tr < tiles - 1 else res - 1
            c_end = (tc + 1) * cells_per_tile if tc < tiles - 1 else res - 1

            path = os.path.join(obj_dir, f"terrain_tile_{tr}_{tc}.obj")
            with open(path, "w") as f:
                vert_map: Dict[Tuple[int, int], int] = {}
                vi = 1
                for r in range(r_start, r_end + 1):
                    for c in range(c_start, c_end + 1):
                        x = -half + c * step
                        y = -half + r * step
                        f.write(f"v {x:.4f} {y:.4f} {heights[(r, c)]:.4f}\n")
                        vert_map[(r, c)] = vi
                        vi += 1
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        v00 = vert_map[(r, c)]
                        v10 = vert_map[(r, c + 1)]
                        v01 = vert_map[(r + 1, c)]
                        v11 = vert_map[(r + 1, c + 1)]
                        f.write(f"f {v00} {v10} {v11}\n")
                        f.write(f"f {v00} {v11} {v01}\n")
            tile_paths.append(path)

    return heights, step, tile_paths


def _terrain_z_at(
    x: float, y: float, heights: Dict, res: int, step: float, half: float
) -> float:
    gx = (x + half) / step
    gy = (y + half) / step
    gx = max(0.0, min(res - 1.001, gx))
    gy = max(0.0, min(res - 1.001, gy))
    c0, r0 = int(gx), int(gy)
    c1, r1 = min(c0 + 1, res - 1), min(r0 + 1, res - 1)
    tx, ty = gx - c0, gy - r0
    h00 = heights.get((r0, c0), 0.0)
    h10 = heights.get((r0, c1), 0.0)
    h01 = heights.get((r1, c0), 0.0)
    h11 = heights.get((r1, c1), 0.0)
    return (
        h00 * (1 - tx) * (1 - ty)
        + h10 * tx * (1 - ty)
        + h01 * (1 - tx) * ty
        + h11 * tx * ty
    )


def _spawn_terrain(cli: int, seed: int, obj_dir: str, gs: float):
    noise_params = _make_noise_params(seed, gs)
    amp_rng = random.Random(seed + 12345)
    amplitude = amp_rng.uniform(2.0 * gs, 5.0 * gs)

    mesh_size = 1500.0 * gs
    tile_dir = os.path.join(obj_dir, f"terrain_seed_{seed}")
    os.makedirs(tile_dir, exist_ok=True)
    heights, step, tile_paths = _generate_terrain_tiles(
        tile_dir,
        mesh_size,
        TERRAIN_RESOLUTION,
        noise_params,
        amplitude,
        TERRAIN_TILES,
    )

    for tile_path in tile_paths:
        vi = p.createVisualShape(
            p.GEOM_MESH,
            fileName=tile_path,
            meshScale=[1, 1, 1],
            rgbaColor=SNOW,
            specularColor=[0, 0, 0],
            physicsClientId=cli,
        )
        ci = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=tile_path,
            meshScale=[1, 1, 1],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=cli,
        )
        p.createMultiBody(0, ci, vi, [0, 0, 0], physicsClientId=cli)

    gv = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[2000 * gs, 2000 * gs, 0.5],
        rgbaColor=SNOW,
        specularColor=[0, 0, 0],
        physicsClientId=cli,
    )
    p.createMultiBody(0, -1, gv, [0, 0, -0.45], physicsClientId=cli)

    res = TERRAIN_RESOLUTION
    half = mesh_size / 2.0

    def get_z(x: float, y: float) -> float:
        return _terrain_z_at(x, y, heights, res, step, half)

    return get_z


# ---------------------------------------------------------------------------
# SECTION 5: Placement helpers
