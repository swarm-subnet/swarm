"""Ground texture and flat-ground spawning for forest generation."""

from ._shared import *
from .assets import _clamp_mode_id


# ---------------------------------------------------------------------------
# SECTION 6: Ground texture generation
# ---------------------------------------------------------------------------
def _clamp_u8(v: float) -> int:
    return max(0, min(255, int(round(v))))


def _hash_noise_01(x: int, y: int, seed: int) -> float:
    n = (x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)
    n = (n << 13) ^ n
    return (
        1.0
        - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7FFFFFFF)
        / 1073741824.0
    )


def _write_bmp24(path: str, width: int, height: int, rgb_data: bytearray) -> None:
    row_stride = width * 3
    row_pad = (4 - (row_stride % 4)) % 4
    image_size = (row_stride + row_pad) * height
    file_size = 54 + image_size
    with open(path, "wb") as f:
        f.write(b"BM")
        f.write(file_size.to_bytes(4, "little"))
        f.write((0).to_bytes(4, "little"))
        f.write((54).to_bytes(4, "little"))
        f.write((40).to_bytes(4, "little"))
        f.write(width.to_bytes(4, "little", signed=True))
        f.write(height.to_bytes(4, "little", signed=True))
        f.write((1).to_bytes(2, "little"))
        f.write((24).to_bytes(2, "little"))
        f.write((0).to_bytes(4, "little"))
        f.write(image_size.to_bytes(4, "little"))
        f.write((2835).to_bytes(4, "little"))
        f.write((2835).to_bytes(4, "little"))
        f.write((0).to_bytes(4, "little"))
        f.write((0).to_bytes(4, "little"))
        pad = b"\x00" * row_pad
        for y in range(height - 1, -1, -1):
            row = y * row_stride
            for x in range(width):
                idx = row + x * 3
                f.write(bytes((rgb_data[idx + 2], rgb_data[idx + 1], rgb_data[idx])))
            if row_pad:
                f.write(pad)


def _ensure_ground_texture() -> str:
    if os.path.exists(GROUND_TEXTURE_PATH):
        return GROUND_TEXTURE_PATH
    os.makedirs(os.path.dirname(GROUND_TEXTURE_PATH), exist_ok=True)
    w = h = GROUND_TEXTURE_RES
    data = bytearray(w * h * 3)
    seed = GROUND_TEXTURE_SEED
    rng = random.Random(seed)
    grass_a = (96.0, 130.0, 88.0)
    grass_b = (109.0, 146.0, 95.0)
    for y in range(h):
        for x in range(w):
            idx = (y * w + x) * 3
            macro = 0.5 + 0.5 * math.sin((x * 0.022) + (y * 0.017))
            noise = _hash_noise_01(x, y, seed) * 0.5 + 0.5
            t = (0.7 * macro) + (0.3 * noise)
            r = grass_a[0] * (1.0 - t) + grass_b[0] * t
            g = grass_a[1] * (1.0 - t) + grass_b[1] * t
            b = grass_a[2] * (1.0 - t) + grass_b[2] * t
            grain = (_hash_noise_01(x * 3, y * 3, seed + 19) * 0.5 + 0.5) - 0.5
            data[idx + 0] = _clamp_u8(r + grain * 14.0)
            data[idx + 1] = _clamp_u8(g + grain * 18.0)
            data[idx + 2] = _clamp_u8(b + grain * 12.0)
    for _ in range(56):
        cx = rng.uniform(0.0, w - 1.0)
        cy = rng.uniform(0.0, h - 1.0)
        radius = rng.uniform(16.0, 52.0)
        dirt = (
            rng.uniform(103.0, 126.0),
            rng.uniform(93.0, 111.0),
            rng.uniform(74.0, 93.0),
        )
        min_x = max(0, int(cx - radius - 1))
        max_x = min(w - 1, int(cx + radius + 1))
        min_y = max(0, int(cy - radius - 1))
        max_y = min(h - 1, int(cy + radius + 1))
        inv_r2 = 1.0 / max(1.0, radius * radius)
        for py in range(min_y, max_y + 1):
            dy = py - cy
            for px in range(min_x, max_x + 1):
                dx = px - cx
                d2 = (dx * dx + dy * dy) * inv_r2
                if d2 >= 1.0:
                    continue
                edge = 1.0 - d2
                blend = (edge * edge) * rng.uniform(0.28, 0.58)
                idx = (py * w + px) * 3
                data[idx + 0] = _clamp_u8(
                    data[idx + 0] * (1.0 - blend) + dirt[0] * blend
                )
                data[idx + 1] = _clamp_u8(
                    data[idx + 1] * (1.0 - blend) + dirt[1] * blend
                )
                data[idx + 2] = _clamp_u8(
                    data[idx + 2] * (1.0 - blend) + dirt[2] * blend
                )
    _write_bmp24(GROUND_TEXTURE_PATH, w, h, data)
    return GROUND_TEXTURE_PATH


def _ground_texture_id(cli: int) -> Optional[int]:
    if cli in _CLI_TEX_CACHE:
        return _CLI_TEX_CACHE[cli]
    tex_path = _ensure_ground_texture()
    try:
        tex_id = p.loadTexture(tex_path, physicsClientId=cli)
    except Exception:
        tex_id = None
    _CLI_TEX_CACHE[cli] = tex_id
    return tex_id


# ---------------------------------------------------------------------------
# SECTION 7: Ground spawning
# ---------------------------------------------------------------------------
def _ground_rgba_for_mode(mode_id: int) -> List[float]:
    cat = MAP_MODE_CONFIG[_clamp_mode_id(mode_id)]["primary_category"]
    if cat == "autumn":
        return GROUND_RGBA_AUTUMN
    if cat == "dead":
        return GROUND_RGBA_DEAD
    if cat == "snow":
        return GROUND_RGBA_SNOW
    return GROUND_RGBA


def _spawn_ground(cli: int, mode_id: int = 1) -> None:
    rgba = _ground_rgba_for_mode(mode_id)
    half = GROUND_SIZE_M * 0.5
    col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[half, half, GROUND_HALF_THICKNESS_M],
        physicsClientId=cli,
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half, half, GROUND_HALF_THICKNESS_M],
        rgbaColor=rgba,
        specularColor=[0.0, 0.0, 0.0],
        physicsClientId=cli,
    )
    body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[0.0, 0.0, -GROUND_HALF_THICKNESS_M],
        physicsClientId=cli,
    )
    if mode_id == 1:
        tex_id = _ground_texture_id(cli)
        if tex_id is not None:
            p.changeVisualShape(body, -1, textureUniqueId=tex_id, physicsClientId=cli)


__all__ = [name for name in globals() if not name.startswith("__")]
