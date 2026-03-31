from ._shared import *


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    @property
    def area(self):
        return self.w * self.h


@dataclass
class Building:
    rect: Rect
    type: str
    facing: int = 0
    color: List[float] = field(default_factory=lambda: [1, 1, 1])


@dataclass
class Block:
    id: int
    rect: Rect
    too_small: bool


@dataclass
class RoadTile:
    x: float
    y: float
    type: str
    rotation: int = 0
    debug_label: str = ""


class SeededRNG:
    def __init__(self, seed):
        self._seed = seed if seed is not None else random.randint(0, 999999)
        self._rng = random.Random(self._seed)

    def range(self, min_val, max_val):
        return self._rng.uniform(min_val, max_val)

    def rand_int(self, min_val, max_val):
        return self._rng.randint(min_val, max_val)

    def next_float(self):
        return self._rng.random()

    def choice(self, seq):
        if not seq:
            return None
        return self._rng.choice(seq)


def generate_road_positions(rng, min_spacing, target_area, tile_size=None):
    if tile_size is None:
        tile_size = TILE_SIZE
    min_side_from_area = math.sqrt(target_area)
    needed_gap = max(min_spacing, min_side_from_area)
    raw_min_step = needed_gap + tile_size
    min_step = math.ceil(raw_min_step / tile_size) * tile_size
    max_step = math.ceil((min_step * 1.5) / tile_size) * tile_size
    num_tiles_f = round(MAP_SIZE / tile_size)
    if num_tiles_f < 1:
        num_tiles_f = 1
    effective_map_size = num_tiles_f * tile_size

    def get_auto_positions():
        raw_positions = [0]
        current_pos = 0
        safety = 0
        while current_pos < effective_map_size - tile_size and safety < 100:
            safety += 1
            jump_raw = rng.range(min_step, max_step)
            jump = round(jump_raw / tile_size) * tile_size
            next_pos = current_pos + jump
            if next_pos <= effective_map_size - tile_size:
                raw_positions.append(next_pos)
                current_pos = next_pos
            else:
                break
        if raw_positions[-1] != effective_map_size - tile_size:
            raw_positions.append(effective_map_size - tile_size)
        valid_positions = [0]
        for i in range(1, len(raw_positions) - 1):
            prev = valid_positions[-1]
            curr = raw_positions[i]
            gap = curr - prev - tile_size
            if gap >= needed_gap:
                valid_positions.append(curr)
        end_pos = effective_map_size - tile_size
        prev = valid_positions[-1]
        last_gap = end_pos - prev - tile_size
        if last_gap < needed_gap:
            if len(valid_positions) > 1:
                valid_positions.pop()
        valid_positions.append(end_pos)
        return valid_positions

    return get_auto_positions()


def extract_blocks(v_pos, h_pos, min_area, removed_h_segments=None,
                   removed_v_segments=None, added_h_segments=None,
                   added_v_segments=None, effective_tile_size=None):
    if effective_tile_size is None:
        effective_tile_size = TILE_SIZE
    if removed_h_segments is None:
        removed_h_segments = []
    if removed_v_segments is None:
        removed_v_segments = []
    if added_h_segments is None:
        added_h_segments = []
    if added_v_segments is None:
        added_v_segments = []
    h_removed_set = set(removed_h_segments)
    v_removed_set = set(removed_v_segments)
    n_cols = len(v_pos) - 1
    n_rows = len(h_pos) - 1
    parent = {}
    for i in range(n_cols):
        for j in range(n_rows):
            parent[(i, j)] = (i, j)

    def find(cell):
        if parent[cell] != cell:
            parent[cell] = find(parent[cell])
        return parent[cell]

    def union(cell1, cell2):
        root1, root2 = find(cell1), find(cell2)
        if root1 != root2:
            parent[root1] = root2

    for i in range(n_cols - 1):
        for j in range(n_rows):
            x = v_pos[i + 1]
            y1, y2 = h_pos[j], h_pos[j + 1]
            if (x, y1, y2) in v_removed_set:
                union((i, j), (i + 1, j))
    for i in range(n_cols):
        for j in range(n_rows - 1):
            y = h_pos[j + 1]
            x1, x2 = v_pos[i], v_pos[i + 1]
            if (y, x1, x2) in h_removed_set:
                union((i, j), (i, j + 1))
    groups = {}
    for i in range(n_cols):
        for j in range(n_rows):
            root = find((i, j))
            if root not in groups:
                groups[root] = []
            groups[root].append((i, j))
    initial_blocks = []
    bid = 0
    for root, cells in groups.items():
        min_i = min(c[0] for c in cells)
        max_i = max(c[0] for c in cells)
        min_j = min(c[1] for c in cells)
        max_j = max(c[1] for c in cells)
        x1 = v_pos[min_i] + effective_tile_size
        x2 = v_pos[max_i + 1]
        y1 = h_pos[min_j] + effective_tile_size
        y2 = h_pos[max_j + 1]
        w = x2 - x1
        h = y2 - y1
        if w <= 1 or h <= 1:
            continue
        initial_blocks.append(Block(bid, Rect(x1, y1, w, h), False))
        bid += 1
    final_blocks = []
    queue = list(initial_blocks)
    while queue:
        block = queue.pop(0)
        splitted = False
        for (y, sx1, sx2) in added_h_segments:
            if block.rect.y < y and (y + effective_tile_size) < (block.rect.y + block.rect.h):
                if sx1 < block.rect.x + block.rect.w and sx2 > block.rect.x:
                    h1 = y - block.rect.y
                    b1 = Block(bid, Rect(block.rect.x, block.rect.y, block.rect.w, h1), False)
                    bid += 1
                    y2_new = y + effective_tile_size
                    h2 = (block.rect.y + block.rect.h) - y2_new
                    b2 = Block(bid, Rect(block.rect.x, y2_new, block.rect.w, h2), False)
                    bid += 1
                    queue.append(b1)
                    queue.append(b2)
                    splitted = True
                    break
        if splitted:
            continue
        for (x, sy1, sy2) in added_v_segments:
            if block.rect.x < x and (x + effective_tile_size) < (block.rect.x + block.rect.w):
                if sy1 < block.rect.y + block.rect.h and sy2 > block.rect.y:
                    w1 = x - block.rect.x
                    b1 = Block(bid, Rect(block.rect.x, block.rect.y, w1, block.rect.h), False)
                    bid += 1
                    x2_new = x + effective_tile_size
                    w2 = (block.rect.x + block.rect.w) - x2_new
                    b2 = Block(bid, Rect(x2_new, block.rect.y, w2, block.rect.h), False)
                    bid += 1
                    queue.append(b1)
                    queue.append(b2)
                    splitted = True
                    break
        if not splitted:
            area = block.rect.w * block.rect.h
            block.too_small = area < (min_area * 0.95)
            final_blocks.append(block)
    return final_blocks


def generate_road_tiles(v_pos, h_pos, rng, max_block_size=60,
                        tile_size=None, difficulty=1):
    if tile_size is None:
        tile_size = TILE_SIZE
    num_tiles = round(MAP_SIZE / tile_size)
    effective_map_size = num_tiles * tile_size
    tiles = []
    occupied = set()
    v_set = set(v_pos)
    h_set = set(h_pos)
    h_pos_list = sorted(list(h_pos)) if not isinstance(h_pos, list) else h_pos
    v_pos_list = sorted(list(v_pos)) if not isinstance(v_pos, list) else v_pos
    if rng is None:
        rng = SeededRNG(42)
    removed_h_segments = []
    added_h_segments = []
    shift_chance = 0.6 if difficulty == 3 else 0.2

    for y in h_pos_list:
        if y == h_pos_list[0] or y == h_pos_list[-1]:
            continue
        idx = h_pos_list.index(y)
        y_prev = h_pos_list[idx - 1]
        y_next = h_pos_list[idx + 1]
        for i in range(len(v_pos_list) - 1):
            x1, x2 = v_pos_list[i], v_pos_list[i + 1]
            segment_width = x2 - x1
            is_too_wide = segment_width > max_block_size
            if rng.next_float() < shift_chance:
                if not is_too_wide:
                    removed_h_segments.append((y, x1, x2))
                else:
                    margin_check = 2 * tile_size
                    max_shift_up = (y - y_prev) - margin_check
                    max_shift_down = (y_next - y) - margin_check
                    possible_shifts = []
                    if max_shift_up >= tile_size:
                        possible_shifts.append(-tile_size)
                    if max_shift_up >= 2 * tile_size:
                        possible_shifts.append(-2 * tile_size)
                    if max_shift_down >= tile_size:
                        possible_shifts.append(tile_size)
                    if max_shift_down >= 2 * tile_size:
                        possible_shifts.append(2 * tile_size)
                    if possible_shifts:
                        shift = rng.choice(possible_shifts)
                        branch_y = y + shift
                        removed_h_segments.append((y, x1, x2))
                        added_h_segments.append((branch_y, x1, x2))

    removed_v_segments = []
    added_v_segments = []
    h_removed_at_intersection = set()
    for (y, x1, x2) in removed_h_segments:
        h_removed_at_intersection.add((x1, y))
        h_removed_at_intersection.add((x2, y))

    for x in v_pos_list:
        if x == v_pos_list[0] or x == v_pos_list[-1]:
            continue
        idx = v_pos_list.index(x)
        x_prev = v_pos_list[idx - 1]
        x_next = v_pos_list[idx + 1]
        for i in range(len(h_pos_list) - 1):
            y1, y2 = h_pos_list[i], h_pos_list[i + 1]
            segment_height = y2 - y1
            is_too_tall = segment_height > max_block_size
            would_create_corner = (
                (x, y1) in h_removed_at_intersection or
                (x, y2) in h_removed_at_intersection
            )
            if not would_create_corner and rng.next_float() < shift_chance:
                if not is_too_tall:
                    removed_v_segments.append((x, y1, y2))
                else:
                    margin_check = 2 * tile_size
                    max_shift_left = (x - x_prev) - margin_check
                    max_shift_right = (x_next - x) - margin_check
                    possible_shifts = []
                    if max_shift_left >= tile_size:
                        possible_shifts.append(-tile_size)
                    if max_shift_left >= 2 * tile_size:
                        possible_shifts.append(-2 * tile_size)
                    if max_shift_right >= tile_size:
                        possible_shifts.append(tile_size)
                    if max_shift_right >= 2 * tile_size:
                        possible_shifts.append(2 * tile_size)
                    if possible_shifts:
                        shift = rng.choice(possible_shifts)
                        branch_x = x + shift
                        removed_v_segments.append((x, y1, y2))
                        added_v_segments.append((branch_x, y1, y2))

    def to_grid_key(x, y):
        return f"{int(round(x))},{int(round(y))}"

    blocked_positions = set()
    for (y, x1, x2) in removed_h_segments:
        start_t = int(round((x1 + tile_size) / tile_size))
        end_t = int(round(x2 / tile_size))
        for i in range(start_t, end_t):
            bx = i * tile_size
            blocked_positions.add(to_grid_key(bx, y))
    for (x, y1, y2) in removed_v_segments:
        start_t = int(round((y1 + tile_size) / tile_size))
        end_t = int(round(y2 / tile_size))
        for i in range(start_t, end_t):
            by = i * tile_size
            blocked_positions.add(to_grid_key(x, by))

    added_intersections = {}
    for (y, x1, x2) in added_h_segments:
        added_intersections[(x1, y)] = added_intersections.get((x1, y), "") + "E"
        added_intersections[(x2, y)] = added_intersections.get((x2, y), "") + "W"
    for (x, y1, y2) in added_v_segments:
        added_intersections[(x, y1)] = added_intersections.get((x, y1), "") + "S"
        added_intersections[(x, y2)] = added_intersections.get((x, y2), "") + "N"

    def get_neighbors(x, y):
        neighbors = {"N": False, "S": False, "E": False, "W": False}
        if x in v_set:
            if y > 0:
                north_y = None
                for hy in reversed(h_pos_list):
                    if hy < y:
                        north_y = hy
                        break
                if north_y is not None:
                    if (x, north_y, y) not in removed_v_segments:
                        neighbors["N"] = True
                else:
                    neighbors["N"] = True
            if y < effective_map_size - tile_size:
                south_y = None
                for hy in h_pos_list:
                    if hy > y:
                        south_y = hy
                        break
                if south_y is not None:
                    if (x, y, south_y) not in removed_v_segments:
                        neighbors["S"] = True
                else:
                    neighbors["S"] = True
        if y in h_set:
            if x > 0:
                west_x = None
                for vx in reversed(v_pos_list):
                    if vx < x:
                        west_x = vx
                        break
                if west_x is not None:
                    if (y, west_x, x) not in removed_h_segments:
                        neighbors["W"] = True
                else:
                    neighbors["W"] = True
            if x < effective_map_size - tile_size:
                east_x = None
                for vx in v_pos_list:
                    if vx > x:
                        east_x = vx
                        break
                if east_x is not None:
                    if (y, x, east_x) not in removed_h_segments:
                        neighbors["E"] = True
                else:
                    neighbors["E"] = True
        added = added_intersections.get((x, y), "")
        if "N" in added:
            neighbors["N"] = True
        if "S" in added:
            neighbors["S"] = True
        if "E" in added:
            neighbors["E"] = True
        if "W" in added:
            neighbors["W"] = True
        return neighbors

    def classify_intersection(neighbors):
        n, s, e, w = neighbors["N"], neighbors["S"], neighbors["E"], neighbors["W"]
        count = sum([n, s, e, w])
        if count == 4:
            return ("intersection", 0)
        elif count == 3:
            if not w:
                return ("t_junction", 0)
            elif not n:
                return ("t_junction", 90)
            elif not e:
                return ("t_junction", 180)
            else:
                return ("t_junction", 270)
        elif count == 2:
            if n and s:
                return ("straight_v", 0)
            if e and w:
                return ("straight_h", 90)
            if n and w:
                return ("corner", 270)
            elif n and e:
                return ("corner", 0)
            elif e and s:
                return ("corner", 90)
            else:
                return ("corner", 180)
        elif count == 1:
            if n:
                return ("dead_end", 0)
            if s:
                return ("dead_end", 180)
            if e:
                return ("dead_end", 270)
            if w:
                return ("dead_end", 90)
            return ("straight_v", 0)
        else:
            if n or s:
                return ("straight_v", 0)
            return ("straight_h", 90)

    tile_map = {}

    for x in v_pos:
        for y in h_pos:
            neighbors = get_neighbors(x, y)
            tile_type, rotation = classify_intersection(neighbors)
            k = to_grid_key(x, y)
            tile_map[k] = RoadTile(x, y, tile_type, rotation)
            occupied.add(k)

    num_map_tiles = int(round(effective_map_size / tile_size))
    for x in v_pos:
        for i in range(num_map_tiles):
            y = i * tile_size
            key = to_grid_key(x, y)
            if key not in occupied and key not in blocked_positions:
                tile_map[key] = RoadTile(x, y, "straight_v", 0)
                occupied.add(key)
    for y in h_pos:
        for i in range(num_map_tiles):
            x = i * tile_size
            key = to_grid_key(x, y)
            if key not in occupied and key not in blocked_positions:
                tile_map[key] = RoadTile(x, y, "straight_h", 90)
                occupied.add(key)

    for (y, x1, x2) in added_h_segments:
        start_t = int(round((x1 + tile_size) / tile_size))
        end_t = int(round(x2 / tile_size))
        for i in range(start_t, end_t):
            bx = i * tile_size
            key = to_grid_key(bx, y)
            if key not in tile_map:
                tile_map[key] = RoadTile(bx, y, "straight_h", 90)
                occupied.add(key)
    for (x, y1, y2) in added_v_segments:
        start_t = int(round((y1 + tile_size) / tile_size))
        end_t = int(round(y2 / tile_size))
        for i in range(start_t, end_t):
            by = i * tile_size
            key = to_grid_key(x, by)
            if key not in tile_map:
                tile_map[key] = RoadTile(x, by, "straight_v", 0)
                occupied.add(key)

    for (x, y), added_dir in added_intersections.items():
        key = to_grid_key(x, y)
        if key in tile_map:
            existing_tile = tile_map[key]
            neighbors = {"N": False, "S": False, "E": False, "W": False}
            if existing_tile.type == "straight_v":
                neighbors["N"] = True
                neighbors["S"] = True
            elif existing_tile.type == "straight_h":
                neighbors["E"] = True
                neighbors["W"] = True
            elif existing_tile.type == "intersection":
                neighbors["N"] = True
                neighbors["S"] = True
                neighbors["E"] = True
                neighbors["W"] = True
            elif existing_tile.type == "t_junction":
                rot = existing_tile.rotation
                if rot == 0:
                    neighbors.update({"N": True, "S": True, "E": True})
                elif rot == 90:
                    neighbors.update({"S": True, "E": True, "W": True})
                elif rot == 180:
                    neighbors.update({"N": True, "S": True, "W": True})
                elif rot == 270:
                    neighbors.update({"N": True, "E": True, "W": True})
            if "N" in added_dir:
                neighbors["N"] = True
            if "S" in added_dir:
                neighbors["S"] = True
            if "E" in added_dir:
                neighbors["E"] = True
            if "W" in added_dir:
                neighbors["W"] = True
            new_type, new_rot = classify_intersection(neighbors)
            tile_map[key] = RoadTile(x, y, new_type, new_rot)

    roundabout_candidates = [t for t in tile_map.values() if t.type == "intersection"]
    roundabout_min_dist = 3 * tile_size
    existing_roundabouts = []
    for tile in roundabout_candidates:
        if rng.next_float() > 0.3:
            continue
        ts = tile_size
        n_positions = [
            (tile.x, tile.y - ts), (tile.x, tile.y + ts),
            (tile.x - ts, tile.y), (tile.x + ts, tile.y),
        ]
        safe = True
        for nx, ny in n_positions:
            nk = to_grid_key(nx, ny)
            if nk not in tile_map:
                safe = False
                break
            neighbor = tile_map[nk]
            if neighbor.type not in ["straight_v", "straight_h"]:
                safe = False
                break
        if safe:
            for ex_r in existing_roundabouts:
                dist = math.sqrt((tile.x - ex_r.x) ** 2 + (tile.y - ex_r.y) ** 2)
                if dist < roundabout_min_dist:
                    safe = False
                    break
        if safe:
            tile.type = "roundabout"
            existing_roundabouts.append(tile)
            for nx, ny in n_positions:
                nk = to_grid_key(nx, ny)
                if nk in tile_map:
                    tile_map[nk].type = "roundabout_arm"

    crossing_candidates = [
        t for t in tile_map.values() if t.type in ["straight_v", "straight_h"]
    ]
    for tile in crossing_candidates:
        if rng.next_float() > 0.15:
            continue
        ts = tile_size
        n_positions = [
            (tile.x, tile.y - ts), (tile.x, tile.y + ts),
            (tile.x - ts, tile.y), (tile.x + ts, tile.y),
        ]
        safe = True
        for nx, ny in n_positions:
            nk = to_grid_key(nx, ny)
            if nk in tile_map:
                if tile_map[nk].type not in ["straight_v", "straight_h"]:
                    safe = False
                    break
        if safe:
            tile.type = "crossing"

    tiles = list(tile_map.values())
    return (tiles, blocked_positions, removed_h_segments,
            removed_v_segments, added_h_segments, added_v_segments)


# ---------------------------------------------------------------------------
# SECTION 2b: Building generation (ported from city_gen.py)
# ---------------------------------------------------------------------------
TEMPLATES = {
    "house": [
        {"w": 12, "d": 12}, {"w": 13, "d": 13},
        {"w": 14, "d": 14}, {"w": 15, "d": 15},
    ],
    "apt": [
        {"w": 18, "d": 18}, {"w": 20, "d": 20},
        {"w": 22, "d": 22}, {"w": 24, "d": 24},
    ],
    "tower": [
        {"w": 28, "d": 28}, {"w": 32, "d": 32},
        {"w": 36, "d": 36}, {"w": 40, "d": 40},
    ],
}


def generate_buildings(blocks, rng, city_type=2, _b_margin=4, difficulty=1):
    buildings = []
    occupied_rects = []
    min_gap = 1.0 if difficulty == 3 else 1.5
    if difficulty == 1:
        base_road_margin = 2
        base_center_gap = 0
        squeeze_road_margin = 1.5
        squeeze_center_gap = 0
        ultra_road_margin = 1.5
        ultra_center_gap = 0
    elif difficulty == 2:
        base_road_margin = 3
        base_center_gap = 0
        squeeze_road_margin = 3
        squeeze_center_gap = 0
        ultra_road_margin = 3
        ultra_center_gap = 0
    else:
        base_road_margin = 3
        base_center_gap = 0
        squeeze_road_margin = 3
        squeeze_center_gap = 0
        ultra_road_margin = 3
        ultra_center_gap = 0

    def rects_overlap(r1, r2):
        return not (r1.x2 <= r2.x or r2.x2 <= r1.x or r1.y2 <= r2.y or r2.y2 <= r1.y)

    def can_place(new_rect, current_occupied):
        for existing in current_occupied:
            if rects_overlap(new_rect, existing):
                return False
        return True

    def get_category_for_block(_block_h, rng_inst, c_type, diff=1, block_tier=None):
        if diff == 3:
            return "tower"
        if block_tier in ["house", "apt", "tower"]:
            return block_tier
        rand = rng_inst.next_float()
        if c_type == 1:
            return "house" if rand < 0.7 else "apt"
        elif c_type == 3:
            return "tower" if rand < 0.5 else "apt"
        else:
            if rand < 0.33:
                return "house"
            elif rand < 0.66:
                return "apt"
            return "tower"

    def fill_row_justified(start_val, end_val, fixed_val, depth, axis, category,
                           facing, rng_inst, current_builds, current_occupied):
        length = end_val - start_val
        if length < 5:
            return
        valid_tmpls = [
            t for t in TEMPLATES.get(category, TEMPLATES["house"])
            if t["d"] == depth
        ]
        if not valid_tmpls:
            if difficulty == 3:
                return
            valid_tmpls = TEMPLATES.get(category, TEMPLATES["house"])
        candidates = []
        current_used = 0
        safety = 0
        while safety < 50:
            safety += 1
            tmpl = rng_inst.choice(valid_tmpls)
            if tmpl is None:
                break
            if current_used + tmpl["w"] > length:
                break
            candidates.append(tmpl)
            current_used += tmpl["w"] + min_gap
        if not candidates:
            return
        final_gap = min_gap
        curr_pos = start_val + min_gap
        for tmpl in candidates:
            if axis == "x":
                r = Rect(curr_pos, fixed_val, tmpl["w"], depth)
            else:
                r = Rect(fixed_val, curr_pos, depth, tmpl["w"])
            if can_place(r, current_occupied):
                current_builds.append(Building(r, category, facing=facing))
                current_occupied.append(r)
            curr_pos += tmpl["w"] + final_gap

    def process_block(block, road_margin=4, center_gap=1.5, block_tier="commercial"):
        local_builds = []
        local_occupied = []
        valid_rect = Rect(
            block.rect.x + road_margin,
            block.rect.y + road_margin,
            max(0, block.rect.w - road_margin * 2),
            max(0, block.rect.h - road_margin * 2),
        )
        if valid_rect.w < 1 or valid_rect.h < 1:
            return [], 0
        category = get_category_for_block(
            valid_rect.h, rng, city_type, difficulty, block_tier=block_tier
        )
        ref_tmpl = rng.choice(TEMPLATES.get(category, TEMPLATES["house"]))
        if ref_tmpl is None:
            return [], 0
        north_depth = ref_tmpl["d"]
        fill_row_justified(
            valid_rect.x, valid_rect.x2 - 0.1,
            valid_rect.y, north_depth, "x", category, 0, rng,
            local_builds, local_occupied,
        )
        has_north = len(local_builds) > 0
        south_depth = 0
        available_depth = valid_rect.h - north_depth - center_gap
        has_south = False
        if available_depth > 8:
            cat_chain = [category]
            selected_cat = None
            selected_ref = None
            for cat in cat_chain:
                tmpls = TEMPLATES.get(cat, [])
                valid_tmpls = [t for t in tmpls if t["d"] <= available_depth]
                if valid_tmpls:
                    selected_cat = cat
                    selected_ref = rng.choice(valid_tmpls)
                    break
            if selected_cat and selected_ref:
                south_depth = selected_ref["d"]
                y_pos = valid_rect.y2 - south_depth
                prev_count = len(local_builds)
                fill_row_justified(
                    valid_rect.x, valid_rect.x2 - 0.1,
                    y_pos, south_depth, "x", selected_cat, 180, rng,
                    local_builds, local_occupied,
                )
                if len(local_builds) > prev_count:
                    has_south = True
        start_y = valid_rect.y + north_depth + center_gap
        end_y = valid_rect.y2 - south_depth - center_gap
        if end_y - start_y > 4:
            ref_tmpl = rng.choice(TEMPLATES.get(category, TEMPLATES["house"]))
            if ref_tmpl:
                fill_row_justified(
                    start_y, end_y, valid_rect.x, ref_tmpl["d"], "y",
                    category, 270, rng, local_builds, local_occupied,
                )
            ref_tmpl = rng.choice(TEMPLATES.get(category, TEMPLATES["house"]))
            if ref_tmpl:
                x_pos = valid_rect.x2 - ref_tmpl["d"]
                fill_row_justified(
                    start_y, end_y, x_pos, ref_tmpl["d"], "y",
                    category, 90, rng, local_builds, local_occupied,
                )
        score = 0
        if has_north:
            score += 1
        if has_south:
            score += 1
        return local_builds, score

    def process_block_centered(block, road_margin=4, block_tier="commercial"):
        local_builds = []
        valid_rect = Rect(
            block.rect.x + road_margin, block.rect.y + road_margin,
            block.rect.w - 2 * road_margin, block.rect.h - 2 * road_margin,
        )
        if valid_rect.w <= 0 or valid_rect.h <= 0:
            return []
        category = get_category_for_block(
            valid_rect.h, rng, city_type, difficulty, block_tier=block_tier
        )
        is_horizontal = valid_rect.w > valid_rect.h
        local_occupied = []
        if is_horizontal:
            cy = valid_rect.y + valid_rect.h / 2
            tmpl = rng.choice(TEMPLATES.get(category, TEMPLATES["house"]))
            if tmpl is None:
                return []
            depth = tmpl["d"]
            y_pos = cy - depth / 2
            fill_row_justified(
                valid_rect.x, valid_rect.x2, y_pos, depth, "x",
                category, 0, rng, local_builds, local_occupied,
            )
        else:
            cx = valid_rect.x + valid_rect.w / 2
            tmpl = rng.choice(TEMPLATES.get(category, TEMPLATES["house"]))
            if tmpl is None:
                return []
            depth = tmpl["d"]
            x_pos = cx - depth / 2
            fill_row_justified(
                valid_rect.y, valid_rect.y2, x_pos, depth, "y",
                category, 90, rng, local_builds, local_occupied,
            )
        return local_builds

    def process_block_chaotic(block):
        local_builds = []
        local_occupied = []
        valid_rect = Rect(block.rect.x, block.rect.y, block.rect.w, block.rect.h)
        if valid_rect.w < 10 or valid_rect.h < 10:
            return []
        tower_tmpls = TEMPLATES.get("tower", [{"w": 25, "d": 25}])
        attempts = 0
        while attempts < 100:
            attempts += 1
            tmpl = rng.choice(tower_tmpls)
            if tmpl is None:
                break
            tw, td = tmpl["w"], tmpl["d"]
            if valid_rect.w - tw < 1 or valid_rect.h - td < 1:
                continue
            rx = valid_rect.x + rng.next_float() * (valid_rect.w - tw)
            ry = valid_rect.y + rng.next_float() * (valid_rect.h - td)
            new_rect = Rect(rx, ry, tw, td)
            if can_place(new_rect, local_occupied):
                facing = rng.choice([0, 90, 180, 270])
                local_builds.append(Building(new_rect, "tower", facing=facing))
                local_occupied.append(new_rect)
        return local_builds

    for block in blocks:
        tier_rand = rng.next_float()
        if city_type == 1:
            block_tier = "house" if tier_rand < 0.7 else "apt"
        elif city_type == 3:
            block_tier = "tower" if tier_rand < 0.5 else "apt"
        else:
            if tier_rand < 0.33:
                block_tier = "house"
            elif tier_rand < 0.66:
                block_tier = "apt"
            else:
                block_tier = "tower"
        if difficulty == 3:
            final_builds = process_block_chaotic(block)
            buildings.extend(final_builds)
            for b in final_builds:
                occupied_rects.append(b.rect)
            continue
        builds_4, score_4 = process_block(
            block, road_margin=base_road_margin,
            center_gap=base_center_gap, block_tier=block_tier,
        )
        final_builds = builds_4
        if score_4 < 2:
            builds_2, score_2 = process_block(
                block, road_margin=squeeze_road_margin,
                center_gap=squeeze_center_gap, block_tier=block_tier,
            )
            if score_2 >= 2 or len(builds_2) > len(builds_4):
                final_builds = builds_2
                score_4 = score_2
            if score_4 < 2:
                builds_1, score_1 = process_block(
                    block, road_margin=ultra_road_margin,
                    center_gap=ultra_center_gap, block_tier=block_tier,
                )
                if score_1 >= 2 or len(builds_1) > len(final_builds):
                    final_builds = builds_1
            if score_4 < 2 or len(final_builds) == 0:
                builds_center = process_block_centered(
                    block, road_margin=base_road_margin, block_tier=block_tier
                )
                if len(builds_center) > 0:
                    final_builds = builds_center
        buildings.extend(final_builds)
        for b in final_builds:
            occupied_rects.append(b.rect)
    return buildings


def generate_city(seed=42, min_spacing=20, target_area=1000, city_type=2,
                  max_block_size=2000, difficulty=1, road_scale=1.0):
    rng = SeededRNG(seed)
    effective_tile_size = TILE_SIZE * road_scale
    max_block_dimension = int(math.sqrt(max_block_size))
    v_pos = generate_road_positions(
        rng, min_spacing, target_area, tile_size=effective_tile_size
    )
    h_pos = generate_road_positions(
        rng, min_spacing, target_area, tile_size=effective_tile_size
    )
    tiles, blocked_positions, rm_h, rm_v, add_h, add_v = generate_road_tiles(
        v_pos, h_pos, rng, max_block_dimension,
        tile_size=effective_tile_size, difficulty=difficulty,
    )
    blocks = extract_blocks(
        v_pos, h_pos, 400, rm_h, rm_v, add_h, add_v,
        effective_tile_size=effective_tile_size,
    )
    buildings = generate_buildings(blocks, rng, city_type=city_type, difficulty=difficulty)
    safe_buildings = []
    road_rects = []
    margin = 0.5
    for t in tiles:
        road_rects.append((
            t.x - margin, t.y - margin,
            t.x + effective_tile_size + margin,
            t.y + effective_tile_size + margin,
        ))
    roundabout_centers = []
    for t in tiles:
        if t.type == "roundabout":
            cx = t.x + effective_tile_size / 2
            cy = t.y + effective_tile_size / 2
            roundabout_centers.append((cx, cy))
    diamond_limit = effective_tile_size * 1.9
    for b in buildings:
        corners = [
            (b.rect.x, b.rect.y),
            (b.rect.x + b.rect.w, b.rect.y),
            (b.rect.x, b.rect.y + b.rect.h),
            (b.rect.x + b.rect.w, b.rect.y + b.rect.h),
        ]
        is_safe = True
        for (rcx, rcy) in roundabout_centers:
            for (bx, by) in corners:
                manhattan_dist = abs(bx - rcx) + abs(by - rcy)
                if manhattan_dist < diamond_limit:
                    is_safe = False
                    break
            if not is_safe:
                break
        if not is_safe:
            continue
        bx1, by1 = b.rect.x, b.rect.y
        bx2, by2 = b.rect.x + b.rect.w, b.rect.y + b.rect.h
        for (rx1, ry1, rx2, ry2) in road_rects:
            if not (bx1 >= rx2 or bx2 <= rx1 or by1 >= ry2 or by2 <= ry1):
                is_safe = False
                break
        if is_safe:
            safe_buildings.append(b)
    buildings = safe_buildings
    return {"seed": seed, "roads": tiles, "blocks": blocks, "buildings": buildings}


# ---------------------------------------------------------------------------
