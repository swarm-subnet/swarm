"""Seeded furniture layout for the office.

The layout is a pure function of the seed and the room scale: every validator
places the same pieces in the same spots, and a different seed gives a
different office. Pieces are placed the way a real office is arranged:

  wall units   shelves, cabinets, the wardrobe, the printer table - back flat
               against a wall run in their zone, sliding along it
  desk row     the four east desks stay two back-to-back pairs with their
               chairs and drawers, in a row along either axis
  meeting      the two table halves keep their six chairs
  lounge       the armchairs, tables, stool and lamp move as one, wall-backed
  free items   bins, lamps, stools - near a wall or beside a desk
  fixed        walls, columns, partitions and wall pictures never move

All footprints are axis-aligned (yaw in 90 degree steps), so overlap and
clearance are box tests. Doors keep a 1 m apron clear and the floor must stay
one walkable region; an attempt that fails is retried with the next sub-seed.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from importlib.resources import files as _pkg_files

import numpy as np

from swarm.constants import OFFICE_LAYOUT_RETRIES, OFFICE_LAYOUT_SEED_OFFSET

CLEAR = 0.15            # gap kept between any two pieces
WALK = 0.90             # walkway width the floor must keep
DOOR_APRON = 1.0        # nothing within this distance in front of a door
TRIES = 80
GRID = 0.15             # walkability grid pitch

# Nominal (unscaled) room geometry. Walls a piece may back onto:
# (name, axis, coordinate, span, inward normal). Door openings are the gaps in
# the skirting; the window and radiator keep tall units off the east wall.
WALL_RUNS = (
    ("south_A", "y", 0.0, (0.05, 1.35), +1), ("south_A2", "y", 0.0, (1.8, 3.3), +1),
    ("west_A", "x", 0.0, (0.05, 6.15), +1),
    ("alcove_e", "x", 3.35, (5.2, 7.05), -1), ("alcove_n", "y", 7.1, (0.75, 3.3), -1),
    ("south_M", "y", 0.0, (3.55, 12.4), +1),
    ("corridor_1", "y", 5.125, (3.4, 3.95), -1), ("corridor_2", "y", 5.125, (4.95, 10.1), -1),
    ("corridor_3", "y", 5.125, (11.0, 12.4), -1),
    ("south_E", "y", 0.0, (12.7, 17.9), +1),
    ("east_lo", "x", 18.0, (0.05, 2.6), -1), ("east_hi", "x", 18.0, (5.0, 7.5), -1),
    ("north_E", "y", 7.6, (12.7, 17.9), -1), ("step_e", "x", 12.6, (5.3, 7.5), +1),
)
DOORS = (((4.0, 4.9), 5.2), ((10.15, 10.95), 5.2))        # (x span, y); the apron opens to -y
ENTRANCE = (0.4, 6.7, 1.3)                                  # keep-out disc at the diagonal corner
ZONES = {"A": (0.0, 3.36, 0.0, 7.1), "M": (3.4, 12.45, 0.0, 5.125), "E": (12.62, 18.0, 0.0, 7.6)}
ZONE_WALLS = {"A": ("south_A", "south_A2", "west_A", "alcove_e", "alcove_n"),
              "M": ("south_M", "corridor_1", "corridor_2", "corridor_3"),
              "E": ("south_E", "east_lo", "east_hi", "north_E", "step_e")}
FIXED_BOXES = ((1.4, 0.0, 1.76, 0.3), (14.065, 5.775, 14.435, 6.145), (17.86, 2.65, 18.0, 4.95))
WALL_BOXES = ((3.35, 5.125, 3.5, 7.175), (3.275, 5.2, 12.675, 5.35), (12.45, 5.05, 12.6, 7.75),
              (0.665, 7.1, 3.425, 7.25), (-0.6, 6.2, 0.8, 7.27))
SETS = {
    "lounge": ("B1_chair_final", "B1_chair_final_2", "B2a_final", "B2b_final", "B3_final", "B6_final"),
    "meeting": ("C1_table_final", "C1_table_final_2", "C2_chair_final", "C2_chair_final_2",
                "C2_chair_final_3", "C2_chair_final_4", "C2_chair_final_5", "C2_chair_final_6"),
    "printer": ("E4_table_big", "E4_printer_final"),
}
DESK_PAIRS = (("E1_desk_final", "E1_desk_final_2"), ("E1_desk_final_3", "E1_desk_final_4"))
ZONE_A_WALL_UNITS = ("A1_final", "A1B_final", "A2_final", "A3_final", "A3_final_2", "A4_final")
FREE_ITEMS = (("A6", "A"), ("A9_final", "A"), ("A10", "A"), ("D3_final", "M"), ("E6_final", "E"), ("E8_final", "E"))
LOUNGE_WALLS = {"corridor_2": "M", "south_M": "M", "south_E": "E", "east_lo": "E"}
SIDE_ANGLE = {"-y": 0, "+x": 90, "+y": 180, "-x": 270}
WALL_SIDE = {("y", +1): "-y", ("y", -1): "+y", ("x", +1): "-x", ("x", -1): "+x"}

_MANIFEST = None


def office_pieces() -> dict:
    """The shipped pieces manifest, read once per process."""
    global _MANIFEST
    if _MANIFEST is None:
        path = _pkg_files("swarm").joinpath("assets", "maps", "custom", "office", "pieces.json")
        _MANIFEST = json.loads(path.read_text())
    return _MANIFEST


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    def grow(self, m):
        return Box(self.x0 - m, self.y0 - m, self.x1 + m, self.y1 + m)

    def hits(self, o):
        return self.x0 < o.x1 and o.x0 < self.x1 and self.y0 < o.y1 and o.y0 < self.y1

    def inside(self, z, tol=0.005):
        return (z[0] - tol <= self.x0 and self.x1 <= z[1] + tol
                and z[2] - tol <= self.y0 and self.y1 <= z[3] + tol)


@dataclass
class Placed:
    piece: str
    x: float
    y: float
    z: float
    yaw: int
    box: Box


def footprint(size, x, y, yaw):
    w, d = (size[0], size[1]) if yaw % 180 == 0 else (size[1], size[0])
    return Box(x - w / 2, y - d / 2, x + w / 2, y + d / 2)


def rot(dx, dy, yaw):
    c, s = round(math.cos(math.radians(yaw))), round(math.sin(math.radians(yaw)))
    return dx * c - dy * s, dx * s + dy * c


def yaw_for_wall(back_side, axis, normal):
    """Yaw that turns the piece's back side into the wall."""
    return (SIDE_ANGLE[WALL_SIDE[(axis, normal)]] - SIDE_ANGLE[back_side]) % 360


class _Layout:
    def __init__(self, seed, scale):
        self.pieces = {p["id"]: p for p in office_pieces()["pieces"]}
        self.rng = random.Random(seed)
        self.placed: dict = {}
        sx, sy = float(scale[0]), float(scale[1])
        self.sx, self.sy = sx, sy
        # everything the pieces are placed against lives in the scaled room
        self.room = (0.0, 18.0 * sx, 0.0, 7.6 * sy)
        self.zones = {k: (z[0] * sx, z[1] * sx, z[2] * sy, z[3] * sy) for k, z in ZONES.items()}
        self.walls = [(n, a, c * (sx if a == "x" else sy),
                       (s0 * (sy if a == "x" else sx), s1 * (sy if a == "x" else sx)), nm)
                      for n, a, c, (s0, s1), nm in WALL_RUNS]
        self.doors = [((d0 * sx, d1 * sx), y * sy) for (d0, d1), y in DOORS]
        self.entrance = (ENTRANCE[0] * sx, ENTRANCE[1] * sy, ENTRANCE[2])
        self.blocked = [Box(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in FIXED_BOXES]
        self.wall_boxes = [Box(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in WALL_BOXES]
        self.forced: list = []

    # ---- geometry --------------------------------------------------------
    def wall(self, name):
        return next(w for w in self.walls if w[0] == name)

    def free(self, box, zone=None, clear=CLEAR):
        if zone and not box.inside(self.zones[zone]):
            return False
        if not box.inside(self.room):
            return False
        g = box.grow(clear)
        if any(g.hits(b) for b in self.blocked) or any(g.hits(p.box) for p in self.placed.values()):
            return False
        for (dx0, dx1), dy in self.doors:
            if box.hits(Box(dx0 - 0.3, dy - DOOR_APRON, dx1 + 0.3, dy)):
                return False
        ex, ey, er = self.entrance
        if math.hypot(max(box.x0 - ex, 0, ex - box.x1), max(box.y0 - ey, 0, ey - box.y1)) < er:
            return False
        return True

    def put(self, pid, x, y, yaw, z=None):
        p = self.pieces[pid]
        z = p["base_z"] if z is None else z
        self.placed[pid] = Placed(pid, x, y, z, yaw, footprint(p["size"], x, y, yaw))

    def authored_offsets(self, ids, anchor):
        a = self.pieces[anchor]["authored"]
        return {i: ((self.pieces[i]["authored"]["x"] - a["x"]) * self.sx,
                    (self.pieces[i]["authored"]["y"] - a["y"]) * self.sy) for i in ids}

    # ---- rules -------------------------------------------------------------
    def wall_unit(self, pid, zone, walls):
        p = self.pieces[pid]
        size, back = p["size"], p["back_side"] or "-y"
        gap = max(0.0, min(p["wall_gap"], 0.1))
        runs = [self.wall(n) for n in walls]
        self.rng.shuffle(runs)
        for name, axis, coord, span, normal in runs:
            yaw = yaw_for_wall(back, axis, normal)
            w, d = (size[0], size[1]) if yaw % 180 == 0 else (size[1], size[0])
            along = w if axis == "y" else d
            if span[1] - span[0] < along + 2 * CLEAR:
                continue
            for _ in range(TRIES // 2):
                t = self.rng.uniform(span[0] + along / 2 + CLEAR, span[1] - along / 2 - CLEAR)
                x, y = (t, coord + normal * (d / 2 + gap)) if axis == "y" else (coord + normal * (w / 2 + gap), t)
                if self.free(footprint(size, x, y, yaw), zone):
                    self.put(pid, x, y, yaw)
                    return True
        return False

    def rigid_set(self, ids, anchor, zone, yaw_choices=(0, 90, 180, 270), wall=None, back=None):
        """A group placed as one body, positioned by its own hull so a wall-backed
        set touches the wall with its outermost member."""
        offs = self.authored_offsets(ids, anchor)
        z = self.zones[zone]
        back = back or self.pieces[anchor]["back_side"] or "+y"
        for _ in range(TRIES):
            if wall:
                name, axis, coord, span, normal = self.wall(wall)
                yaw = yaw_for_wall(back, axis, normal)
            else:
                yaw = self.rng.choice(yaw_choices)
            rel = {i: footprint(self.pieces[i]["size"], *rot(*offs[i], yaw), yaw) for i in ids}
            hull = Box(min(b.x0 for b in rel.values()), min(b.y0 for b in rel.values()),
                       max(b.x1 for b in rel.values()), max(b.y1 for b in rel.values()))
            if wall:
                along = (hull.x1 - hull.x0) if axis == "y" else (hull.y1 - hull.y0)
                if span[1] - span[0] < along + 2 * CLEAR:
                    return False
                t = self.rng.uniform(span[0] + along / 2 + CLEAR, span[1] - along / 2 - CLEAR)
                if axis == "y":
                    ax, ay = t - (hull.x0 + hull.x1) / 2, coord - (hull.y0 if normal > 0 else hull.y1)
                else:
                    ay, ax = t - (hull.y0 + hull.y1) / 2, coord - (hull.x0 if normal > 0 else hull.x1)
            else:
                ax = self.rng.uniform(z[0] - hull.x0 + CLEAR, z[1] - hull.x1 - CLEAR)
                ay = self.rng.uniform(z[2] - hull.y0 + CLEAR, z[3] - hull.y1 - CLEAR)
            if not self.free(Box(hull.x0 + ax, hull.y0 + ay, hull.x1 + ax, hull.y1 + ay), zone):
                continue
            for i, b in rel.items():
                self.put(i, (b.x0 + b.x1) / 2 + ax, (b.y0 + b.y1) / 2 + ay, yaw)
            return True
        return False

    def desk_row(self, zone):
        """Two back-to-back desk pairs in a row along either axis, each desk
        keeping its chair and drawer where the real office has them."""
        z = self.zones[zone]
        members = {d: [i for i, p in self.pieces.items() if p.get("desk") == d] for pr in DESK_PAIRS for d in pr}

        def extent(pair):
            ids = list(pair) + members[pair[0]] + members[pair[1]]
            boxes = [footprint(self.pieces[i]["size"], self.pieces[i]["authored"]["x"] * self.sx,
                               self.pieces[i]["authored"]["y"] * self.sy, 0) for i in ids]
            return max(b.y1 for b in boxes) - min(b.y0 for b in boxes)

        depth = max(extent(pr) for pr in DESK_PAIRS)
        for _ in range(TRIES):
            along_y = self.rng.random() < 0.6
            yaw = 0 if along_y else 90
            gap = self.rng.uniform(WALK, WALK + 1.0)
            if along_y:
                x = self.rng.uniform(z[0] + 1.6, z[1] - 1.6)
                y0 = self.rng.uniform(z[2] + depth / 2 + CLEAR, z[3] - depth - gap - depth / 2 - CLEAR)
                anchors = [(x, y0), (x, y0 + depth + gap)]
            else:
                y = self.rng.uniform(z[2] + 1.6, z[3] - 1.6)
                x0 = self.rng.uniform(z[0] + depth / 2 + CLEAR, z[1] - depth - gap - depth / 2 - CLEAR)
                anchors = [(x0, y), (x0 + depth + gap, y)]
            self.rng.shuffle(anchors)
            plan, ok = {}, True
            for (a, b), (ax, ay) in zip(DESK_PAIRS, anchors):
                pa, pb = self.pieces[a]["authored"], self.pieces[b]["authored"]
                mid = ((pa["x"] + pb["x"]) / 2 * self.sx, (pa["y"] + pb["y"]) / 2 * self.sy)
                for d in (a, b):
                    for i in [d] + members[d]:
                        p = self.pieces[i]
                        dx, dy = rot(p["authored"]["x"] * self.sx - mid[0], p["authored"]["y"] * self.sy - mid[1], yaw)
                        jit = (self.rng.uniform(-0.05, 0.05), self.rng.uniform(-0.05, 0.05)) if p["role"] == "chair" else (0.0, 0.0)
                        x, y = ax + dx + jit[0], ay + dy + jit[1]
                        if not self.free(footprint(p["size"], x, y, yaw), zone, clear=CLEAR if p["role"] == "desk" else 0.0):
                            ok = False
                            break
                        plan[i] = (x, y)
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                for i, (x, y) in plan.items():
                    self.put(i, x, y, yaw)
                return True
        return False

    def free_item(self, pid, zone):
        p = self.pieces[pid]
        z = self.zones[zone]
        for _ in range(TRIES):
            yaw = self.rng.choice((0, 90, 180, 270))
            roll = self.rng.random()
            desks = [q for q in self.placed.values() if self.pieces[q.piece]["role"] == "desk" and q.box.inside(z)]
            if roll < 0.7:                                    # by a wall
                name, axis, coord, span, normal = self.wall(self.rng.choice(ZONE_WALLS[zone]))
                t, off = self.rng.uniform(*span), self.rng.uniform(0.15, 0.5)
                x, y = (t, coord + normal * off) if axis == "y" else (coord + normal * off, t)
            elif desks and roll < 0.9:                        # beside a desk
                d = self.rng.choice(desks)
                x, y = self.rng.uniform(d.box.x0 - 0.6, d.box.x1 + 0.6), self.rng.uniform(d.box.y0 - 0.6, d.box.y1 + 0.6)
            else:
                x, y = self.rng.uniform(z[0] + 0.3, z[1] - 0.3), self.rng.uniform(z[2] + 0.3, z[3] - 0.3)
            if self.free(footprint(p["size"], x, y, yaw), zone):
                self.put(pid, x, y, yaw)
                return True
        return False

    # ---- checks --------------------------------------------------------------
    def walkable(self):
        """One connected floor for a WALK-wide walker: the largest free region
        covers at least 90% of the free floor and reaches both doors. Small
        dead pockets, which real offices have, are allowed."""
        r = WALK / 2
        x0, x1, y0, y1 = self.room
        nx, ny = int((x1 - x0) / GRID), int((y1 - y0) / GRID)
        xs = x0 + (np.arange(nx) + 0.5) * GRID
        ys = y0 + (np.arange(ny) + 0.5) * GRID
        free = np.zeros((nx, ny), dtype=bool)
        for zx0, zx1, zy0, zy1 in self.zones.values():
            free[np.ix_((xs - r >= zx0) & (xs + r <= zx1), (ys - r >= zy0) & (ys + r <= zy1))] = True
        for b in self.blocked + self.wall_boxes + [p.box for p in self.placed.values()]:
            free[np.ix_((xs + r > b.x0) & (xs - r < b.x1), (ys + r > b.y0) & (ys - r < b.y1))] = False
        total = int(free.sum())
        if total == 0:
            return False
        label = np.zeros_like(free, dtype=np.int32)
        sizes = []
        for i, j in zip(*np.nonzero(free)):
            if label[i, j]:
                continue
            k = len(sizes) + 1
            stack, n = [(i, j)], 0
            label[i, j] = k
            while stack:
                a, b = stack.pop()
                n += 1
                for c, d in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                    if 0 <= c < nx and 0 <= d < ny and free[c, d] and not label[c, d]:
                        label[c, d] = k
                        stack.append((c, d))
            sizes.append(n)
        main = int(np.argmax(sizes)) + 1
        if sizes[main - 1] < 0.9 * total:
            return False
        for (dx0, dx1), dy in self.doors:
            i, j = int(((dx0 + dx1) / 2 - x0) / GRID), int((dy - r - 0.1 - y0) / GRID)
            if not (0 <= i < nx and 0 <= j < ny and label[i, j] == main):
                return False
        return True

    # ---- the whole office ----------------------------------------------------
    def generate(self):
        pz = self.pieces
        fallback = []
        for pid in sorted(ZONE_A_WALL_UNITS, key=lambda i: -pz[i]["size"][2]):
            if not self.wall_unit(pid, "A", ZONE_WALLS["A"]):
                fallback.append(pid)
        if not self.wall_unit("D1_final", "E", ("step_e", "north_E")):
            fallback.append("D1_final")
        walls = ["east_hi", "east_lo"]
        self.rng.shuffle(walls)
        if not any(self.rigid_set(SETS["printer"], "E4_table_big", "E", wall=w, back="+x") for w in walls):
            fallback.append("printer")
        # the desk row and the meeting table may swap rooms, like a real reorganisation
        desk_zone, meet_zone = self.rng.choice((("E", "M"), ("M", "E"), ("E", "E"), ("M", "M")))
        other = {"E": "M", "M": "E"}
        big = [("desks", desk_zone), ("meeting", meet_zone), ("lounge", None)]
        self.rng.shuffle(big)
        for kind, zone in big:
            if kind == "desks":
                ok = self.desk_row(zone) or self.desk_row(other[zone])
            elif kind == "meeting":
                ok = (self.rigid_set(SETS["meeting"], "C1_table_final", zone, yaw_choices=(0, 90))
                      or self.rigid_set(SETS["meeting"], "C1_table_final", other[zone], yaw_choices=(0, 90)))
            else:
                walls = list(LOUNGE_WALLS)
                self.rng.shuffle(walls)
                ok = any(self.rigid_set(SETS["lounge"], "B1_chair_final", LOUNGE_WALLS[w], wall=w) for w in walls)
            if not ok:
                fallback.append(kind)
        for pid, zone in FREE_ITEMS:
            if not self.free_item(pid, zone):
                fallback.append(pid)
        # anything unplaced keeps the real office's position, but only on free floor
        for pid, p in pz.items():
            if pid in self.placed:
                continue
            x, y = p["authored"]["x"] * self.sx, p["authored"]["y"] * self.sy
            if not self.free(footprint(p["size"], x, y, 0), clear=0.0):
                self.forced.append(pid)
            self.put(pid, x, y, 0)
        # what rests on a piece rides with it
        for pid, p in pz.items():
            top = p.get("on_top_of")
            if top and top in self.placed:
                base = self.placed[top]
                dx, dy = rot((p["authored"]["x"] - pz[top]["authored"]["x"]) * self.sx,
                             (p["authored"]["y"] - pz[top]["authored"]["y"]) * self.sy, base.yaw)
                self.put(pid, base.x + dx, base.y + dy, base.yaw)
        return fallback


def office_layout(map_seed: int, scale=(1.0, 1.0, 1.0)) -> list:
    """Furniture placements for a seed in a room of this scale: a list of
    (piece id, x, y, z, yaw_deg). An attempt is accepted when every piece found
    a place and the floor is walkable; otherwise the next sub-seed is tried, and
    after the last one the best attempt is kept."""
    base = (int(map_seed) ^ OFFICE_LAYOUT_SEED_OFFSET) & 0xFFFFFFFF
    best = None
    for k in range(OFFICE_LAYOUT_RETRIES):
        lay = _Layout((base * 7919 + k) & 0xFFFFFFFF, scale)
        fallback = lay.generate()
        walk = lay.walkable()
        score = (len(lay.forced), len(fallback), 0 if walk else 1)
        if best is None or score < best[0]:
            best = (score, lay)
        if not fallback and walk:
            break
    lay = best[1]
    return [(p.piece, round(p.x, 4), round(p.y, 4), round(p.z, 4), p.yaw)
            for p in sorted(lay.placed.values(), key=lambda q: q.piece)]
