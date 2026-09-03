"""Office map builder: geometry, collision, layout and determinism checks."""

import numpy as np
import pybullet as p
import pytest

from swarm.core.maps.office import OFFICE_CEILING_M, build_office_map, office_layout, office_pieces
from swarm.core.maps.office.builder import office_scale
from swarm.core.maps.office.layout import DESK_PAIRS, SETS, footprint


SEED = 0
SCALE = office_scale(SEED)


@pytest.fixture()
def office_client():
    cli = p.connect(p.DIRECT)
    info = build_office_map(seed=SEED, cli=cli)
    yield cli, info
    p.disconnect(cli)


def _at(x=0.0, y=0.0, z=0.0):
    """A nominal point placed in this seed's room."""
    sx, sy, sz = SCALE
    return [x * sx, y * sy, z * sz]


def _ray_hit(cli, start, end):
    hit = p.rayTest(start, end, physicsClientId=cli)[0]
    return hit[0], hit[3]


def test_office_map_bodies(office_client):
    cli, info = office_client
    names = set(info["bodies"])
    assert {"floor", "shell", "led", "backdrop"} <= names
    assert sum(n.startswith("piece:") for n in names) == len(office_pieces()["pieces"])
    assert sum(n.startswith("fixed:") for n in names) == len(office_pieces()["fixed"])
    assert info["window_plug"] >= 0
    assert info["ceiling_m"] == pytest.approx(OFFICE_CEILING_M * SCALE[2])


def test_office_map_shell_geometry(office_client):
    cli, info = office_client
    _, floor_pt = _ray_hit(cli, _at(9, 3.8, 2.0), _at(9, 3.8, -1))
    assert abs(floor_pt[2] - 0.0) < 1e-3, "the floor stays at z=0 whatever the room size"
    _, ceil_pt = _ray_hit(cli, _at(9, 3.8, 1.0), _at(9, 3.8, 5))
    assert abs(ceil_pt[2] - 3.0 * SCALE[2]) < 1e-3
    # above the tallest furniture, so no layout can stand between the ray and the column
    _, col_pt = _ray_hit(cli, _at(13.0, 5.96, 2.8), _at(16, 5.96, 2.8))
    assert abs(col_pt[0] - 14.065 * SCALE[0]) < 5e-3


def test_office_map_furniture_collision(office_client):
    """A piece stands where the layout put it, at its true size, and is solid."""
    cli, info = office_client
    pieces = {q["id"]: q for q in office_pieces()["pieces"]}
    pid, x, y, z, yaw = next(row for row in info["layout"] if row[0] == "C1_table_final")
    body, top = _ray_hit(cli, [x, y, 2.0], [x, y, 0.2])
    assert body == info["bodies"]["piece:" + pid]
    assert abs(top[2] - pieces[pid]["size"][2]) < 2e-2, "furniture is never scaled with the room"


def test_office_map_window_plug(office_client):
    cli, info = office_client
    for y, z in [(3.2, 1.8), (4.4, 1.2)]:
        body, pt = _ray_hit(cli, _at(16.5, y, z), _at(21, y, z))
        assert body == info["window_plug"]
        assert pt[0] < 18.1 * SCALE[0]


def _aabbs(seed):
    cli = p.connect(p.DIRECT)
    info = build_office_map(seed=seed, cli=cli)
    out = [p.getAABB(b, physicsClientId=cli) for _, b in sorted(info["bodies"].items())]
    p.disconnect(cli)
    return np.array(out)


def test_office_map_deterministic():
    """One seed builds one room with one furniture layout, every time and on
    every validator."""
    assert np.allclose(_aabbs(1), _aabbs(1))
    assert office_layout(7) == office_layout(7)


def test_office_map_size_follows_the_seed():
    """Different seeds build differently sized rooms, so a floorplan fitted to one
    episode does not line up with the next."""
    a, b = _aabbs(1), _aabbs(999)
    assert not np.allclose(a, b), "two seeds produced the same room"
    span_a = float(a[:, 1, 0].max() - a[:, 0, 0].min())
    span_b = float(b[:, 1, 0].max() - b[:, 0, 0].min())
    assert abs(span_a - span_b) > 0.05, f"room length barely moved: {span_a:.2f} vs {span_b:.2f}"


def _groups():
    """Pieces that legitimately overlap each other: a chair under its desk, the
    two desks of a pair, the members of a set."""
    grp = {}
    for name, ids in SETS.items():
        for i in ids:
            grp[i] = name
    for a, b in DESK_PAIRS:
        grp[a] = grp[b] = "pair:" + a
    for q in office_pieces()["pieces"]:
        if q.get("desk"):
            grp[q["id"]] = grp[q["desk"]]
        if q.get("on_top_of"):
            grp[q["id"]] = grp.get(q["on_top_of"], q["on_top_of"])
    return grp


@pytest.mark.parametrize("seed", [1, 2, 3, 1117, 11199, 99989, 334844])
def test_office_layout_is_a_real_office(seed):
    """Every piece is placed, inside the room, with nothing from another group
    overlapping it, and all four desks keep their chair and drawer beside them."""
    pieces = {q["id"]: q for q in office_pieces()["pieces"]}
    scale = office_scale(seed)
    rows = office_layout(seed, scale)
    assert {r[0] for r in rows} == set(pieces), "every piece is placed exactly once"
    grp = _groups()
    boxes = [(pid, footprint(pieces[pid]["size"], x, y, yaw))
             for pid, x, y, z, yaw in rows if pieces[pid]["floor_standing"]]
    for pid, b in boxes:
        assert 0.0 - 1e-3 <= b.x0 and b.x1 <= 18.0 * scale[0] + 1e-3, pid
        assert 0.0 - 1e-3 <= b.y0 and b.y1 <= 7.6 * scale[1] + 1e-3, pid
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            assert not (ba.hits(bb) and grp.get(a, a) != grp.get(b, b)), f"{a} overlaps {b}"
    pose = {r[0]: r for r in rows}
    for q in pieces.values():
        if q.get("desk"):
            d = pose[q["desk"]]
            dist = float(np.hypot(pose[q["id"]][1] - d[1], pose[q["id"]][2] - d[2]))
            assert dist < 1.6, f"{q['id']} lost its desk"
            assert pose[q["id"]][4] == d[4], "a chair turns with its desk"


def test_office_layout_changes_with_the_seed():
    """A memorised map is worthless: the same piece stands somewhere else on
    another seed, for most of the furniture."""
    a = {r[0]: r[1:] for r in office_layout(1)}
    b = {r[0]: r[1:] for r in office_layout(2)}
    moved = sum(1 for k in a if abs(a[k][0] - b[k][0]) > 0.3 or abs(a[k][1] - b[k][1]) > 0.3)
    assert moved > 0.7 * len(a), f"only {moved} of {len(a)} pieces moved"
