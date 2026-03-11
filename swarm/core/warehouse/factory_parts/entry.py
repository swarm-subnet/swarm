from ._shared import *
from .barriers import build_factory_barrier_ring
from .belts import build_single_belt_network


def build_embedded_factory(conveyor_loader, floor_top_z, area_layout, seed, cli=0):
    if not ENABLE_EMBEDDED_FACTORY_MAP:
        return {"factory_map_embedded": False}

    factory_area = (area_layout or {}).get("FACTORY")
    if not factory_area:
        return {
            "factory_map_embedded": False,
            "factory_map_reason": "FACTORY area not present in layout.",
        }

    size_xy = (float(factory_area["sx"]), float(factory_area["sy"]))
    center_xy = (float(factory_area["cx"]), float(factory_area["cy"]))
    try:
        network = build_single_belt_network(
            conveyor_loader,
            seed=int(seed) + EMBEDDED_FACTORY_SEED_OFFSET,
            cli=cli,
            center_xy=center_xy,
            size_xy=size_xy,
            floor_z=float(floor_top_z),
        )
        barrier_info = build_factory_barrier_ring(
            conveyor_loader=conveyor_loader,
            floor_top_z=float(floor_top_z),
            factory_area=factory_area,
            network=network,
            cli=cli,
        )
        if SHOW_AREA_LAYOUT_MARKERS:
            p.addUserDebugText(
                text=f"EMBEDDED FACTORY | {size_xy[0]:.0f} x {size_xy[1]:.0f} m",
                textPosition=[center_xy[0], center_xy[1], float(floor_top_z) + 0.03],
                textColorRGB=[0.08, 0.10, 0.12],
                textSize=1.2,
                lifeTime=0.0,
                physicsClientId=cli,
            )
        return {
            "factory_map_embedded": True,
            "factory_area_center_xy": center_xy,
            "factory_area_size_m": size_xy,
            "factory_network": network,
            **barrier_info,
        }
    except Exception as exc:
        return {"factory_map_embedded": False, "factory_map_reason": str(exc)}
