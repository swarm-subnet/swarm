from ._shared import *


def build_curved_roof(loader, roof_base_z, shell_meshes, cli):
    shell_sx, shell_sy = _shell_mesh_scale_xy(shell_meshes)
    _spawn_generated_mesh(
        shell_meshes["roof"],
        loader.texture_path,
        cli,
        with_collision=True,
        use_texture=False,
        rgba=ROOF_UNIFORM_COLOR,
        double_sided=True,
        base_position=(0.0, 0.0, roof_base_z),
        mesh_scale_xyz=(shell_sx, shell_sy, 1.0),
    )
    _spawn_generated_mesh(
        shell_meshes["fillers"],
        loader.texture_path,
        cli,
        with_collision=True,
        use_texture=False,
        rgba=WALL_UNIFORM_COLOR,
        double_sided=True,
        base_position=(0.0, 0.0, roof_base_z),
        mesh_scale_xyz=(shell_sx, shell_sy, 1.0),
    )


def build_roof_truss_system(floor_top_z, roof_base_z, shell_meshes, cli):
    _ = floor_top_z
    if not ENABLE_ROOF_TRUSS_SYSTEM:
        return {
            "roof_ribs": 0,
            "roof_rib_segments": 0,
            "truss_frames": 0,
            "truss_members": 0,
            "interior_columns": 0,
        }

    shell_sx, shell_sy = _shell_mesh_scale_xy(shell_meshes)
    _spawn_generated_mesh(
        shell_meshes["truss"],
        texture_path="",
        cli=cli,
        with_collision=TRUSS_WITH_COLLISION,
        use_texture=False,
        rgba=TRUSS_UNIFORM_COLOR,
        double_sided=True,
        base_position=(0.0, 0.0, roof_base_z),
        mesh_scale_xyz=(shell_sx, shell_sy, 1.0),
    )

    cfg = shell_meshes.get("config", {})
    rib_count = max(1, int(cfg.get("truss_rib_count", 5)))
    node_count = max(3, int(cfg.get("truss_node_count", 9)))
    panel_count = node_count - 1
    top_profile_segments = max(
        int(cfg.get("truss_top_profile_segments", 96)), node_count * 8
    )
    top_segments = rib_count * top_profile_segments
    lower_segments = rib_count * panel_count
    web_segments = rib_count * max(0, panel_count - 2)
    total_segments = top_segments + lower_segments + web_segments
    return {
        "roof_ribs": rib_count,
        "roof_rib_segments": total_segments,
        "truss_frames": rib_count,
        "truss_members": total_segments,
        "interior_columns": 0,
    }
