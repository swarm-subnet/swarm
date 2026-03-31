from ._shared import *


def _spawn_obj_with_mtl_parts(
    loader,
    model_name,
    world_anchor_xyz,
    yaw_deg,
    mesh_scale_xyz,
    local_anchor_xyz,
    cli,
    with_collision=True,
    fallback_rgba=(0.78, 0.78, 0.78, 1.0),
    rgba_gain=1.0,
):
    def _gain_rgba(c):
        r = max(0.0, min(1.0, float(c[0]) * float(rgba_gain)))
        g = max(0.0, min(1.0, float(c[1]) * float(rgba_gain)))
        b = max(0.0, min(1.0, float(c[2]) * float(rgba_gain)))
        a = float(c[3]) if len(c) >= 4 else 1.0
        return (r, g, b, a)

    if with_collision:
        _spawn_mesh_with_anchor(
            loader=loader,
            model_name=model_name,
            world_anchor_xyz=world_anchor_xyz,
            yaw_deg=yaw_deg,
            mesh_scale_xyz=mesh_scale_xyz,
            local_anchor_xyz=local_anchor_xyz,
            cli=cli,
            with_collision=True,
            use_texture=False,
            rgba=(1.0, 1.0, 1.0, 0.0),
            double_sided=False,
        )

    model_path = loader._asset_path(model_name)
    material_parts = _obj_material_parts(model_path)
    if material_parts:
        for part in material_parts:
            part_rgba = _gain_rgba(part["rgba"])
            _spawn_mesh_with_anchor(
                loader=loader,
                model_name=part["path"],
                world_anchor_xyz=world_anchor_xyz,
                yaw_deg=yaw_deg,
                mesh_scale_xyz=mesh_scale_xyz,
                local_anchor_xyz=local_anchor_xyz,
                cli=cli,
                with_collision=False,
                use_texture=False,
                rgba=part_rgba,
                double_sided=False,
            )
        return len(material_parts)

    fallback_rgba_adj = _gain_rgba(fallback_rgba)
    _spawn_mesh_with_anchor(
        loader=loader,
        model_name=model_name,
        world_anchor_xyz=world_anchor_xyz,
        yaw_deg=yaw_deg,
        mesh_scale_xyz=mesh_scale_xyz,
        local_anchor_xyz=local_anchor_xyz,
        cli=cli,
        with_collision=False,
        use_texture=False,
        rgba=fallback_rgba_adj,
        double_sided=False,
    )
    return 0


def _truss_rib_x_positions(shell_meshes):
    cfg = (shell_meshes or {}).get("config", {}) or {}
    rib_count = max(1, int(cfg.get("truss_rib_count", 5)))
    base_x = float(cfg.get("warehouse_size_x", WAREHOUSE_BASE_SIZE_X))
    end_margin_base = max(0.0, float(cfg.get("truss_end_margin_x", 6.0)))
    shell_sx = (float(WAREHOUSE_SIZE_X) / base_x) if abs(base_x) > 1e-9 else 1.0
    end_margin = end_margin_base * shell_sx

    half_x = float(WAREHOUSE_SIZE_X) * 0.5
    lo = -half_x + end_margin
    hi = half_x - end_margin
    if hi <= lo:
        lo = -half_x * 0.8
        hi = half_x * 0.8
    if rib_count <= 1:
        return [0.0]
    step = (hi - lo) / float(rib_count - 1)
    return [float(lo + (i * step)) for i in range(rib_count)]
