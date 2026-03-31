from ._shared import *


def _forklift_yaw_back_to_wall(attached_wall):
    if attached_wall == "north":
        return 180.0
    if attached_wall == "south":
        return 0.0
    if attached_wall == "east":
        return 270.0
    if attached_wall == "west":
        return 90.0
    return 0.0


def _purge_generated_model_artifacts(model_path):
    cache_key = os.path.abspath(model_path)
    _OBJ_MTL_SPLIT_CACHE.pop(cache_key, None)
    _OBJ_COLLISION_PROXY_CACHE.pop(cache_key, None)
    _OBJ_MTL_VISUAL_PROXY_CACHE.pop(cache_key, None)
    _OBJ_DOUBLE_SIDED_PROXY_CACHE.pop(cache_key, None)
    _TEXTURE_CACHE.clear()

    split_root = os.path.join(
        os.path.dirname(model_path),
        "_split_by_mtl",
        _safe_token_name(os.path.splitext(os.path.basename(model_path))[0]),
    )
    if os.path.isdir(split_root):
        shutil.rmtree(split_root, ignore_errors=True)

    double_sided_root = os.path.join(
        os.path.dirname(model_path),
        "_double_sided",
    )
    if os.path.isdir(double_sided_root):
        shutil.rmtree(double_sided_root, ignore_errors=True)
