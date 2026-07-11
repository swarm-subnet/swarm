from ._shared import *


def _admitted(path: Path, expected_family: str | None = None) -> bool:
    result = admit_artifact(path)
    if not result.accepted:
        bt.logging.error(f"Graph admission failed: {result.reason_code}: {result.detail}")
        return False
    return expected_family is None or result.family_id == expected_family


async def _download_model_from_github(
    github_url: str,
    artifact_path: str,
    expected_hash: str,
    expected_family: str,
    dest: Path,
    uid: int,
) -> bool:
    validated = validate_github_url(github_url, uid=uid)
    if not validated:
        return False
    try:
        candidates = build_raw_urls(validated, artifact_path)
    except ValueError:
        return False
    downloaded = False
    for url in candidates:
        if await download_from_github(url, dest, max_bytes=MAX_MODEL_BYTES):
            downloaded = True
            break
    if not downloaded:
        dest.unlink(missing_ok=True)
        return False
    if sha256sum(dest) != expected_hash or not _admitted(dest, expected_family):
        dest.unlink(missing_ok=True)
        return False
    return True


async def _download_private_model(
    self, uid: int, model_hash: str, expected_family: str, dest: Path
) -> bool:
    ok = await self.backend_api.fetch_private_artifact(model_hash, dest)
    if not ok or not dest.is_file():
        dest.unlink(missing_ok=True)
        return False
    if sha256sum(dest) != model_hash or not _admitted(dest, expected_family):
        dest.unlink(missing_ok=True)
        return False
    return True


async def _ensure_models_from_backend(
    self, pending_models: list[dict]
) -> Dict[int, Tuple[Path, str]]:
    if not pending_models:
        return {}
    MODEL_DIR.mkdir(exist_ok=True)
    paths: Dict[int, Tuple[Path, str]] = {}
    for entry in pending_models:
        uid = int(entry.get("uid", -1))
        model_hash = str(entry.get("model_hash", ""))
        family_id = str(entry.get("family_id", ""))
        interface_version = str(entry.get("interface_version", ""))
        github_url = str(entry.get("github_url", "") or "")
        artifact_path = str(entry.get("artifact_path", "") or "")
        is_private = bool(entry.get("is_private"))
        if uid < 0 or not model_hash or not family_id or interface_version != "model_graph.v1":
            continue
        if not is_private and (not github_url or not artifact_path):
            continue
        model_fp = MODEL_DIR / f"UID_{uid}.zip"
        try:
            if model_fp.is_file() and sha256sum(model_fp) == model_hash and _admitted(model_fp, family_id):
                paths[uid] = (model_fp, github_url)
                continue
            model_fp.unlink(missing_ok=True)
            if is_private:
                ok = await _download_private_model(self, uid, model_hash, family_id, model_fp)
            else:
                ok = await _download_model_from_github(
                    github_url, artifact_path, model_hash, family_id, model_fp, uid
                )
            if ok:
                paths[uid] = (model_fp, github_url)
        except OSError as exc:
            bt.logging.warning(f"Model discovery failed for UID {uid}: {exc}")
            model_fp.unlink(missing_ok=True)
    bt.logging.info(f"Backend discovery: {len(paths)} admitted graph artifact(s)")
    return paths
