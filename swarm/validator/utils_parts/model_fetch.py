from ._shared import *


def _set_private_marker(model_fp: Path, is_private: bool) -> None:
    """Mark a stored model as private so the Docker evaluator refuses to run a
    networked pip phase with the private bytes mounted."""
    marker = model_fp.with_suffix(".private")
    if is_private:
        marker.touch()
    else:
        marker.unlink(missing_ok=True)


async def _download_model_from_github(
    github_url: str, expected_hash: str, dest: Path, uid: int
) -> bool:
    """Download model ZIP from GitHub and verify its hash."""
    validated = validate_github_url(github_url, uid=uid)
    if not validated:
        bt.logging.warning(f"UID {uid}: invalid github_url")
        return False

    candidate_urls = build_raw_urls(validated)
    downloaded = False
    for raw_url in candidate_urls:
        if await download_from_github(raw_url, dest, max_bytes=MAX_MODEL_BYTES):
            downloaded = True
            break

    if not downloaded:
        bt.logging.warning(f"UID {uid}: GitHub download failed")
        dest.unlink(missing_ok=True)
        return False

    if not zip_is_safe(dest, max_uncompressed=MAX_MODEL_BYTES):
        bt.logging.error(f"UID {uid}: unsafe ZIP from GitHub")
        dest.unlink(missing_ok=True)
        return False

    downloaded_hash = sha256sum(dest)
    if downloaded_hash != expected_hash:
        bt.logging.error(
            f"UID {uid}: SHA256 mismatch — "
            f"expected {expected_hash[:16]}..., got {downloaded_hash[:16]}..."
        )
        dest.unlink(missing_ok=True)
        return False

    bt.logging.info(f"Stored model for UID {uid} from GitHub at {dest}")
    await verify_new_model_with_docker(dest, expected_hash, f"github-uid-{uid}", uid)
    return True


async def _download_private_model(
    self, uid: int, model_hash: str, dest: Path
) -> bool:
    """Fetch a private model from the operator vault and verify its hash."""
    ok = await self.backend_api.fetch_private_artifact(model_hash, dest)
    if not ok or not dest.is_file():
        bt.logging.warning(f"UID {uid}: private artifact fetch failed")
        dest.unlink(missing_ok=True)
        return False

    if not zip_is_safe(dest, max_uncompressed=MAX_MODEL_BYTES):
        bt.logging.error(f"UID {uid}: unsafe private ZIP")
        dest.unlink(missing_ok=True)
        return False

    downloaded_hash = sha256sum(dest)
    if downloaded_hash != model_hash:
        bt.logging.error(
            f"UID {uid}: private SHA256 mismatch — "
            f"expected {model_hash[:16]}..., got {downloaded_hash[:16]}..."
        )
        dest.unlink(missing_ok=True)
        return False

    bt.logging.info(f"Stored private model for UID {uid} at {dest}")
    await verify_new_model_with_docker(dest, model_hash, f"private-uid-{uid}", uid)
    return True


async def _ensure_models_from_backend(
    self, pending_models: list[dict]
) -> Dict[int, Tuple[Path, str]]:
    """Download models reported by the backend sync endpoint.

    For each pending model, checks if we already have it locally.
    If not, downloads from GitHub and verifies the hash matches
    what the backend computed.
    """
    if not pending_models:
        return {}

    MODEL_DIR.mkdir(exist_ok=True)
    paths: Dict[int, Tuple[Path, str]] = {}

    for entry in pending_models:
        uid = int(entry.get("uid", -1))
        model_hash = str(entry.get("model_hash", ""))
        github_url = str(entry.get("github_url", "") or "")
        is_private = bool(entry.get("is_private"))

        if uid < 0 or not model_hash:
            continue
        if not is_private and not github_url:
            continue

        model_fp = MODEL_DIR / f"UID_{uid}.zip"

        if model_fp.is_file():
            try:
                if sha256sum(model_fp) == model_hash:
                    _set_private_marker(model_fp, is_private)
                    paths[uid] = (model_fp, github_url)
                    continue
            except Exception:
                pass
            model_fp.unlink(missing_ok=True)

        blacklist = load_blacklist()
        if model_hash in blacklist:
            bt.logging.warning(f"Skipping blacklisted model {model_hash[:16]}... from UID {uid}")
            continue

        if is_private:
            ok = await _download_private_model(self, uid, model_hash, model_fp)
        else:
            ok = await _download_model_from_github(github_url, model_hash, model_fp, uid)
        if ok and model_fp.is_file():
            _set_private_marker(model_fp, is_private)
            paths[uid] = (model_fp, github_url)
        else:
            model_fp.unlink(missing_ok=True)
            _set_private_marker(model_fp, False)

    bt.logging.info(f"Backend discovery: {len(paths)} model(s) ready from {len(pending_models)} pending")
    return paths
