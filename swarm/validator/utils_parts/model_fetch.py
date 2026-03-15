from ._shared import *


async def _download_model_from_github(
    github_url: str, expected_hash: str, dest: Path, uid: int
) -> bool:
    """Download model ZIP from GitHub and verify its hash."""
    validated = validate_github_url(github_url, uid=uid)
    if not validated:
        bt.logging.warning(f"UID {uid}: invalid github_url: {github_url}")
        return False

    candidate_urls = build_raw_urls(validated)
    downloaded = False
    for raw_url in candidate_urls:
        if await download_from_github(raw_url, dest, max_bytes=MAX_MODEL_BYTES):
            downloaded = True
            break

    if not downloaded:
        bt.logging.warning(f"UID {uid}: GitHub download failed from {validated}")
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
        github_url = str(entry.get("github_url", ""))

        if uid < 0 or not model_hash or not github_url:
            continue

        model_fp = MODEL_DIR / f"UID_{uid}.zip"

        if model_fp.is_file():
            try:
                if sha256sum(model_fp) == model_hash:
                    paths[uid] = (model_fp, github_url)
                    continue
            except Exception:
                pass
            model_fp.unlink(missing_ok=True)

        blacklist = load_blacklist()
        if model_hash in blacklist:
            bt.logging.warning(f"Skipping blacklisted model {model_hash[:16]}... from UID {uid}")
            continue

        ok = await _download_model_from_github(github_url, model_hash, model_fp, uid)
        if ok and model_fp.is_file():
            paths[uid] = (model_fp, github_url)
        else:
            model_fp.unlink(missing_ok=True)

    bt.logging.info(f"Backend discovery: {len(paths)} model(s) ready from {len(pending_models)} pending")
    return paths
