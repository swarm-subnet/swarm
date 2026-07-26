from ._shared import *


def _set_private_marker(model_fp: Path, is_private: bool) -> None:
    """Mark a stored model as private so the evaluator refuses to run a networked
    pip phase with the private bytes mounted."""
    marker = model_fp.with_suffix(".private")
    if is_private:
        marker.touch()
    else:
        marker.unlink(missing_ok=True)


def _admitted(path: Path, expected_family: str | None = None) -> bool:
    if is_model_graph_artifact(path):
        # a legacy graph artifact declares its family in the graph manifest and
        # is admitted against the ONNX rules by the runner, inside the sandbox
        accepted, detail = check_safety(path)
        declared_family = graph_declared_family
    else:
        accepted, detail = validate_submission_zip(path)
        declared_family = _declared_family

    if not accepted:
        bt.logging.error(f"Submission rejected: {detail}")
        return False
    if expected_family is None:
        return True

    declared = declared_family(path)
    if declared is not None and declared != expected_family:
        bt.logging.error(
            f"Submission declares family {declared!r}, expected {expected_family!r}"
        )
        return False
    return True


def _declared_family(path: Path) -> str | None:
    """Family named by the packaged policy contract, or None when absent.

    A hand-built zip may omit the contract; only a contract that disagrees with
    the commitment is a rejection."""
    try:
        return str(read_policy_contract_from_zip(path).get("family_id") or "") or None
    except PolicyInterfaceError:
        return None


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
    await verify_new_model_with_docker(dest, expected_hash, f"github-uid-{uid}", uid)
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
    await verify_new_model_with_docker(dest, model_hash, f"private-uid-{uid}", uid)
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
        if uid < 0 or not model_hash or not family_id:
            continue
        if interface_version not in RUNNABLE_INTERFACE_VERSIONS:
            continue
        if model_hash in load_blacklist():
            bt.logging.warning(
                f"Skipping blacklisted model {model_hash[:16]}... from UID {uid}"
            )
            continue
        if not is_private and (not github_url or not artifact_path):
            continue
        model_fp = MODEL_DIR / f"UID_{uid}.zip"
        try:
            if model_fp.is_file() and sha256sum(model_fp) == model_hash and _admitted(model_fp, family_id):
                _set_private_marker(model_fp, is_private)
                paths[uid] = (model_fp, github_url)
                continue
            model_fp.unlink(missing_ok=True)
            if is_private:
                ok = await _download_private_model(self, uid, model_hash, family_id, model_fp)
            else:
                ok = await _download_model_from_github(
                    github_url, artifact_path, model_hash, family_id, model_fp, uid
                )
            if ok and model_fp.is_file():
                _set_private_marker(model_fp, is_private)
                paths[uid] = (model_fp, github_url)
            else:
                _set_private_marker(model_fp, False)
                model_fp.unlink(missing_ok=True)
        except OSError as exc:
            bt.logging.warning(f"Model discovery failed for UID {uid}: {exc}")
            _set_private_marker(model_fp, False)
            model_fp.unlink(missing_ok=True)
    bt.logging.info(f"Backend discovery: {len(paths)} admitted submission(s)")
    return paths
