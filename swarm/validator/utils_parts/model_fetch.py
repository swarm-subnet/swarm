from ._shared import *
from .state import check_repo_ownership


async def _download_model_from_github(
    github_url: str, ref: PolicyRef, dest: Path, uid: int
) -> bool:
    """Download model ZIP from a miner's public GitHub repository."""
    validated = validate_github_url(github_url, uid=uid)
    if not validated:
        bt.logging.warning(f"UID {uid}: invalid github_url: {github_url}")
        return False

    cache_key = f"{validated}:{ref.sha256}"
    if cache_key not in _readme_ok_cache:
        if not await check_readme_matches(validated, uid=uid):
            bt.logging.warning(
                f"UID {uid}: README.md missing or does not match template, skipping"
            )
            return False
        _readme_ok_cache.add(cache_key)

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
    if downloaded_hash != ref.sha256:
        bt.logging.error(
            f"UID {uid}: SHA256 mismatch — "
            f"expected {ref.sha256[:16]}..., got {downloaded_hash[:16]}..."
        )
        dest.unlink(missing_ok=True)
        return False

    bt.logging.info(f"Stored model for UID {uid} from GitHub at {dest}")
    await verify_new_model_with_docker(dest, ref.sha256, f"github-uid-{uid}", uid)
    return True


async def _process_single_uid(self, uid: int) -> Tuple[int, Optional[Path], str]:
    """Fetch and verify a single miner's model."""
    try:
        axon = self.metagraph.axons[uid]

        try:
            responses = await send_with_fresh_uuid(
                wallet=self.wallet,
                synapse=PolicySynapse.request_ref(),
                axon=axon,
                timeout=QUERY_REF_TIMEOUT,
            )

            if not responses:
                return (uid, None, "")

            syn = responses[0]

            if not syn.ref:
                return (uid, None, "")

            ref = PolicyRef(**syn.ref)
        except Exception:
            return (uid, None, "")

        blacklist = load_blacklist()
        if ref.sha256 in blacklist:
            bt.logging.warning(f"Skipping blacklisted model {ref.sha256[:16]}... from UID {uid}")
            return (uid, None, "")

        model_fp = MODEL_DIR / f"UID_{uid}.zip"
        if model_fp.exists() and model_fp.is_dir():
            shutil.rmtree(model_fp)

        up_to_date = False
        if model_fp.is_file():
            try:
                up_to_date = sha256sum(model_fp) == ref.sha256
            except Exception:
                up_to_date = False

        if up_to_date:
            if (
                model_fp.stat().st_size <= MAX_MODEL_BYTES
                and zip_is_safe(model_fp, max_uncompressed=MAX_MODEL_BYTES)
            ):
                return (uid, model_fp, ref.github_url or "")
            else:
                model_fp.unlink(missing_ok=True)

        if not ref.github_url:
            bt.logging.warning(f"UID {uid}: no github_url in PolicyRef, skipping")
            return (uid, None, "")

        hotkey = self.metagraph.hotkeys[uid]
        if not check_repo_ownership(ref.github_url, hotkey, uid):
            return (uid, None, "")

        ok = await _download_model_from_github(ref.github_url, ref, model_fp, uid)
        if ok and model_fp.is_file():
            return (uid, model_fp, ref.github_url)
        else:
            model_fp.unlink(missing_ok=True)
            return (uid, None, "")

    except Exception:
        return (uid, None, "")


async def _ensure_models(self, uids: List[int]) -> Dict[int, Tuple[Path, str]]:
    """Fetch models from all given UIDs in parallel batches."""
    MODEL_DIR.mkdir(exist_ok=True)
    paths: Dict[int, Tuple[Path, str]] = {}
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONNECTIONS)
    total_batches = (len(uids) + PARALLEL_BATCH_SIZE - 1) // PARALLEL_BATCH_SIZE

    bt.logging.info(f"Starting model fetch for {len(uids)} UIDs in {total_batches} batches")

    async def _limited_process(uid: int) -> Tuple[int, Optional[Path], str]:
        async with semaphore:
            return await _process_single_uid(self, uid)

    for batch_start in range(0, len(uids), PARALLEL_BATCH_SIZE):
        batch = uids[batch_start:batch_start + PARALLEL_BATCH_SIZE]
        batch_num = batch_start // PARALLEL_BATCH_SIZE + 1

        results = await asyncio.gather(
            *[_limited_process(uid) for uid in batch],
            return_exceptions=True
        )

        batch_found = 0
        for result in results:
            if isinstance(result, Exception):
                continue
            uid, path, github_url = result
            if path is not None:
                paths[uid] = (path, github_url)
                batch_found += 1

        if batch_num % 5 == 0 or batch_found > 0:
            bt.logging.debug(
                f"Batch {batch_num}/{total_batches}: "
                f"found {batch_found} models, total so far: {len(paths)}"
            )

        if batch_start + PARALLEL_BATCH_SIZE < len(uids):
            await asyncio.sleep(BATCH_DELAY_SEC)

    bt.logging.info(f"Model fetch complete: found {len(paths)} models from {len(uids)} UIDs")
    return paths


# ──────────────────────────────────────────────────────────────────────────
# Benchmark evaluation
# ──────────────────────────────────────────────────────────────────────────
